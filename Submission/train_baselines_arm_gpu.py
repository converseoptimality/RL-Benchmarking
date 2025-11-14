# train_baselines_arm_gpu.py
import argparse, os, time, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF INFO/WARN

import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

# IMPORTANT: importing this registers "ConverseArm-v0"
import converse_arm_env  # noqa: F401

ENV_ID = "ConverseArm-v0"
ALGOS = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "TD3": TD3, "DDPG": DDPG}

# ----------------------- helpers -----------------------
def make_run_name(args):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fix = "fixs0" if args.fixed_init else "varys0"
    return (f"{args.algo}"
            f"_n{args.n_links}_p{args.p}_tau{args.tau}_ru{args.r_u}"
            f"_sigw{args.sigma_omega}_seed{args.seed}_{fix}_{stamp}")

def make_env_single(args, seed, log_dir=None, monitor=True, idx=0, eval_mode=False):
    """Factory that returns ONE env (wrapped with Monitor if asked)."""
    def _thunk():
        env = gym.make(
            ENV_ID,
            n_links=args.n_links,
            gamma=args.gamma,
            r_u=args.r_u,
            tau=args.tau,
            sigma_theta=args.sigma_theta,
            sigma_omega=args.sigma_omega,
            p=args.p,
            horizon=args.horizon,
            act_limit=args.act_limit,
            grav0=args.grav0,
            swirl_gain=args.swirl_gain,
            grav_always_on=args.grav_always_on,
            render_mode=None,
            # --- FAIRNESS flags ---
            fixed_init=args.fixed_init,
            reset_seed_per_episode=True,
            seed=seed + idx,  # defines the family + episode sequence (eval uses +10000)
        )
        if monitor and (log_dir is not None):
            mon_file = os.path.join(log_dir, f"monitor-{idx}.csv")
            env = Monitor(env, filename=mon_file)
        return env
    return _thunk

def make_vec_env(args, seed, log_dir=None, monitor=True, eval_mode=False):
    """Create a (Subproc|Dummy)VecEnv of size args.num_envs."""
    fns = [make_env_single(args, seed, log_dir=log_dir, monitor=monitor, idx=i, eval_mode=eval_mode)
           for i in range(args.num_envs)]
    if args.num_envs > 1 and not eval_mode:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)

def vec_get_attr(vecenv, name, index=0):
    # unchanged (your robust getter) ...
    try:
        val = vecenv.get_attr(name, indices=index)
        if isinstance(val, list):
            return val[0]
        return val
    except Exception:
        pass
    try:
        inner = getattr(vecenv, "venv", None)
        if inner is not None:
            val = inner.get_attr(name, indices=index)
            if isinstance(val, list):
                return val[0]
            return val
    except Exception:
        pass
    try:
        return getattr(vecenv.envs[index].unwrapped, name)
    except Exception as e:
        raise RuntimeError(f"Could not fetch attribute '{name}' from vec env: {e}")

# ----------------------- callbacks -----------------------
class TensorboardCostAndValueCallback(BaseCallback):
    # unchanged (your implementation) ...
    def __init__(self, gamma: float, verbose=0):
        super().__init__(verbose)
        self.gamma = float(gamma)
        self._fam = None
        self._prev_V = None
    @staticmethod
    def _V_of(fam, s_vec: np.ndarray) -> float:
        return float(s_vec.T @ fam.P @ s_vec + fam.c_cost)
    def _on_training_start(self) -> None:
        fam = vec_get_attr(self.training_env, "fam", index=0)
        self._fam = fam
        self._prev_V = None
    def _on_rollout_start(self) -> None:
        self._prev_V = None
    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs", None)
        rewards = self.locals.get("rewards", None)
        dones   = self.locals.get("dones", None)
        if new_obs is None or rewards is None:
            return True
        obs_next = np.array(new_obs[0], dtype=float)
        reward   = float(rewards[0])
        cost_k   = -reward
        V_next = self._V_of(self._fam, obs_next)
        self.logger.record("train/stage_cost", cost_k)
        self.logger.record("train/V_true", V_next)
        if self._prev_V is not None:
            delta = self._prev_V - (cost_k + self.gamma * V_next)
            self.logger.record("train/bellman_cost_residual", float(delta))
            self.logger.record("train/abs_bellman_cost_residual", float(abs(delta)))
        done0 = bool(dones[0]) if dones is not None else False
        self._prev_V = None if done0 else V_next
        return True

# ----------------------- main -----------------------
def main():
    from stable_baselines3.common.utils import set_random_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), default="PPO")
    ap.add_argument("--total_steps", type=int, default=350_000)
    ap.add_argument("--gamma", type=float, default=0.99)

    ap.add_argument("--n_links", type=int, default=6)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--r_u", type=float, default=2.0)
    ap.add_argument("--sigma_theta", type=float, default=0.0)
    ap.add_argument("--sigma_omega", type=float, default=1e-4)
    ap.add_argument("--p", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=512)
    ap.add_argument("--act_limit", type=float, default=2.0)
    ap.add_argument("--grav0", type=float, default=0.8)
    ap.add_argument("--swirl_gain", type=float, default=0.5)
    ap.add_argument("--grav_always_on", action="store_true")

    # FAIRNESS switches
    ap.add_argument("--fixed_init", action="store_true",
                    help="If set, every episode starts from the same s0 (strongest control). Default: same sequence across algos.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--normalize", action="store_true",
                    help="Use VecNormalize(obs); reward left unnormalized.")

    # logging / saving
    ap.add_argument("--log_root", type=str, default="runs_2")
    ap.add_argument("--eval_every", type=int, default=10_000)   # steps
    ap.add_argument("--save_checkpoints", action="store_true",
                    help="If set, also save periodic checkpoints. Default: OFF (save best only).")

    # device control
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="auto: SAC/TD3/DDPG->cuda if available; PPO/A2C->cpu")

    # optional net sizes: "256,256" etc.
    ap.add_argument("--policy_net", type=str, default="256,256")
    # off-policy knobs
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--buffer_size", type=int, default=100000)
    ap.add_argument("--learning_starts", type=int, default=10_000)
    ap.add_argument("--train_freq", type=int, default=64)
    ap.add_argument("--gradient_steps", type=int, default=64)
    ap.add_argument("--action_noise_sigma", type=float, default=0.10)

    args = ap.parse_args()

    # reproducibility across workers
    set_random_seed(args.seed)

    # device policy
    def pick_device(algo: str, dev_flag: str) -> str:
        if dev_flag in ("cpu", "cuda"):
            return dev_flag
        if algo.upper() in ("SAC", "TD3", "DDPG") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device = pick_device(args.algo, args.device)
    print(f"[device] algo={args.algo} -> using device={device}")

    # ------------- run dirs -------------
    run_name = make_run_name(args)
    run_dir  = os.path.join(args.log_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_dir   = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------- envs -------------
    train_env = make_vec_env(args, args.seed, log_dir=run_dir, monitor=True, eval_mode=False)
    eval_env  = make_vec_env(args, args.seed + 10_000, log_dir=None, monitor=False, eval_mode=True)

    # optional normalization (obs only; reward left raw so -reward = stage_cost)
    if args.normalize:
        train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env  = VecNormalize(eval_env,  training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = train_env.obs_rms

    # ------------- per-algo policy kwargs -------------
    hidden = [int(x) for x in args.policy_net.split(",") if x]
    policy_kwargs_onpolicy = dict(net_arch=dict(pi=hidden, vf=hidden))
    policy_kwargs_offpolicy = dict(net_arch=hidden)

    # ------------- model -------------
    Algo = ALGOS[args.algo]
    if args.algo in ("PPO", "A2C"):
        common_kwargs = dict(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            gamma=args.gamma,
            tensorboard_log=tb_dir,
            device=device,
            policy_kwargs=policy_kwargs_onpolicy,
        )
        if args.algo == "PPO":
            model = Algo(**common_kwargs,
                         n_steps=max(8, 2048 // max(1, args.num_envs)),
                         batch_size=64,
                         gae_lambda=0.95,
                         clip_range=0.2,
                         ent_coef=0.0001,
                         learning_rate=3e-4)
        else:  # A2C
            model = Algo(**common_kwargs,
                         n_steps=32,
                         learning_rate=7e-4,
                         ent_coef=0.0001,
                         vf_coef=0.5,
                         max_grad_norm=0.5)

    elif args.algo in ("SAC", "TD3", "DDPG"):
        common_kwargs = dict(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            gamma=args.gamma,
            tensorboard_log=tb_dir,
            device=device,
            policy_kwargs=policy_kwargs_offpolicy,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            learning_rate=3e-4,
        )
        if args.algo == "SAC":
            model = Algo(**common_kwargs)
        else:
            act_limit = float(args.act_limit)
            sigma = args.action_noise_sigma * act_limit
            action_noise = NormalActionNoise(mean=np.zeros(args.n_links), sigma=sigma * np.ones(args.n_links))
            model = Algo(**common_kwargs, action_noise=action_noise)
    else:
        raise ValueError(f"Unknown algo {args.algo}")

    # --------- Save family + meta for airtight analysis ----------
    try:
        fam = vec_get_attr(train_env, "fam", index=0)
        np.savez(os.path.join(run_dir, "family.npz"),
                 P=fam.P, Q=fam.Q, R=fam.R, Sigma=fam.Sigma,
                 gamma=fam.gamma, c_cost=fam.c_cost,
                 tau=float(np.trace(fam.Q)/np.trace(fam.P)))
    except Exception as e:
        print(f"[warn] could not save family.npz: {e}")

    # Save env kwargs / meta
    meta = dict(
        algo=args.algo, seed=args.seed, n_links=args.n_links, p=args.p, tau=args.tau, r_u=args.r_u,
        sigma_theta=args.sigma_theta, sigma_omega=args.sigma_omega, gamma=args.gamma,
        horizon=args.horizon, act_limit=args.act_limit,
        grav0=args.grav0, swirl_gain=args.swirl_gain, grav_always_on=bool(args.grav_always_on),
        fixed_init=bool(args.fixed_init), normalize=bool(args.normalize)
    )
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ------------- callbacks -------------
    tb_cost_cb = TensorboardCostAndValueCallback(gamma=args.gamma)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=args.eval_every,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    cblist = [tb_cost_cb, eval_cb]

    if args.save_checkpoints:
        ckpt_cb = CheckpointCallback(
            save_freq=max(1, args.eval_every),  # piggyback on eval cadence
            save_path=ckpt_dir,
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        cblist.append(ckpt_cb)

    # ------------- learn -------------
    model.learn(total_timesteps=args.total_steps, callback=cblist)

    # Save final model (best is already under ckpt/)
    model.save(os.path.join(run_dir, "final_model"))
    if args.normalize:
        try:
            train_env.save(os.path.join(run_dir, "vecnorm.pkl"))
        except Exception:
            pass

    train_env.close(); eval_env.close()
    print(f"\nAll logs and models saved under: {run_dir}")
    print(f"TensorBoard:  tensorboard --logdir {args.log_root}\n")

if __name__ == "__main__":
    main()