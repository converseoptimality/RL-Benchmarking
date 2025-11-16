# train_baselines_nudex.py
# NUDEx benchmarks with reward-based logging, robust CRN metrics, and TD3 arg fix.
import argparse, os, time, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

# registers NUDEx-v0
import nudex_env  # noqa: F401

ENV_ID = "NUDEx-v0"
ALGOS = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "TD3": TD3, "DDPG": DDPG}

# ----------------------- helpers -----------------------
def make_run_name(args):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fix = "fixs0" if args.fixed_init else "varys0"
    return (f"{args.algo}_K{args.K}_rv{args.r_v}_rw{args.r_w}_p{args.p}"
            f"_rho{args.rho_target}_tau{args.tau}_seed{args.seed}_{fix}_{stamp}")

def make_env_single(args, seed, log_dir=None, monitor=True, idx=0, eval_mode=False):
    def _thunk():
        env = gym.make(
            ENV_ID,
            # structure
            K=args.K, r_v=args.r_v, r_w=args.r_w, tau=args.tau,
            # discount/control
            gamma=args.gamma, p=args.p,
            # spectra
            weight_pose=args.weight_pose, weight_v=args.weight_v, weight_w=args.weight_w,
            r_u_v=args.r_u_v, r_u_w=args.r_u_w,
            # noise
            sigma_pose=args.sigma_pose, sigma_v=args.sigma_v, sigma_w=args.sigma_w,
            # field
            alpha=args.alpha, kappa=args.kappa,
            # instability
            rho_target=args.rho_target, allow_R_rescale=not args.no_rescale,
            R_scale=args.R_scale, max_rescale_iter=args.max_rescale_iter,
            # gym/fairness
            horizon=args.horizon, act_limit=args.act_limit,
            obs_clip=args.obs_clip,
            fixed_init=args.fixed_init, reset_seed_per_episode=True,
            seed=seed + idx, render_mode=None
        )
        if monitor and (log_dir is not None):
            mon_file = os.path.join(log_dir, f"monitor-{idx}.csv")
            env = Monitor(env, filename=mon_file)
        return env
    return _thunk

def make_vec_env(args, seed, log_dir=None, monitor=True, eval_mode=False):
    fns = [make_env_single(args, seed, log_dir=log_dir, monitor=monitor, idx=i, eval_mode=eval_mode)
           for i in range(args.num_envs)]
    if args.num_envs > 1 and not eval_mode:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)

def vec_get_attr(vecenv, name, index=0):
    try:
        val = vecenv.get_attr(name, indices=index)
        return val[0] if isinstance(val, list) else val
    except Exception:
        pass
    try:
        inner = getattr(vecenv, "venv", None)
        if inner is not None:
            val = inner.get_attr(name, indices=index)
            return val[0] if isinstance(val, list) else val
    except Exception:
        pass
    return getattr(vecenv.envs[index].unwrapped, name)

# ----------------------- callbacks -----------------------
class TensorboardCostAndValueCallback(BaseCallback):
    """Logs stage_reward (primary), stage_cost (sanity), V* and Bellman residuals."""
    def __init__(self, gamma: float, verbose=0):
        super().__init__(verbose)
        self.gamma = float(gamma)
        self._fam = None
        self._prev_V = None

    @staticmethod
    def _V_of(fam, s_vec: np.ndarray) -> float:
        return float(s_vec.T @ fam.P @ s_vec + fam.c_cost)

    def _on_training_start(self) -> None:
        self._fam = vec_get_attr(self.training_env, "fam", index=0)
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
        reward_k = float(rewards[0])
        cost_k   = -reward_k
        V_next = self._V_of(self._fam, obs_next)
        self.logger.record("train/stage_reward", reward_k)
        self.logger.record("train/stage_cost",   cost_k)
        self.logger.record("train/V_true_reward", -V_next)
        self.logger.record("train/V_true_cost",    V_next)
        if self._prev_V is not None:
            delta = self._prev_V - (cost_k + self.gamma * V_next)
            self.logger.record("train/bellman_cost_residual", float(delta))
            self.logger.record("train/abs_bellman_cost_residual", float(abs(delta)))
            self.logger.record("train/bellman_reward_residual", float(-delta))
            self.logger.record("train/abs_bellman_reward_residual", float(abs(-delta)))
        done0 = bool(dones[0]) if dones is not None else False
        self._prev_V = None if done0 else V_next
        return True

class CRNMetricsCallback(BaseCallback):
    """OptGap/Regret under CRN, works with (VecEnv of size 1) or raw env."""
    def __init__(self, eval_env, gamma: float, eval_freq: int = 10_000, n_eval_episodes: int = 5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.gamma = float(gamma)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)

    @staticmethod
    def _oracle_action(fam, s: np.ndarray) -> np.ndarray:
        if hasattr(fam, "a_star_p"):
            try:
                return fam.a_star_p(s, fam.p)
            except TypeError:
                return fam.a_star_p(s)
        if hasattr(fam, "a_star"):
            return fam.a_star(s)
        raise AttributeError("Family has no oracle action method a_star_p/a_star")

    def _oracle_reward(self, fam, s: np.ndarray) -> float:
        a_star = self._oracle_action(fam, s)
        return -float(s.T @ fam.Q @ s + a_star.T @ fam.R @ a_star)

    def _reset_vec(self):
        out = self.eval_env.reset()
        if isinstance(out, tuple):
            out = out[0]
        return out if getattr(out, "ndim", 1) > 1 else out[None, :]

    def _step_vec(self, action):
        try:
            obs, r, dones, infos = self.eval_env.step(action)
            r0 = float(r[0]) if np.ndim(r) else float(r)
            done0 = bool(dones[0]) if np.ndim(dones) else bool(dones)
            return obs, r0, done0
        except Exception:
            obs, r, term, trunc, _ = self.eval_env.step(action[0] if np.ndim(action) == 2 else action)
            obs_batched = obs if getattr(obs, "ndim", 1) > 1 else obs[None, :]
            return obs_batched, float(r), bool(term or trunc)

    def _evaluate_once(self):
        fam = vec_get_attr(self.eval_env, "fam", index=0)
        returns, optgaps, regrets = [], [], []
        for _ in range(self.n_eval_episodes):
            obs = self._reset_vec()
            s = np.array(obs[0], dtype=float)
            V_star0_cost = float(s.T @ fam.P @ s + fam.c_cost)
            V_star0_rew  = -V_star0_cost
            Gt, ret, reg = 1.0, 0.0, 0.0
            horizon = int(vec_get_attr(self.eval_env, "horizon", 0))
            for _t in range(horizon):
                a, _ = self.model.predict(obs, deterministic=True)
                a_in = a if (getattr(a, "ndim", 1) > 1) else a[None, :]
                r_star = self._oracle_reward(fam, s)
                obs, r, done = self._step_vec(a_in)
                ret += Gt * r
                reg += Gt * (r - r_star)
                s = np.array(obs[0], dtype=float)
                Gt *= self.gamma
                if done:
                    break
            eps = 1e-9
            optgap = (ret - V_star0_rew) / (abs(V_star0_rew) + eps)
            returns.append(ret); regrets.append(reg); optgaps.append(optgap)
        self.logger.record("eval/return", float(np.mean(returns)))
        self.logger.record("eval/opt_gap", float(np.mean(optgaps)))
        self.logger.record("eval/regret", float(np.mean(regrets)))
        self.logger.record("eval/episodes", self.n_eval_episodes)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            self._evaluate_once()
        return True

# ----------------------- main -----------------------
def main():
    from stable_baselines3.common.utils import set_random_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), default="PPO")
    ap.add_argument("--total_steps", type=int, default=350_000)
    ap.add_argument("--gamma", type=float, default=0.99)

    # structure / dynamics
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--r_v", type=int, default=1)
    ap.add_argument("--r_w", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--p", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=700)
    ap.add_argument("--act_limit", type=float, default=2.0)
    ap.add_argument("--obs_clip", type=float, default=None)

    # spectra and field
    ap.add_argument("--weight_pose", type=float, default=2.0)
    ap.add_argument("--weight_v", type=float, default=1.0)
    ap.add_argument("--weight_w", type=float, default=1.0)
    ap.add_argument("--r_u_v", type=float, default=0.4)
    ap.add_argument("--r_u_w", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--kappa", type=float, default=0.5)

    # noise
    ap.add_argument("--sigma_pose", type=float, default=1e-6)
    ap.add_argument("--sigma_v", type=float, default=1e-3)
    ap.add_argument("--sigma_w", type=float, default=1e-3)

    # instability
    ap.add_argument("--rho_target", type=float, default=1.04)
    ap.add_argument("--no_rescale", action="store_true")
    ap.add_argument("--R_scale", type=float, default=0.6)
    ap.add_argument("--max_rescale_iter", type=int, default=12)

    # fairness / seeds / vec
    ap.add_argument("--fixed_init", dest="fixed_init", action="store_true")
    ap.add_argument("--vary_init",  dest="fixed_init", action="store_false")
    ap.set_defaults(fixed_init=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--normalize", action="store_true")

    # logging / saving
    ap.add_argument("--log_root", type=str, default="runs_Nudex_1")
    ap.add_argument("--eval_every", type=int, default=10_000)
    ap.add_argument("--metrics_every", type=int, default=10_000)
    ap.add_argument("--n_eval_episodes", type=int, default=5)
    ap.add_argument("--save_checkpoints", action="store_true")
    ap.add_argument("--save_best", action="store_true")

    # device
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # nets / off-policy knobs
    ap.add_argument("--policy_net", type=str, default="256,256")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--buffer_size", type=int, default=100000)
    ap.add_argument("--learning_starts", type=int, default=10_000)
    ap.add_argument("--train_freq", type=int, default=64)
    ap.add_argument("--gradient_steps", type=int, default=64)
    ap.add_argument("--action_noise_sigma", type=float, default=0.10)

    # PPO overrides
    ap.add_argument("--ppo_n_steps", type=int, default=None)
    ap.add_argument("--ppo_batch_size", type=int, default=None)
    ap.add_argument("--ppo_n_epochs", type=int, default=None)
    ap.add_argument("--ppo_lr", type=float, default=None)
    ap.add_argument("--ppo_ent_coef", type=float, default=None)
    ap.add_argument("--ppo_clip_range", type=float, default=None)

    # A2C overrides
    ap.add_argument("--a2c_n_steps", type=int, default=None)
    ap.add_argument("--a2c_lr", type=float, default=None)
    ap.add_argument("--a2c_ent_coef", type=float, default=None)

    # Off-policy shared overrides (SAC/TD3/DDPG)
    ap.add_argument("--off_batch_size", type=int, default=None)
    ap.add_argument("--off_buffer_size", type=int, default=None)
    ap.add_argument("--off_learning_starts", type=int, default=None)
    ap.add_argument("--off_train_freq", type=int, default=None)
    ap.add_argument("--off_gradient_steps", type=int, default=None)
    ap.add_argument("--off_lr", type=float, default=None)
    ap.add_argument("--off_tau", type=float, default=None)

    # SAC-specific
    ap.add_argument("--sac_ent_coef", type=str, default=None)  # e.g., "auto" or "0.1"

    # TD3-specific (map your runner's flags)
    ap.add_argument("--td3_policy_delay", type=int, default=None)
    ap.add_argument("--td3_target_noise", type=float, default=None)  # -> target_policy_noise
    ap.add_argument("--td3_target_clip", type=float, default=None)   # -> target_noise_clip

    args = ap.parse_args()
    set_random_seed(args.seed)

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
    tb_dir   = os.path.join(run_dir, "tb"); os.makedirs(tb_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt"); os.makedirs(ckpt_dir, exist_ok=True)

    # ------------- envs -------------
    train_env = make_vec_env(args, args.seed, log_dir=run_dir, monitor=True, eval_mode=False)
    eval_env  = make_vec_env(args, args.seed + 10_000, log_dir=None, monitor=True, eval_mode=True)
    eval_env_crn = make_vec_env(args, args.seed + 20_000, log_dir=None, monitor=True, eval_mode=True)

    if args.normalize:
        train_env = VecNormalize(train_env, training=True,  norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env  = VecNormalize(eval_env,  training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env_crn = VecNormalize(eval_env_crn, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = train_env.obs_rms
        eval_env_crn.obs_rms = train_env.obs_rms

    # ------------- per-algo policy kwargs -------------
    hidden = [int(x) for x in args.policy_net.split(",") if x]
    policy_kwargs_on = dict(net_arch=dict(pi=hidden, vf=hidden))
    policy_kwargs_off = dict(net_arch=hidden)

    # ------------- model -------------
    Algo = ALGOS[args.algo]

    if args.algo in ("PPO", "A2C"):
        common = dict(policy="MlpPolicy", env=train_env, verbose=1, gamma=args.gamma,
                      tensorboard_log=tb_dir, device=device, policy_kwargs=policy_kwargs_on)
        if args.algo == "PPO":
            n_steps    = args.ppo_n_steps    or max(8, 2048 // max(1, args.num_envs))
            batch_size = args.ppo_batch_size or 64
            n_epochs   = args.ppo_n_epochs   or 10
            lr         = args.ppo_lr         or 3e-4
            ent_coef   = args.ppo_ent_coef   or 1e-4
            clip_range = args.ppo_clip_range or 0.2
            model = PPO(**common, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                        gae_lambda=0.95, clip_range=clip_range, ent_coef=ent_coef, learning_rate=lr)
        else:
            n_steps  = args.a2c_n_steps  or 32
            lr       = args.a2c_lr       or 7e-4
            ent_coef = args.a2c_ent_coef or 1e-4
            model = A2C(**common, n_steps=n_steps, learning_rate=lr, ent_coef=ent_coef,
                        vf_coef=0.5, max_grad_norm=0.5)

    elif args.algo in ("SAC", "TD3", "DDPG"):
        off_lr             = args.off_lr             or 3e-4
        off_batch_size     = args.off_batch_size     or args.batch_size
        off_buffer_size    = args.off_buffer_size    or args.buffer_size
        off_learning_starts= args.off_learning_starts or args.learning_starts
        off_train_freq     = args.off_train_freq     or args.train_freq
        off_gradient_steps = args.off_gradient_steps or args.gradient_steps
        off_tau            = args.off_tau            or 0.005

        common = dict(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            gamma=args.gamma,
            tensorboard_log=tb_dir,
            device=device,
            policy_kwargs=policy_kwargs_off,
            batch_size=off_batch_size,
            buffer_size=off_buffer_size,
            learning_starts=off_learning_starts,
            train_freq=off_train_freq,
            gradient_steps=off_gradient_steps,
            learning_rate=off_lr,
            tau=off_tau,
        )

        if args.algo == "SAC":
            ent_coef = args.sac_ent_coef if args.sac_ent_coef is not None else "auto"
            model = SAC(**common, ent_coef=ent_coef)

        elif args.algo == "TD3":
            act_space = vec_get_attr(train_env, "action_space", 0)
            act_dim = act_space.shape[0]
            sigma = args.action_noise_sigma * float(args.act_limit)
            action_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=sigma*np.ones(act_dim))

            policy_delay       = args.td3_policy_delay if args.td3_policy_delay is not None else 2
            target_policy_noise= args.td3_target_noise if args.td3_target_noise is not None else 0.2
            target_noise_clip  = args.td3_target_clip  if args.td3_target_clip  is not None else 0.5

            model = TD3(**common,
                        action_noise=action_noise,
                        policy_delay=policy_delay,
                        target_policy_noise=target_policy_noise,
                        target_noise_clip=target_noise_clip)

        else:  # DDPG
            act_space = vec_get_attr(train_env, "action_space", 0)
            act_dim = act_space.shape[0]
            sigma = args.action_noise_sigma * float(args.act_limit)
            action_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=sigma*np.ones(act_dim))
            model = DDPG(**common, action_noise=action_noise)

    else:
        raise ValueError(f"Unknown algo {args.algo}")

    # --------- Save family + meta ----------
    try:
        fam = vec_get_attr(train_env, "fam", index=0)
        np.savez(os.path.join(run_dir, "family.npz"),
                 P=fam.P, Q=fam.Q, R=fam.R, Sigma=fam.Sigma,
                 gamma=fam.gamma, c_cost=fam.c_cost,
                 K=args.K, r_v=args.r_v, r_w=args.r_w)
    except Exception as e:
        print(f"[warn] could not save family.npz: {e}")

    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------- callbacks -------------
    tb_cost_cb = TensorboardCostAndValueCallback(gamma=args.gamma)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=(ckpt_dir if args.save_best else None),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    crn_metrics_cb = CRNMetricsCallback(
        eval_env=eval_env_crn, gamma=args.gamma,
        eval_freq=args.metrics_every, n_eval_episodes=args.n_eval_episodes
    )
    cbs = [tb_cost_cb, eval_cb, crn_metrics_cb]

    if args.save_checkpoints:
        ckpt_cb = CheckpointCallback(
            save_freq=max(1, args.eval_every),
            save_path=ckpt_dir,
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        cbs.append(ckpt_cb)

    # ------------- learn -------------
    model.learn(total_timesteps=args.total_steps, callback=cbs)

    model.save(os.path.join(run_dir, "final_model"))
    if args.normalize:
        try:
            train_env.save(os.path.join(run_dir, "vecnorm.pkl"))
        except Exception:
            pass

    train_env.close(); eval_env.close(); eval_env_crn.close()
    print(f"\nAll logs and models saved under: {run_dir}")
    print(f"TensorBoard:  tensorboard --logdir {args.log_root}\n")

if __name__ == "__main__":
    main()
