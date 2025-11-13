"""
train_all.py

Train PPO, A2C and SAC on your N-crank system using the same environment,
your custom PPO hyperparameters, and the symbolic cost c(s,u) as reward.

Usage:
    python train_all.py --algo ppo        # just PPO
    python train_all.py --run_both        # PPO, A2C and SAC in sequence
    python train_all.py --algo sac --timesteps 500000
"""
import os
import argparse
import datetime

import numpy as np
import sympy as sp

from gymnasium import Env, spaces
from gymnasium.spaces import Box
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from crank_system import (N, xs, ys, us, state, f, G, c, params)
from systems_m    import GenericODE
from cost_gym_env import CostGymEnv


# ─── 2) BUILD GENERICODE from symbolic f, G ───────────────────────────────
def build_system(param_subs):
    f_num = f.subs(param_subs)
    G_num = G.subs(param_subs)
    state_vars  = sp.Matrix(state)
    action_vars = sp.Matrix(us)
    rhs_sym  = f_num + G_num * action_vars
    rhs_fun  = sp.lambdify([state_vars, action_vars], rhs_sym, "numpy")
    return GenericODE(
        name               = f"{N}Crank",
        dim_state          = 2 * N,
        dim_inputs         = N,
        dim_observation    = 2 * N,
        observation_naming = [str(s) for s in state_vars],
        state_naming       = [str(s) for s in state_vars],
        inputs_naming      = [str(u_) for u_ in us],
        action_bounds      = [[-100,100]] * N,
        rhs                = rhs_sym,
        rhs_func           = lambda s,u: np.asarray(rhs_fun(s,u),dtype=np.float32).flatten(),
    )


# ─── 3) ARGUMENTS ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["ppo","a2c","sac"], default="PPO")
parser.add_argument("--run_both", action="store_true",
                    help="train PPO, A2C and SAC in sequence")
parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--dt",        type=float, default=0.01)
parser.add_argument("--max_episode_steps", type=int, default=700)
args = parser.parse_args()


# ─── 4) SAMPLE PARAMETERS & LAMBDIFY COST ────────────────────────────────
params   = params[0]
system   = build_system(params)

# symbolic cost → numeric
state_syms  = list(sp.Matrix(state))
action_syms = list(sp.Matrix(us))
c_sym       = c.subs(params)
cost_fun    = sp.lambdify([state_syms, action_syms], c_sym, "numpy")



# ─── 5) ENV FACTORY ───────────────────────────────────────────────────────
def make_env():
    return Monitor(
        CostGymEnv(
            system,
            #cost_fun=cost_fun,
            #dt=args.dt,
            max_time_steps=args.max_episode_steps
        ),
        filename=None,
        allow_early_resets=True
    )


# ─── 6) TRAINING LOOP ─────────────────────────────────────────────────────
algorithms = ["ppo","a2c","sac"] if args.run_both else [args.algo]
algorithms = ["sac"]
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)
print(algorithms)

for algo in algorithms:
    # pick class & hyperparams
    print(args.timesteps)
    if algo == "ppo":
        AlgoCls = PPO
        hyper = dict(
            learning_rate = 0.001,
            n_steps       = 2048,
            batch_size    = 64,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            max_grad_norm = 0.3,
        )
    elif algo == "a2c":
        AlgoCls = A2C
        hyper = dict(
            learning_rate = 7e-4,
            n_steps       = 5,
            gamma         = 0.99,
            gae_lambda    = 1.0,
            ent_coef      = 0.01,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
        )
    else:  # sac
        args.timesteps=500_000
        print(args.timesteps)
        print(args.timesteps)
        print(args.timesteps)
        print(args.timesteps)

        AlgoCls = SAC
        hyper = dict(
            
            learning_rate   = 1e-3,
            buffer_size     = 1400,
            learning_starts = 100,
            batch_size      = 64,
            tau             = 0.005,
            gamma           = 0.99,
            train_freq      = 1,
            gradient_steps  = 1,
            ent_coef        = "auto",
        )

    # vectored & normalized env
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    

    # build, train & save
    tag    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join("logs_cost", f"{algo}_{tag}")
    os.makedirs(logdir, exist_ok=True)

    model = AlgoCls(
        "MlpPolicy",
        vec_env,
        verbose         = 1,
        tensorboard_log = logdir,
        **hyper
    )
    model.learn(total_timesteps=args.timesteps)

    os.makedirs("crank_models", exist_ok=True)
    fname = f"{algo}_{tag}_dt_{args.dt:.3f}"
    model.save(os.path.join("crank_models", fname))

    print(f"[✓] {algo.upper()} done; logs→{logdir}, model→crank_models/{fname}.zip")