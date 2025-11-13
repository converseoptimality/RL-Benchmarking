#!/usr/bin/env python3
import os, argparse, datetime
import numpy as np
import sympy as sp

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env   import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor   import Monitor

from crank_system import (  # your existing symbolic file
    N, xs, ys, us, Ks,       # N and symbol arrays
    state, V,                 # state symbols & Lyapunov V (if you need)
    f, G,                     # symbolic drift & control matrices
    c,                        # symbolic running cost
    gen_params                # your parameter sampler
)
from systems_m   import GenericODE
from gym_env_c   import CustomGymEnv

def build_system(param_subs):
    """Instantiate GenericODE with numeric RHS from symbolic f, G."""
    f_num = f.subs(param_subs)
    G_num = G.subs(param_subs)
    state_vars  = sp.Matrix(state)
    action_vars = sp.Matrix(us)
    rhs_sym  = f_num + G_num*action_vars
    rhs_fun  = sp.lambdify([state_vars, action_vars], rhs_sym, "numpy")
    return GenericODE(
        name               = f"{N}Crank",
        dim_state          = 2*N,
        dim_inputs         = N,
        dim_observation    = 2*N,
        observation_naming = [str(s) for s in state_vars],
        state_naming       = [str(s) for s in state_vars],
        inputs_naming      = [str(u_) for u_ in us],
        action_bounds      = [[-100,100]]*N,
        rhs                = rhs_sym,
        rhs_func           = lambda s,u: np.asarray(rhs_fun(s,u),dtype=np.float32).flatten(),
    )

# ─── CLI ────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--algo",    choices=["ppo","sac"], default="ppo")
p.add_argument("--timesteps",type=int,   default=1_000_000)
p.add_argument("--dt",       type=float, default=0.01,
                help="integration timestep (used for reward scaling)")
p.add_argument("--max_episode_steps", type=int, default=700)
args = p.parse_args()

# ─── 1) SAMPLE a parameter set & build the system ───────────────────────
params = gen_params(1)[0]
system = build_system(params)

# ─── 2) LAMBDIFY YOUR RUNNING-COST c(s,u) ──────────────────────────────
# substitute sampled params into symbolic cost
c_sym = c.subs(params)
# lists of symbols for s + u
state_syms  = list(sp.Matrix(state))
action_syms = list(sp.Matrix(us))
# lambdify into numpy function: cost_fun(s_vec,u_vec) → scalar
cost_fun = sp.lambdify([state_syms, action_syms], c_sym, "numpy")

# ─── 3) ENV CLASS THAT USES cost_fun AS REWARD ──────────────────────────
class CostGymEnv(CustomGymEnv):
    def __init__(self,system, max_time_steps=700):
        super().__init__(system, max_time_steps=max_time_steps)
        self.cost_fun = cost_fun
        self.dt       = args.dt

    def step(self, action):
        # copy of your integration logic
        self.current_time_step += 1
        scaled = self.state / self.state_scale
        sdot   = self.system.compute_state_dynamics(scaled, action)
        if np.isnan(sdot).any() or np.isnan(scaled).any():
            raise ValueError("NaN in state/dynamics")
        self.state = (self.state + sdot * self.dt).astype(np.float32)
        self.state = self.state.flatten() / self.state_scale

        done      = (np.linalg.norm(self.state) < 1e-4
                        or self.current_time_step>=self.max_time_steps)
        truncated = self.current_time_step>=self.max_time_steps
        info      = {}

        # negative running-cost as reward
        reward = - float(self.cost_fun(self.state, action)) * self.dt

        if self.current_time_step % 10 == 0:
            print("obs:",self.state," - reward: ", reward, " - action: ", action, " - current_timestep: ", self.current_time_step)


        return self.state, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info