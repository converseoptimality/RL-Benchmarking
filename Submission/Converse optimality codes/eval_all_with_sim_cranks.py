#!/usr/bin/env python3
# simulate_n_cranks_lite_ext.py
import os
import numpy as np
import sympy as sp

from crank_system   import N, xs, ys, us, state, V, f, G, c, params
from systems_m      import GenericODE
from cost_gym_env   import CostGymEnv
from system_utils   import plot_signals, plot_value, plot_stage_cost, animate_and_save

from stable_baselines3           import PPO, SAC, A2C
from stable_baselines3.common.vec_env   import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor   import Monitor

# ──────────── 1) BUILD & V₀ ──────────── #
params     = params[0]
V0_sym     = V.subs(params)
state_syms = list(sp.Matrix(state))
V_fun      = sp.lambdify([state_syms], V0_sym, "numpy")
v0         = float(V_fun([1.1] * (2 * N)))
print("→ nominal cost V₀ =", v0)

def build_system(ps):
    f_num, G_num = f.subs(ps), G.subs(ps)
    s_syms, u_syms = sp.Matrix(state), sp.Matrix(us)
    rhs = f_num + G_num * u_syms
    rhs_fun = sp.lambdify([s_syms, u_syms], rhs, "numpy")
    return GenericODE(
        name=f"{N}Crank",
        dim_state=2*N, dim_inputs=N, dim_observation=2*N,
        observation_naming=[str(s) for s in s_syms],
        state_naming      =[str(s) for s in s_syms],
        inputs_naming     =[str(u) for u in us],
        action_bounds     =[[-100,100]]*N,
        rhs=rhs, rhs_func=lambda s,u: np.asarray(rhs_fun(s,u),dtype=np.float32).flatten()
    )

system = build_system(params)

# ──────────── 2) LAMBDIFY c(s,u) ──────────── #
c_sym = c.subs(params)
c_fun = sp.lambdify([state_syms, list(sp.Matrix(us))], c_sym, "numpy")
def running_cost(traj, acts, dt):
    return dt * sum(float(c_fun(s,a)) for s,a in zip(traj,acts))

# ──────────── 3) SIMULATORS ──────────── #
def simulate_uncontrolled(sys, T=7.0, dt=0.001):
    n = int(T/dt)
    t = np.linspace(0, T, n, dtype=np.float32)
    traj = np.zeros((n, sys._dim_state), dtype=np.float32)
    s = np.full((sys._dim_state,), 1.1, dtype=np.float32)
    zeros = np.zeros(sys._dim_inputs, dtype=np.float32)
    for i in range(n):
        traj[i] = s
        s = s + dt * sys.compute_state_dynamics(s, zeros)
    acts = np.zeros((n, sys._dim_inputs), dtype=np.float32)
    return t, traj, acts

def _simulate_rl(sys, model_path, AlgoCls, dt, max_steps):
    mon = Monitor(CostGymEnv(sys, max_time_steps=max_steps),
                  filename=None, allow_early_resets=True)
    vec = VecNormalize(DummyVecEnv([lambda: mon]),
                       norm_obs=True, norm_reward=False)
    model = AlgoCls.load(model_path, env=vec)
    obs = vec.reset()
    traj, acts = [], []
    a=0
    for _ in range(max_steps-1):
        a+=1
        # if a == 700:
        #     break
        u, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = vec.step(u)  # SB3 VecEnv → 4-tuple
        traj.append(vec.get_original_obs()[0].copy())
        acts.append(u.copy().flatten())
        if done: break
    traj = np.vstack(traj); acts = np.vstack(acts)
    t    = np.arange(len(traj), dtype=np.float32)*dt
    return t, traj, acts

def simulate_ppo(sys, model_path, dt=0.001, max_steps=700):
    return _simulate_rl(sys, model_path, PPO, dt, max_steps)

def simulate_sac(sys, model_path, dt=0.001, max_steps=700):
    return _simulate_rl(sys, model_path, SAC, dt, max_steps)

def simulate_a2c(sys, model_path, dt=0.001, max_steps=700):
    return _simulate_rl(sys, model_path, A2C, dt, max_steps)

# optimal‐policy lambdify
gradV = sp.Matrix([V]).jacobian(sp.Matrix(state)).T.subs(params)
G_num = G.subs(params)
a_star = -sp.Rational(1,2) * G_num.T * gradV
u_opt  = sp.lambdify([sp.Matrix(state)], a_star, "numpy")

def simulate_optimal(sys, T=7.0, dt=0.001):
    n = int(T/dt)
    t = np.linspace(0, T, n, dtype=np.float32)
    traj = np.zeros((n, sys._dim_state), dtype=np.float32)
    acts = np.zeros((n, sys._dim_inputs), dtype=np.float32)
    s = np.full((sys._dim_state,), 1.1, dtype=np.float32)
    for i in range(n):
        traj[i] = s
        u = np.asarray(u_opt(s),dtype=np.float32).flatten()
        acts[i] = u
        s = s + dt * sys.compute_state_dynamics(s, u)
    return t, traj, acts

# ──────────── 4) RUN & PLOT ──────────── #
os.makedirs("plots2", exist_ok=True)
costs=[]
for label, (sim, mdl) in {
    "uncontrolled": (lambda: simulate_uncontrolled(system, T=10.0, dt=0.01), None),
    #"ppo"         : (lambda: simulate_ppo(system, "crank_models/ppo_20250508_010948_dt_0.010", dt=0.01), PPO), #trained on an old system
    "ppo"         : (lambda: simulate_ppo(system, "crank_models/ppo_20250509_012547_dt_0.010", dt=0.01), PPO),
    "sac"         : (lambda: simulate_sac(system, "crank_models/sac_20250508_020749_dt_0.010", dt=0.01), SAC),
    "a2c"         : (lambda: simulate_a2c(system, "crank_models/a2c_20250509_022254_dt_0.010", dt=0.01), A2C),
    "optimal"     : (lambda: simulate_optimal(system, T=10.0, dt=0.01), None),
}.items():
    t, x, u = sim()
    plot_signals(t, x, u, label)
    print(f"{label:11s} cost ≈", running_cost(x, u, 0.01))
    costs.append(f"{label:11s} cost ≈")
    costs.append( running_cost(x, u, 0.01))

    # also plot & save V(s) and stage‐cost
    V_vals = np.array([float(V_fun(s)) for s in x], dtype=np.float32)
    plot_value(t, V_vals, label)
    plot_stage_cost(t, x, u, lambda s,a: float(c_fun(s,a)), label)

    # and GIF
    # animate_and_save(t, x, f"{label}", label,forces=  u,skip=2)
    animate_and_save(t, x, "", label,forces=  u,skip=2)
print(costs)
print("→ done.  V₀ =", v0)