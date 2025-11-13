import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from systems_m import BenchmarkSysType1
from gym_env_m import CustomGymEnv
from stable_baselines3.common.env_util import make_vec_env
from custom_mlp import CustomMLP
from system_utils import generate_g_matrix

# Configuration
dim_state = 3
dim_inputs = 3
N = 2
Per_mag = 1  # Perturbation magnitude

# Experiment configuration
seed_configs = [
    (2, 10),   # Seed 2, 10 runs
    (4, 10),   # Seed 4, 10 runs
    (5, 20),   # Seed 5, 20 runs
    (5, 40)    # Seed 5, 40 runs (additional runs)
]

optimizers = [th.optim.RAdam, th.optim.Adam]  # First 80 runs with RAdam, then 80 with Adam

# Main experiment loop
for optimizer in optimizers:
    for seed, num_runs in seed_configs:
        # Set seed for system generation
        np.random.seed(seed)
        
        for run in range(num_runs):
            # Generate system dynamics (consistent within seed group)
            g, state_vars, c, V, xi = generate_g_matrix(dim_state, dim_inputs, N, Per_mag)
            system = BenchmarkSysType1(dim_state, dim_inputs, V=V, xi=xi, g=g)

            # Create environment
            env_fn = lambda: CustomGymEnv(system=system, max_time_steps=200)
            vec_env = make_vec_env(env_fn, n_envs=1)
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

            # Configure policy with current optimizer
            policy_kwargs = {
                "features_extractor_class": CustomMLP,
                "features_extractor_kwargs": dict(features_dim=128),
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "optimizer_class": optimizer,
            }

            # Initialize and train agent
            model = PPO(
                "MlpPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./multi_opt_full_experiment/",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.1,
                ent_coef=0.01,
                max_grad_norm=0.3,
            )

            # Create unique run identifier
            optimizer_name = optimizer.__name__
            model.learn(
                total_timesteps=600000,
                tb_log_name=f"{optimizer_name}_seed{seed}_runs{num_runs}_run{run}"
            )