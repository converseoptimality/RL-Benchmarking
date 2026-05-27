import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C

# ------------------------- Mathematical core -------------------------
def p_vec(s, beta, gamma, alpha):
    x, y = s[0], s[1]
    if x > 0:
        return np.array([2*x + 2*beta*y, 2*beta*x + 4*y])
    elif x < 0:
        return np.array([4*x + 2*gamma*y, 2*gamma*x + 4*y])
    else:
        p_plus  = np.array([2*beta*y, 4*y])
        p_minus = np.array([2*gamma*y, 4*y])
        return alpha * p_plus + (1 - alpha) * p_minus

def V_value(s, beta, gamma):
    x, y = s[0], s[1]
    if x >= 0:
        return x**2 + 2*beta*x*y + 2*y**2
    else:
        return 2*x**2 + 2*gamma*x*y + 2*y**2

def q_cost(s, beta, gamma, k, alpha):
    p = p_vec(s, beta, gamma, alpha)
    return (0.5 + k) * np.dot(p, p)

# ------------------------- Gym environment -------------------------
class NonSmoothEnv(gym.Env):
    def __init__(self, beta=0.5, gamma=-0.3, k=0.2, alpha=1.0,
                 dt=0.05, horizon=200):
        super().__init__()
        self.beta, self.gamma, self.k, self.alpha = beta, gamma, k, alpha
        self.dt, self.horizon = dt, horizon
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float64)
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-2.0, high=2.0, size=2)
        self.step_count = 0
        return self.state.astype(np.float64), {}

    def step(self, action):
        s = self.state
        a = np.clip(action, self.action_space.low, self.action_space.high)
        p = p_vec(s, self.beta, self.gamma, self.alpha)
        s_next = s + (-self.k * p + a) * self.dt
        s_next = np.clip(s_next, -3.0, 3.0)
        r = -(q_cost(s, self.beta, self.gamma, self.k, self.alpha) + 0.5*np.dot(a, a)) * self.dt
        self.state = s_next
        self.step_count += 1
        terminated = self.step_count >= self.horizon
        truncated = False
        return s_next.astype(np.float64), r, terminated, truncated, {}

    def set_state(self, state):
        self.state = state
        self.step_count = 0

# ------------------------- Evaluation -------------------------
def evaluate_policy_on_states(model, env, init_states, eps=1e-6):
    """
    Returns mean normalized OptGap and mean regret over init_states.
    Cost is computed from the true stage cost, not from the environment reward.
    """
    norm_gaps = []
    regrets = []
    for s0 in init_states:
        env.set_state(s0)
        true_total_cost = 0.0
        s = s0
        done = False
        while not done:
            a, _ = model.predict(s, deterministic=True)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            # Compute true stage cost at current state-action
            cost = (q_cost(s, env.beta, env.gamma, env.k, env.alpha) + 0.5 * np.dot(a, a)) * env.dt
            true_total_cost += cost
            # Step the environment (ignore its reward)
            s, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
        V_opt = V_value(s0, env.beta, env.gamma)
        regret = true_total_cost - V_opt
        norm_gap = regret / (abs(V_opt) + eps)
        regrets.append(regret)
        norm_gaps.append(norm_gap)
    return np.mean(norm_gaps), np.mean(regrets)

# ------------------------- Training -------------------------
if __name__ == "__main__":
    # Parameter sets (9 configs)
    param_list = [
        # Sharp discontinuity, hardest (small k)
        {"beta": 1.4, "gamma": -1.9, "k": 0.01, "alpha": 1.0},
    ]

    algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG, "A2C": A2C}
    n_seeds = 5                # fewer seeds to finish faster
    total_timesteps = 30_000   # shorter training

    # Fixed evaluation initial states
    np.random.seed(0)
    eval_states = [np.random.uniform(-2, 2, 2) for _ in range(15)]

    # Results storage: dict[algo] = list of (cid, mean_norm_gap, mean_regret, list_norm_gaps, list_regrets)
    results = {algo: [] for algo in algorithms}

    for cid, params in enumerate(param_list, start=4):
        for algo_name, AlgoClass in algorithms.items():
            seed_norm_gaps = []
            seed_regrets = []
            for seed in range(n_seeds):
                env = NonSmoothEnv(**params)
                model = AlgoClass("MlpPolicy", env, seed=seed,
                                  policy_kwargs=dict(net_arch=[64, 64]),
                                  verbose=0)
                model.learn(total_timesteps=total_timesteps)
                norm_gap, regret = evaluate_policy_on_states(model, env, eval_states)
                seed_norm_gaps.append(norm_gap)
                seed_regrets.append(regret)
                print(f"Config {cid} {algo_name} seed {seed}: normGap={norm_gap:.4f}, regret={regret:.4f}")
            results[algo_name].append((cid, np.mean(seed_norm_gaps), np.mean(seed_regrets),
                                       seed_norm_gaps, seed_regrets))

    print("Training complete!")