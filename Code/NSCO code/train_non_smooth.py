import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        # Smooth baseline
        {"beta": 0.0, "gamma": 0.0, "k": 0.2, "alpha": 1.0},
        # Sharp discontinuity, hard (small k)
        {"beta": 1.0, "gamma": -1.0, "k": 0.1, "alpha": 1.0},
        # Sharp discontinuity, drift-assisted (large k)
        {"beta": 1.0, "gamma": -1.0, "k": 1.0, "alpha": 1.0},
        # Medium discontinuity
        {"beta": 0.7, "gamma": -0.7, "k": 0.2, "alpha": 1.0}
    ]

    algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG, "A2C": A2C}
    n_seeds = 5                # fewer seeds to finish faster
    total_timesteps = 30_000   # shorter training

    # Fixed evaluation initial states
    np.random.seed(0)
    eval_states = [np.random.uniform(-2, 2, 2) for _ in range(15)]

    # Results storage: dict[algo] = list of (cid, mean_norm_gap, mean_regret, list_norm_gaps, list_regrets)
    results = {algo: [] for algo in algorithms}

    for cid, params in enumerate(param_list):
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

    # ------------------------- Plotting -------------------------
    algo_names = list(algorithms.keys())
    n_configs = len(param_list)
    k_labels = [f"β={p['beta']},γ={p['gamma']},k={p['k']}" for p in param_list]

    # --- Heatmap for normalized OptGap ---
    mean_norm_gaps = np.zeros((len(algo_names), n_configs))
    for i, algo in enumerate(algo_names):
        for cid in range(n_configs):
            entry = [x for x in results[algo] if x[0] == cid][0]
            mean_norm_gaps[i, cid] = entry[1]   # mean normalized gap

    plt.figure(figsize=(10,5))
    sns.heatmap(mean_norm_gaps, annot=True, fmt='.3f', xticklabels=k_labels,
                yticklabels=algo_names, cmap='YlOrRd', cbar_kws={'label': 'Normalized OptGap'})
    plt.title('Normalized Optimality Gap')
    plt.tight_layout()
    plt.savefig('heatmap_normgap.png', dpi=150)
    plt.show()

    # --- Heatmap for Regret ---
    mean_regrets = np.zeros((len(algo_names), n_configs))
    for i, algo in enumerate(algo_names):
        for cid in range(n_configs):
            entry = [x for x in results[algo] if x[0] == cid][0]
            mean_regrets[i, cid] = entry[2]   # mean regret

    plt.figure(figsize=(10,5))
    sns.heatmap(mean_regrets, annot=True, fmt='.3f', xticklabels=k_labels,
                yticklabels=algo_names, cmap='YlOrRd', cbar_kws={'label': 'Regret'})
    plt.title('Regret')
    plt.tight_layout()
    plt.savefig('heatmap_regret.png', dpi=150)
    plt.show()

    # --- Bar plots (averaged over configs) ---
    for metric_name, metric_idx, fname in [('Normalized OptGap', 1, 'bar_normgap.png'),
                                            ('Regret', 2, 'bar_regret.png')]:
        # Let's collect per-seed values across all configs
        all_per_seed = {algo: [] for algo in algo_names}
        for algo in algo_names:
            for entry in results[algo]:
                # if metric_idx==1, seed list is entry[3]; else entry[4]
                seed_list = entry[3] if metric_idx == 1 else entry[4]
                all_per_seed[algo].extend(seed_list)

        means = []
        cis = []
        for algo in algo_names:
            g = np.array(all_per_seed[algo])
            means.append(np.mean(g))
            n_bs = 1000
            bs_means = [np.mean(np.random.choice(g, size=len(g), replace=True))
                        for _ in range(n_bs)]
            cis.append(1.96 * np.std(bs_means))

        plt.figure(figsize=(8,5))
        plt.bar(algo_names, means, yerr=cis, capsize=5, color=['C0','C1','C2','C3','C4'])
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} (average over configs)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.show()

        # Table
        print(f"\n{metric_name} Summary:")
        print(f"{'Algorithm':<10} {'Mean':<12} {'95% CI':<12}")
        for i, algo in enumerate(algo_names):
            g = np.array(all_per_seed[algo])
            m = np.mean(g)
            n_bs = 1000
            bs_means = [np.mean(np.random.choice(g, size=len(g), replace=True))
                        for _ in range(n_bs)]
            ci_half = 1.96 * np.std(bs_means)
            print(f"{algo:<10} {m:<12.4f} ± {ci_half:.4f}")