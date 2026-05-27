import matplotlib
matplotlib.use('Agg')          # <-- must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# ---------- Parse the log ----------
def parse_log(filename):
    pattern = re.compile(r"Config (\d+) (\w+) seed (\d+): normGap=([\d.]+), regret=([\d.]+)")
    data = []
    with open(filename, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                cid = int(m.group(1))
                algo = m.group(2)
                seed = int(m.group(3))
                norm = float(m.group(4))
                reg = float(m.group(5))
                data.append((cid, algo, seed, norm, reg))
    return data

data = parse_log("run_log.txt")

# ---------- Rebuild results dictionary ----------
algo_names = ["PPO", "SAC", "TD3", "DDPG", "A2C"]
# Group by (algo, cid)
from collections import defaultdict
grouped = defaultdict(lambda: defaultdict(list))
for cid, algo, seed, norm, reg in data:
    grouped[algo][cid].append((norm, reg))

final_results = {}
for algo in algo_names:
    final_results[algo] = []
    for cid in sorted(grouped[algo].keys()):
        norms = [x[0] for x in grouped[algo][cid]]
        regs  = [x[1] for x in grouped[algo][cid]]
        final_results[algo].append((cid, np.mean(norms), np.mean(regs), norms, regs))

# ---------- Parameter labels (adjust according to your actual configs) ----------
# The config IDs in the log are 0,1,2,3.  You can add a 4th when you train the extra one.
param_list = [
    {"beta": 0.0, "gamma": 0.0, "k": 0.2, "alpha": 1.0},   # cid 0
    {"beta": 1.0, "gamma": -1.0, "k": 0.1, "alpha": 1.0},  # cid 1
    {"beta": 1.0, "gamma": -1.0, "k": 1.0, "alpha": 1.0},  # cid 2
    {"beta": 0.7, "gamma": -0.7, "k": 0.2, "alpha": 1.0},  # cid 3 (your fourth config)
    {"beta": 1.4, "gamma": -1.9, "k": 0.01, "alpha": 1.0}, # cid 4
]
k_labels = [f"β={p['beta']},γ={p['gamma']},k={p['k']}" for p in param_list]

# ---------- Plotting (same as your original script) ----------
n_configs = len(k_labels)
mean_norm_gaps = np.full((len(algo_names), n_configs), np.nan)
mean_regrets   = np.full((len(algo_names), n_configs), np.nan)
for i, algo in enumerate(algo_names):
    for entry in final_results[algo]:
        cid = entry[0]
        if cid < n_configs:
            mean_norm_gaps[i, cid] = entry[1]
            mean_regrets[i, cid]   = entry[2]

# Heatmap for normalized OptGap
plt.figure(figsize=(10,5))
sns.heatmap(mean_norm_gaps, annot=True, fmt='.3f', xticklabels=k_labels,
            yticklabels=algo_names, cmap='YlOrRd', cbar_kws={'label': 'Normalized OptGap'})
plt.title('Normalized Optimality Gap')
plt.tight_layout()
plt.savefig('heatmap_normgap.png', dpi=150)
print("Saved heatmap_normgap.png")

# Heatmap for Regret
plt.figure(figsize=(10,5))
sns.heatmap(mean_regrets, annot=True, fmt='.3f', xticklabels=k_labels,
            yticklabels=algo_names, cmap='YlOrRd', cbar_kws={'label': 'Regret'})
plt.title('Regret')
plt.tight_layout()
plt.savefig('heatmap_regret.png', dpi=150)
print("Saved heatmap_regret.png")

# Bar plots (average over configs)
for metric_name, metric_idx in [('Normalized OptGap', 1), ('Regret', 2)]:
    all_per_seed = {algo: [] for algo in algo_names}
    for algo in algo_names:
        for entry in final_results[algo]:
            seed_list = entry[3] if metric_idx == 1 else entry[4]
            all_per_seed[algo].extend(seed_list)

    means = []
    cis = []
    for algo in algo_names:
        g = np.array(all_per_seed[algo])
        if len(g) == 0:
            means.append(0); cis.append(0); continue
        m = np.mean(g)
        means.append(m)
        n_bs = 1000
        bs_means = [np.mean(np.random.choice(g, size=len(g), replace=True)) for _ in range(n_bs)]
        cis.append(1.96 * np.std(bs_means))

    plt.figure(figsize=(8,5))
    plt.bar(algo_names, means, yerr=cis, capsize=5, color=['C0','C1','C2','C3','C4'])
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} (average over configs)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = f'bar_{metric_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")

    # Table
    print(f"\n{metric_name} Summary:")
    print(f"{'Algorithm':<10} {'Mean':<12} {'95% CI':<12}")
    for i, algo in enumerate(algo_names):
        g = np.array(all_per_seed[algo])
        if len(g) == 0: continue
        m = np.mean(g)
        n_bs = 1000
        bs_means = [np.mean(np.random.choice(g, size=len(g), replace=True)) for _ in range(n_bs)]
        ci_half = 1.96 * np.std(bs_means)
        print(f"{algo:<10} {m:<12.4f} ± {ci_half:.4f}")