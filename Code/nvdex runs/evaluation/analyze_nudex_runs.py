# analyze_nudex_runs.py
# Usage:
#   python analyze_nudex_runs.py \
#     --runs_dir /home/aida-conv/conv_a_star/publ2025-code-converse/final/runs_2 \
#     --out_dir reports_nudex --episodes 20
#
# What it does:
#   - Iterates run folders (those containing meta.json, family.npz, final_model.zip/ckpt)
#   - Rebuilds the exact NUDEx env for eval (incl. VecNormalize stats if saved)
#   - Computes MC discounted COST with bootstrap J_hat = sum γ^k c_k + γ^T V*(s_T)
#   - Computes both optimality gaps:
#       * reward-form (tail-corrected):  (R̂ + γ^T V*(s_T) - V*(s_0)) / |V*(s_0)|  ≤ 0
#       * cost-form   (tail-corrected):  (Ĵ  - J*(s_0)) / |J*(s_0)|              ≥ 0
#   - Computes discounted regret under CRN:  sum γ^k (r - r*)
#   - Reads TB scalars (V_true_cost, stage_cost, residuals) if available
#   - Saves summary.csv and 4 PNGs (bars, gap, ratio, curves)
#
#   python analyze_nudex_runs.py \
#     --runs_dir /home/aida-conv/conv_a_star/publ2025-code-converse/final/runs_Nudex_1 \
#     --out_dir reports_nudex --episodes 20

import os, json, argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import load_results
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# registers "NUDEx-v0"
import nudex_env  # noqa: F401

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
ENV_ID = "NUDEx-v0"

# --- add these helpers near the top ---
def _unwrap_base_env(vec_env):
    inner = vec_env
    while hasattr(inner, "venv"):
        inner = inner.venv
    return inner.envs[0].unwrapped

def _patch_family_action_scaling(env_vec):
    base = _unwrap_base_env(env_vec)
    fam = base.fam
    if getattr(fam, "_p_patched", False):
        return  # avoid double patch

    # keep a reference to original pieces we still use
    G = fam.G
    sqrt_Sigma = fam.sqrt_Sigma
    Sigma = fam.Sigma
    rng = fam.rng

    def step_patched(s, a, p, stochastic=True):
        # corrected dynamics: s_{k+1} = f_p(s) + p * G a + w
        mean = fam.f_p(s) + (p * (G @ a))
        if stochastic and np.any(Sigma):
            z = rng.normal(size=fam.n)
            return mean + sqrt_Sigma @ z
        return mean

    fam.step = step_patched
    fam._p_patched = True


# ----------------------------- helpers -----------------------------
def nice_style():
    plt.rcParams.update({
        "figure.figsize": (8.5, 5.2),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })

def find_model_file(run_dir: Path) -> Optional[Path]:
    ckpt_best = run_dir / "ckpt" / "best_model.zip"
    if ckpt_best.exists():
        return ckpt_best
    final = run_dir / "final_model.zip"
    if final.exists():
        return final
    cands = sorted((run_dir / "ckpt").glob("model_*_steps.zip"))
    return cands[-1] if cands else None

def load_tb_scalars(tb_dir: Path) -> dict:
    tags_wanted = {
        "train/V_true_cost",
        "train/stage_cost",
        "train/bellman_cost_residual",
        "train/abs_bellman_cost_residual",
        "train/V_true_reward",
        "train/stage_reward",
        "train/bellman_reward_residual",
        "train/abs_bellman_reward_residual",
    }
    results = {}
    for ev in sorted(tb_dir.glob("events.out.tfevents.*")):
        try:
            ea = EventAccumulator(str(ev)); ea.Reload()
            avail = set(ea.Tags().get("scalars", []))
            for tag in (tags_wanted & avail):
                scal = ea.Scalars(tag)
                steps = np.array([s.step for s in scal], dtype=float)
                vals  = np.array([s.value for s in scal], dtype=float)
                results[tag] = (steps, vals)
        except Exception:
            pass
    return results

def _denorm_obs(obs, vecnorm: Optional[VecNormalize]):
    if vecnorm is None:
        return obs
    return vecnorm.unnormalize_obs(obs)

def _maybe_get_vecnorm(env) -> Optional[VecNormalize]:
    v = env
    try:
        while hasattr(v, "venv"):
            if isinstance(v, VecNormalize):
                return v
            v = v.venv
    except Exception:
        return None
    return None

def make_env_from_meta(meta: dict, seed: int, norm_stats_path: Optional[Path]):
    def _thunk():
        env = gym.make(
            ENV_ID,
            # structure
            K=meta["K"], r_v=meta["r_v"], r_w=meta["r_w"], tau=meta["tau"],
            # discount/control
            gamma=meta["gamma"], p=meta["p"],
            # spectra
            weight_pose=meta["weight_pose"], weight_v=meta["weight_v"], weight_w=meta["weight_w"],
            r_u_v=meta["r_u_v"], r_u_w=meta["r_u_w"],
            # noise
            sigma_pose=meta["sigma_pose"], sigma_v=meta["sigma_v"], sigma_w=meta["sigma_w"],
            # field
            alpha=meta["alpha"], kappa=meta["kappa"],
            # instability targeting
            rho_target=meta["rho_target"], allow_R_rescale=not meta.get("no_rescale", False),
            R_scale=meta["R_scale"], max_rescale_iter=meta["max_rescale_iter"],
            # gym/fairness
            horizon=meta["horizon"], act_limit=meta["act_limit"],
            obs_clip=meta.get("obs_clip", None),
            fixed_init=meta.get("fixed_init", True), reset_seed_per_episode=True,
            seed=seed, render_mode=None
        )
        return env
    env = DummyVecEnv([_thunk])
    if meta.get("normalize", False) and norm_stats_path and norm_stats_path.exists():
        env = VecNormalize.load(str(norm_stats_path), env)
        env.training = False
        env.norm_reward = False
    
        # >>> IMPORTANT: make the evaluation MDP match the QG derivation <<<
    _patch_family_action_scaling(env)
    return env

def vec_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out  # (obs, info)
    return out, {}  # VecEnv old API returns only obs

def vec_step(env, action):
    # Ensure batched action for VecEnv(1)
    a = action if (getattr(action, "ndim", 1) == 2) else action[None, :]
    out = env.step(a)
    # VecEnv API: (obs, rewards, dones, infos)
    if isinstance(out, tuple) and len(out) == 4:
        obs, rewards, dones, infos = out
        done = bool(dones[0]) if np.ndim(dones) else bool(dones)
        return obs, done
    # Gymnasium single-env API (fallback): (obs, r, term, trunc, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, term, trunc, info = out
        done = bool(term or trunc)
        return obs, done
    # Last resort
    return out[0], False

def j_star_cost(fam, s_raw: np.ndarray) -> float:
    return float(s_raw @ fam["P"] @ s_raw + fam["c_cost"])

def v_star_reward(fam, s_raw: np.ndarray) -> float:
    return -j_star_cost(fam, s_raw)

def optimal_action_from_fam(fam_obj, s_raw: np.ndarray) -> np.ndarray:
    if hasattr(fam_obj, "a_star_p"):
        try:
            return fam_obj.a_star_p(s_raw)
        except TypeError:
            return fam_obj.a_star_p(s_raw, fam_obj.p)
    if hasattr(fam_obj, "a_star"):
        return fam_obj.a_star(s_raw)
    raise AttributeError("Family has no oracle action method a_star_p/a_star")

def compute_metrics(model, env, fam_np: dict, gamma: float, horizon: int, episodes: int = 10) -> dict:
    vecnorm = _maybe_get_vecnorm(env)

    returns_cost_tail = []    # Ĵ with bootstrap (cost)
    jstarts = []              # J*(s0) (cost)
    optgaps_reward = []       # ≤ 0
    optgaps_cost = []         # ≥ 0
    regrets = []              # discounted regret vs oracle rewards

    fam_obj = None

    for _ in range(episodes):
        obs, info = vec_reset(env)

        if fam_obj is None:
            inner = env
            while hasattr(inner, "venv"):
                inner = inner.venv
            fam_obj = inner.envs[0].unwrapped.fam

        obs_raw = _denorm_obs(obs, vecnorm)
        s = np.array(obs_raw[0], dtype=float)

        J_star0 = j_star_cost(fam_np, s)
        V_star0_R = -J_star0

        disc = 1.0
        J_acc = 0.0
        regret = 0.0
        t = 0
        done = False

        while (not done) and (t < horizon):
            a, _ = model.predict(obs, deterministic=True)

            s_raw = np.array(_denorm_obs(obs, vecnorm)[0], dtype=float)
            a_vec = np.array(a[0] if np.ndim(a) == 2 else a, dtype=float)

            Q = fam_obj.Q; R = fam_obj.R
            c_k = float(s_raw @ Q @ s_raw + a_vec @ R @ a_vec)  # cost
            r_k = -c_k                                          # reward

            a_star = optimal_action_from_fam(fam_obj, s_raw)
            r_star = -float(s_raw @ Q @ s_raw + a_star @ R @ a_star)

            J_acc += disc * c_k
            regret += disc * (r_k - r_star)

            obs, done = vec_step(env, a)
            disc *= gamma
            t += 1

        s_T = np.array(_denorm_obs(obs, vecnorm)[0], dtype=float)
        V_tail_R = v_star_reward(fam_np, s_T)
        J_tail   = j_star_cost(fam_np, s_T)

        # reward-form gap (≤ 0): (ret + γ^T V*(s_T) - V*(s_0)) / |V*(s_0)|
        # ret_R = -J_acc   (so we can express via costs)
        ret_R = -J_acc
        denom_R = max(abs(V_star0_R), 1e-6)
        gap_R = (ret_R + disc * V_tail_R - V_star0_R) / denom_R

        # cost-form gap (≥ 0): (Ĵ - J*(s0)) / |J*(s0)|
        J_hat = J_acc + disc * J_tail
        denom_C = max(abs(J_star0), 1e-6)
        gap_C = (J_hat - J_star0) / denom_C

        returns_cost_tail.append(J_hat)
        jstarts.append(J_star0)
        optgaps_reward.append(gap_R)
        optgaps_cost.append(gap_C)
        regrets.append(regret)

    def mstd(arr):
        arr = np.asarray(arr, float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        return mean, std

    mc_cost_mean, mc_cost_std = mstd(returns_cost_tail)
    jstar_mean, jstar_std = mstd(jstarts)
    optgap_reward_mean, _ = mstd(optgaps_reward)
    optgap_cost_mean, _ = mstd(optgaps_cost)
    regret_mean, _ = mstd(regrets)

    return dict(
        mc_cost_mean=mc_cost_mean,
        mc_cost_std=mc_cost_std,
        jstar_mean=jstar_mean,
        jstar_std=jstar_std,
        optgap_reward_mean=optgap_reward_mean,  # ≤ 0
        optgap_cost_mean=optgap_cost_mean,      # ≥ 0
        regret_mean=regret_mean,
    )

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str,
                    default="/home/aida-conv/conv_a_star/publ2025-code-converse/final/runs_2")
    ap.add_argument("--out_dir", type=str, default="reports_nudex")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in sorted(r for r in runs_dir.iterdir() if r.is_dir()):
        meta_path = run / "meta.json"
        fam_path = run / "family.npz"
        model_path = find_model_file(run)
        if not (meta_path.exists() and fam_path.exists() and model_path):
            print(f"[skip] {run.name}: missing files")
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)
        fam_npz = np.load(fam_path, allow_pickle=False)
        fam = {k: fam_npz[k] for k in fam_npz.files}

        algo = meta["algo"]
        Algo = ALGOS.get(algo)
        if Algo is None:
            print(f"[skip] {run.name}: unknown algo={algo}")
            continue

        eval_seed = int(meta["seed"]) + 20000
        norm_path = run / "vecnorm.pkl"
        env = make_env_from_meta(meta, eval_seed, norm_stats_path=(norm_path if meta.get("normalize", False) else None))

        device = "cpu" if (args.force_cpu or algo in ("PPO", "A2C")) else args.device
        try:
            model = Algo.load(str(model_path), device=device)
        except TypeError:
            model = Algo.load(str(model_path))

        metrics = compute_metrics(
            model, env, fam_np=fam, gamma=meta["gamma"],
            horizon=meta["horizon"], episodes=args.episodes
        )
        env.close()

        # Optional: monitor + eval.npz
        mon_return = None
        try:
            mon_files = list(run.glob("monitor-*.monitor.csv"))
            if mon_files:
                dfm = load_results(run)
                mon_return = float(dfm["r"].rolling(20, min_periods=1).mean().iloc[-1])
        except Exception:
            pass

        eval_mean = None
        try:
            eval_npz = run / "eval" / "evaluations.npz"
            if eval_npz.exists():
                arr = np.load(eval_npz, allow_pickle=True)
                res = arr["results"]  # [n_evals, n_episodes]
                eval_mean = float(res.mean())
        except Exception:
            pass

        tb = load_tb_scalars(run / "tb")
        vtrue_ct = len(tb.get("train/V_true_cost", ([], []))[0]) if "train/V_true_cost" in tb else 0
        scost_ct = len(tb.get("train/stage_cost", ([], []))[0]) if "train/stage_cost" in tb else 0

        row = dict(
            run=str(run),
            algo=algo,
            seed=meta["seed"],
            gamma=meta["gamma"],
            horizon=meta["horizon"],
            K=meta["K"], r_v=meta["r_v"], r_w=meta["r_w"], tau=meta["tau"],
            p=meta["p"], normalize=bool(meta.get("normalize", False)),
            mc_cost_mean=metrics["mc_cost_mean"],
            mc_cost_std=metrics["mc_cost_std"],
            jstar_mean=metrics["jstar_mean"],
            jstar_std=metrics["jstar_std"],
            optgap_reward_mean=metrics["optgap_reward_mean"],
            optgap_cost_mean=metrics["optgap_cost_mean"],
            regret_mean=metrics["regret_mean"],
            eval_reward_mean=eval_mean,
            monitor_reward_smoothed=mon_return,
            tb_Vtrue_count=vtrue_ct, tb_stage_cost_count=scost_ct,
        )
        rows.append(row)
        print(f"[ok] {run.name}: MC={row['mc_cost_mean']:.3f}  J*={row['jstar_mean']:.3f}  "
              f"OptGap(R)={row['optgap_reward_mean']:.3e}  OptGap(C)={row['optgap_cost_mean']:.3e}")

    if not rows:
        print("No runs processed.")
        return

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # --------------- PLOTS ---------------
    nice_style()
    algos = sorted(df["algo"].unique())
    ps = sorted(df["p"].unique())

    # 1) Optimal vs Learned (COST)
    fig1, ax1 = plt.subplots()
    width = 0.12
    off = (np.arange(len(algos)) - (len(algos)-1)/2.0) * width
    for j, pval in enumerate(ps):
        sub = df[df["p"] == pval]
        x = np.arange(len(algos))
        means = [float(sub[sub["algo"] == A]["mc_cost_mean"].mean()) for A in algos]
        stds  = [float(sub[sub["algo"] == A]["mc_cost_std"].mean())  for A in algos]
        ax1.bar(x + off[j], means, width=width, label=f"MC (p={pval:.2f})", alpha=0.85)
        ax1.errorbar(x + off[j], means, yerr=stds, fmt='none', ecolor='k', elinewidth=1, capsize=3, alpha=0.8)
    for pval in ps:
        sub = df[df["p"] == pval]
        Jopt = float(sub["jstar_mean"].mean())
        ax1.plot([-0.5, len(algos)-0.5], [Jopt, Jopt], ls='--', lw=1.7, label=f"J*(p={pval:.2f})")
    ax1.set_xticks(np.arange(len(algos)), algos)
    ax1.set_ylabel("Discounted COST (lower is better)")
    ax1.set_title("Optimal cost J*(s0) vs learned (MC with bootstrap)")
    ax1.legend(ncol=2)
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig_opt_vs_learned.png", dpi=200)

    # 2) Difficulty: gap vs p (COST)
    fig2, ax2 = plt.subplots()
    for algo in algos:
        sub = df[df["algo"] == algo].groupby("p").agg(
            mc_mean=("mc_cost_mean", "mean"),
            jstar=("jstar_mean", "mean"),
        ).reset_index().sort_values("p")
        gap = sub["mc_mean"] - sub["jstar"]
        ax2.plot(sub["p"], gap, '-o', label=f"{algo}: gap (MC - J*)")
    ax2.set_xlabel("control authority p")
    ax2.set_ylabel("Cost gap (↓ easier)")
    ax2.set_title("Difficulty vs p (gap to optimal)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_difficulty_gap.png", dpi=200)

    # 3) Difficulty: ratio vs p (COST)
    fig3, ax3 = plt.subplots()
    for algo in algos:
        sub = df[df["algo"] == algo].groupby("p").agg(
            mc_mean=("mc_cost_mean", "mean"),
            jstar=("jstar_mean", "mean"),
        ).reset_index().sort_values("p")
        ratio = sub["mc_mean"] / sub["jstar"]
        ax3.plot(sub["p"], ratio, '-o', label=f"{algo}: ratio (MC / J*)")
    ax3.set_xlabel("control authority p")
    ax3.set_ylabel("Cost ratio (≈1 is optimal)")
    ax3.set_title("Normalized difficulty vs p")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_difficulty_ratio.png", dpi=200)

    # 4) Learning curves from TB (if present)
    fig4, ax4 = plt.subplots()
    plotted = 0
    for algo in algos:
        for pval in ps:
            cands = [Path(r) for r in df[(df["algo"] == algo) & (df["p"] == pval)]["run"]]
            if not cands:
                continue
            latest = sorted(cands)[-1]
            tb = load_tb_scalars(Path(latest) / "tb")
            if "train/V_true_cost" in tb:
                steps, vals = tb["train/V_true_cost"]
                if len(steps) > 0:
                    ax4.plot(steps, vals, alpha=0.8, label=f"{algo} p={pval:.2f}  V_true_cost")
                    plotted += 1
            if "train/stage_cost" in tb:
                steps, vals = tb["train/stage_cost"]
                if len(steps) > 0:
                    ax4.plot(steps, vals, ls='--', alpha=0.6, label=f"{algo} p={pval:.2f}  stage_cost")
                    plotted += 1
    if plotted == 0:
        ax4.text(0.5, 0.5, "No TensorBoard scalars found", ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xlabel("training steps")
    ax4.set_title("Learning curves (analytic V* cost and stage cost)")
    ax4.legend(ncol=2, fontsize=9)
    fig4.tight_layout()
    fig4.savefig(out_dir / "fig_learning_curves.png", dpi=200)

    print(f"[saved] {out_dir/'summary.csv'} and PNGs in {out_dir}")

if __name__ == "__main__":
    main()