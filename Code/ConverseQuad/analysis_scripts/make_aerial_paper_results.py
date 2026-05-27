#!/usr/bin/env python3
"""Generate paper-style evaluation tables and figures for ConverseQuad3D.

This script is intentionally self-contained around the canonical aerial
benchmark code in ``Code mode 2/Code``.  It scans trained SB3 run directories,
re-evaluates all policies under a shared CRN initial-state/noise schedule,
regenerates the previous per-rollout diagnostic plots, and writes aerial-only
summary CSVs, LaTeX tables, and paper-style figures.

The reported table metrics use the reward convention from the manuscript:

    OptGap(pi; s0) = (V_r^pi(s0) - V_r^*(s0)) / (|V_r^*(s0)| + eps)
    Regret_T       = J_hat^pi(s0) - J*(s0)
                   = sum_k gamma^k c_k + gamma^T J*(s_T) - J*(s0)

Internally, the environment exposes costs c=-r and the analytic optimal cost
value J*(s)=s^T P s + b.  We therefore compute

    V_r^*(s0) = -J*(s0)

and estimate V_r^pi by a finite policy rollout with an oracle/value tail
bootstrap, i.e. ``J_hat = sum gamma^k c_k + gamma^T J*(s_T)``.  The finite
horizon, non-tail OptGap is also stored in the trial-level CSV for auditability.
The old on-policy stage-action oracle gap is retained as
``discounted_regret_stage_action`` for diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Keep TensorBoard/TensorFlow backends quiet when SB3 is imported for inference.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _find_bundle_root(start: Path) -> Path:
    """Return the copied ConverseQuad bundle root from any nested source copy."""
    for parent in (start.parent, *start.parents):
        if (parent / "Code mode 2" / "Code").exists() and (parent / "aerial_results").exists():
            return parent
    return start.resolve().parents[1]

ROOT = _find_bundle_root(Path(__file__).resolve())
CODE_DIR = ROOT / "Code mode 2" / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluate_aerial as aerial_eval  # noqa: E402
import plot_aerial_results as aerial_plots  # noqa: E402


ALGO_ORDER = ("PPO", "A2C", "SAC", "TD3", "DDPG")
STATE_NAMES = aerial_eval.STATE_NAMES
EPS = 1.0e-12
REGRET_MIN = 1.0e-12
SYMLIN = 1.0e-3

ENV_DEFAULTS: Dict[str, Any] = {
    "gamma": 0.99,
    "tau": 0.005,
    "m": 1.0,
    "Jx": 0.03,
    "Jy": 0.03,
    "Jz": 0.03,
    "ell": 0.2,
    "p": 0.7,
    "eta": 0.005,
    "sigma_noise": 0.01,
    "noise_type": "gust",
    "eps_gust": 0.1,
    "gust_scale": 10.0,
    "horizon": 512,
    "act_limit": 1.0,
    "obs_clip": 1.0e6,
    "state_limit": 1.0e6,
    "cost_limit": 1.0e12,
    "terminate_on_unhealthy": True,
    "unhealthy_penalty": 1.0e3,
    "info_clip": 1.0e30,
}

TRAIN_META_KEYS = (
    "total_steps",
    "fixed_init",
    "normalize",
    "num_envs",
    "policy_net",
    "batch_size",
    "buffer_size",
    "learning_starts",
    "train_freq",
    "gradient_steps",
    "action_noise_sigma",
)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_name: str
    algo: str
    p: float
    seed: int
    meta: Mapping[str, Any]
    model_path: Path
    vecnorm_path: Optional[Path]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def sci(x: Any) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return str(x)


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{sci(lo)}, {sci(hi)}]"


def algo_rank(algo: str) -> int:
    try:
        return ALGO_ORDER.index(str(algo))
    except ValueError:
        return len(ALGO_ORDER)


def parse_float_list(text: Optional[str]) -> Optional[Tuple[float, ...]]:
    if text is None or str(text).strip() == "":
        return None
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def parse_str_list(text: Optional[str]) -> Optional[Tuple[str, ...]]:
    if text is None or str(text).strip() == "":
        return None
    return tuple(x.strip().upper() for x in str(text).split(",") if x.strip())


def discover_runs(runs_dir: Path, algos: Optional[Sequence[str]] = None, p_values: Optional[Sequence[float]] = None) -> List[RunInfo]:
    algos_set = None if algos is None else {a.upper() for a in algos}
    p_set = None if p_values is None else {round(float(p), 12) for p in p_values}
    out: List[RunInfo] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        model_path = run_dir / "final_model.zip"
        family_path = run_dir / "family.npz"
        if not (meta_path.exists() and model_path.exists() and family_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] could not read {meta_path}: {exc}")
            continue
        if meta.get("sb3_available") is False:
            print(f"[skip] SB3 fallback artifact: {run_dir.name}")
            continue
        algo = str(meta.get("algo", run_dir.name.split("_")[0])).upper()
        p = float(meta.get("p", np.nan))
        seed = int(meta.get("seed", 0))
        if algos_set is not None and algo not in algos_set:
            continue
        if p_set is not None and round(p, 12) not in p_set:
            continue
        vecnorm_path = run_dir / "vecnorm.pkl"
        out.append(
            RunInfo(
                run_dir=run_dir,
                run_name=run_dir.name,
                algo=algo,
                p=p,
                seed=seed,
                meta=meta,
                model_path=model_path,
                vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
            )
        )
    out.sort(key=lambda r: (r.p, algo_rank(r.algo), r.seed, r.run_name))
    return out


def namespace_from_run(run: RunInfo, *, episodes: int, eval_seed: int, out_dir: Path, device: str) -> argparse.Namespace:
    vals = dict(ENV_DEFAULTS)
    env_params = run.meta.get("env_params") if isinstance(run.meta.get("env_params"), Mapping) else {}
    vals.update({k: env_params[k] for k in ENV_DEFAULTS if k in env_params})
    vals.update({k: run.meta[k] for k in ENV_DEFAULTS if k in run.meta})
    ns = argparse.Namespace(
        model_path=str(run.model_path),
        vecnorm_path=str(run.vecnorm_path) if run.vecnorm_path is not None else None,
        algo=run.algo,
        episodes=int(episodes),
        seed=int(eval_seed),
        out_dir=str(out_dir),
        device=device,
        **vals,
    )
    return ns


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    ensure_dirs(path.parent)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def generate_common_s0(eval_args: argparse.Namespace) -> List[np.ndarray]:
    """Generate the CRN initial-state schedule once and reuse it for all runs."""

    s0_list: List[np.ndarray] = []
    for ep in range(int(eval_args.episodes)):
        env = aerial_eval.make_env(eval_args, int(eval_args.seed) + ep)
        obs, _ = env.reset(seed=int(eval_args.seed) + ep)
        s0_list.append(np.asarray(obs, dtype=float).reshape(12))
        env.close()
    return s0_list


def rollout_policy_enhanced(
    args: argparse.Namespace,
    label: str,
    predictor: Callable[[np.ndarray, Any], np.ndarray],
    s0_list: Sequence[np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Roll out one label and return canonical arrays plus tail-bootstrap summary."""

    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    states_all: List[np.ndarray] = []
    actions_all: List[np.ndarray] = []
    virtual_all: List[np.ndarray] = []
    costs_all: List[float] = []
    rewards_all: List[float] = []
    values_all: List[float] = []
    regrets_all: List[float] = []
    residuals_all: List[float] = []
    energy_all: List[float] = []
    ep_idx_all: List[int] = []
    t_all: List[int] = []

    for ep, s0 in enumerate(s0_list):
        env = aerial_eval.make_env(args, int(args.seed) + ep)
        obs, _ = env.reset(seed=int(args.seed) + ep, options={"s0": np.asarray(s0, dtype=float)})
        fam = env.unwrapped.fam
        V0 = float(fam.value(np.asarray(obs, dtype=float)))
        disc = 1.0
        disc_cost = 0.0
        disc_oracle_cost_on_policy_states = 0.0
        disc_regret = 0.0
        ep_energy: List[float] = []
        ep_bellman: List[float] = []
        final_state_norm = float(np.linalg.norm(obs))
        terminated = False
        truncated = False
        termination_reason = ""
        steps = 0

        for t in range(int(args.horizon)):
            s = np.asarray(obs, dtype=float).reshape(12)
            a = np.asarray(predictor(obs, env), dtype=float).reshape(4)
            u = np.asarray(fam.phi(a), dtype=float)
            oracle_a = fam.nu_star(s)
            oracle_cost = float(fam.stage_cost(s, oracle_a))
            obs_next, reward, terminated, truncated, info = env.step(a.astype(np.float32))
            cost = float(info.get("cost", -float(reward)))
            regret_step = float(cost - oracle_cost)
            V_true = float(fam.value(s))
            row: Dict[str, Any] = {
                "label": label,
                "episode": ep,
                "t": t,
                "cost": cost,
                "reward": float(reward),
                "V_true": V_true,
                "oracle_cost": oracle_cost,
                "regret_step": regret_step,
                "state_norm": float(np.linalg.norm(s)),
                "position_error_norm": float(np.linalg.norm(s[0:3])),
                "attitude_error_norm": float(np.linalg.norm(s[3:6])),
                "action_norm_raw": float(np.linalg.norm(a)),
                "action_norm_virtual": float(np.linalg.norm(u)),
                "energy_residual": float(info.get("energy_residual", fam.energy_residual(s))),
                "oracle_bellman_residual": float(info.get("oracle_bellman_residual", fam.oracle_bellman_residual(s))),
                "unhealthy_transition": bool(info.get("unhealthy_transition", False)),
                "terminated_by_safety": bool(info.get("terminated_by_safety", False)),
                "was_action_clipped": bool(info.get("was_action_clipped", False)),
            }
            for i, name in enumerate(STATE_NAMES):
                row[name] = float(s[i])
            for i in range(4):
                row[f"nu_{i}"] = float(a[i])
                row[f"u_{i}"] = float(u[i])
            rows.append(row)

            states_all.append(s.copy())
            actions_all.append(a.copy())
            virtual_all.append(u.copy())
            costs_all.append(cost)
            rewards_all.append(float(reward))
            values_all.append(V_true)
            regrets_all.append(regret_step)
            residuals_all.append(float(row["oracle_bellman_residual"]))
            energy_all.append(float(row["energy_residual"]))
            ep_idx_all.append(ep)
            t_all.append(t)

            disc_cost += disc * cost
            disc_oracle_cost_on_policy_states += disc * oracle_cost
            disc_regret += disc * regret_step
            disc *= float(args.gamma)
            ep_energy.append(abs(float(row["energy_residual"])))
            ep_bellman.append(abs(float(row["oracle_bellman_residual"])))
            obs = obs_next
            final_state_norm = float(np.linalg.norm(obs))
            steps = t + 1
            if terminated or truncated:
                termination_reason = str(info.get("unhealthy_reason", "")) if terminated else "horizon"
                break

        sT = np.asarray(obs, dtype=float).reshape(12)
        terminal_V_star = float(fam.value(sT))
        tail_bootstrap_cost = float(disc_cost + disc * terminal_V_star)
        finite_reward_return = -float(disc_cost)
        tail_reward_return = -float(tail_bootstrap_cost)
        reward_star_s0 = -float(V0)
        optgap_reward_finite = float((finite_reward_return - reward_star_s0) / (abs(reward_star_s0) + EPS))
        optgap_reward_tail = float((tail_reward_return - reward_star_s0) / (abs(reward_star_s0) + EPS))
        cost_optgap_tail = float((tail_bootstrap_cost - V0) / (abs(V0) + EPS))
        cost_regret_tail = float(tail_bootstrap_cost - V0)

        env.close()
        summary.append(
            {
                "label": label,
                "episode": ep,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": termination_reason,
                "discounted_cost_policy": float(disc_cost),
                "discounted_cost_policy_finite": float(disc_cost),
                "discounted_cost_policy_tail_oracle": tail_bootstrap_cost,
                "discounted_cost_oracle": float(disc_oracle_cost_on_policy_states),
                "discounted_regret": cost_regret_tail,
                "discounted_regret_stage_action": float(disc_regret),
                "finite_horizon_opt_gap": float((disc_cost - V0) / (abs(V0) + EPS)),
                "optgap_reward_finite": optgap_reward_finite,
                "optgap_reward_tail": optgap_reward_tail,
                "optgap_cost_tail": cost_optgap_tail,
                "reward_return_policy_finite": finite_reward_return,
                "reward_return_policy_tail_oracle": tail_reward_return,
                "reward_star_s0": reward_star_s0,
                "final_state_norm": final_state_norm,
                "terminal_discount": float(disc),
                "terminal_V_star": terminal_V_star,
                "mean_energy_residual": float(np.mean(ep_energy)) if ep_energy else np.nan,
                "max_oracle_bellman_residual": float(np.max(ep_bellman)) if ep_bellman else np.nan,
                "V_star_s0": float(V0),
            }
        )

    arrays = {
        "episodes": np.asarray(ep_idx_all, dtype=int),
        "t": np.asarray(t_all, dtype=int),
        "states": np.asarray(states_all, dtype=float),
        "actions_raw": np.asarray(actions_all, dtype=float),
        "actions_virtual": np.asarray(virtual_all, dtype=float),
        "costs": np.asarray(costs_all, dtype=float),
        "rewards": np.asarray(rewards_all, dtype=float),
        "values": np.asarray(values_all, dtype=float),
        "regrets": np.asarray(regrets_all, dtype=float),
        "energy_residuals": np.asarray(energy_all, dtype=float),
        "oracle_bellman_residuals": np.asarray(residuals_all, dtype=float),
    }
    return rows, summary, arrays


def run_metadata_columns(run: RunInfo, eval_seed: int, episodes: int) -> Dict[str, Any]:
    meta = run.meta
    env_params = meta.get("env_params") if isinstance(meta.get("env_params"), Mapping) else {}
    cols: Dict[str, Any] = {
        "run": run.run_name,
        "run_dir": str(run.run_dir),
        "algo": run.algo,
        "p": run.p,
        "train_seed": run.seed,
        "eval_seed": int(eval_seed),
        "eval_episodes": int(episodes),
    }
    for key in ENV_DEFAULTS:
        cols[key] = meta.get(key, env_params.get(key, ENV_DEFAULTS[key]))
    for key in TRAIN_META_KEYS:
        if key in meta:
            cols[key] = meta.get(key)
    return cols


def add_metadata(rows: Sequence[Dict[str, Any]], metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        merged = dict(metadata)
        merged.update(row)
        # CRN unit used for paired/bootstrap resampling.
        if "episode" in merged:
            merged["crn_episode"] = int(merged["episode"])
            merged["crn_unit"] = f"p{float(merged['p']):.12g}_seed{int(merged['train_seed'])}_ep{int(merged['episode'])}"
            merged["crn_unit_within_p"] = f"seed{int(merged['train_seed'])}_ep{int(merged['episode'])}"
        out.append(merged)
    return out


def evaluate_one_run(
    run: RunInfo,
    eval_args: argparse.Namespace,
    s0_list: Sequence[np.ndarray],
    eval_dir: Path,
    *,
    make_plots: bool,
    reuse_existing: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_path = eval_dir / "summary_aerial.csv"
    rollouts_path = eval_dir / "rollouts_aerial.csv"
    if reuse_existing and summary_path.exists() and rollouts_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict("records")
        rollout_rows = pd.read_csv(rollouts_path).to_dict("records")
        if make_plots:
            make_previous_plots(eval_dir)
        return summary_rows, rollout_rows

    ensure_dirs(eval_dir)
    predictor_policy = aerial_eval.make_model_predictor(eval_args) if eval_args.model_path else aerial_eval.zero_predict
    predictors: Dict[str, Callable[[np.ndarray, Any], np.ndarray]] = {
        "policy": predictor_policy,
        "oracle": aerial_eval.oracle_predict,
        "zero": aerial_eval.zero_predict,
    }
    all_summary: List[Dict[str, Any]] = []
    all_rollouts: List[Dict[str, Any]] = []
    for label, predictor in predictors.items():
        rows, summary, arrays = rollout_policy_enhanced(eval_args, label, predictor, s0_list)
        all_rollouts.extend(rows)
        all_summary.extend(summary)
        np.savez(eval_dir / f"rollouts_{label}.npz", **arrays)

    write_csv(summary_path, all_summary)
    write_csv(rollouts_path, all_rollouts)
    if make_plots:
        make_previous_plots(eval_dir)
    return all_summary, all_rollouts


def make_previous_plots(eval_dir: Path) -> None:
    figs = eval_dir / "figs"
    ensure_dirs(figs)
    data = aerial_plots.load_npz(str(eval_dir))
    if not data:
        print(f"[warn] no rollout npz files found for plotting in {eval_dir}")
        return
    aerial_plots.plot_norms(data, str(figs))
    aerial_plots.plot_actions(data, str(figs))
    aerial_plots.plot_trajectory(data, str(figs))
    aerial_plots.plot_cost_bar(str(eval_dir), str(figs))
    aerial_plots.plot_regret_residuals(data, str(figs))


def bootstrap_mean_ci(
    values: Sequence[float],
    units: Optional[Sequence[Any]] = None,
    *,
    iters: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260516,
) -> Tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    if units is None:
        unit_arr = np.arange(x.size).astype(object)
    else:
        unit_arr = np.asarray(units, dtype=object)
    x = x[mask]
    unit_arr = unit_arr[mask]
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    per_unit = pd.DataFrame({"unit": unit_arr, "value": x}).groupby("unit", as_index=False)["value"].mean()
    vals = per_unit["value"].to_numpy(dtype=float)
    mean = float(np.mean(vals))
    if vals.size == 1 or iters <= 0:
        return mean, mean, mean, int(vals.size)
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(int(iters), vals.size), replace=True).mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return mean, lo, hi, int(vals.size)


def build_policy_trials(summary_df: pd.DataFrame) -> pd.DataFrame:
    policy = summary_df[summary_df["label"].astype(str) == "policy"].copy()
    for col in ["discounted_cost_policy_tail_oracle", "V_star_s0", "discounted_regret", "optgap_reward_tail"]:
        policy[col] = pd.to_numeric(policy[col], errors="coerce")
    # Main paper metrics.
    policy["OptGap"] = policy["optgap_reward_tail"]
    policy["Regret"] = policy["discounted_regret"]
    policy["experiment"] = policy["p"].astype(float).map(lambda p: f"aerial_p{p:.1f}")
    policy["Experiment"] = policy["experiment"]
    policy["Algorithm"] = policy["algo"]
    return policy


def grouped_stats(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric: str,
    unit_col: str,
    *,
    iters: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, (key, g) in enumerate(df.groupby(list(group_cols), dropna=False)):
        if not isinstance(key, tuple):
            key = (key,)
        mean, lo, hi, n = bootstrap_mean_ci(g[metric].to_numpy(float), g[unit_col].astype(str).tolist(), iters=iters, seed=seed + idx)
        rows.append({**dict(zip(group_cols, key)), "mean": mean, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows)


def make_table(policy_df: pd.DataFrame, *, iters: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    experiment_order = [f"aerial_p{p:.1f}" for p in sorted(policy_df["p"].astype(float).unique())] + ["aerial_overall"]

    for experiment in experiment_order:
        if experiment == "aerial_overall":
            sub_exp = policy_df.copy()
            unit_col = "crn_unit"
        else:
            sub_exp = policy_df[policy_df["experiment"] == experiment].copy()
            unit_col = "crn_unit_within_p"
        for algo in ALGO_ORDER:
            g = sub_exp[sub_exp["algo"] == algo]
            if g.empty:
                continue
            og_mean, og_lo, og_hi, n_og = bootstrap_mean_ci(g["OptGap"].to_numpy(float), g[unit_col].astype(str).tolist(), iters=iters, seed=11_000 + algo_rank(algo) + 101 * len(rows))
            rg_mean, rg_lo, rg_hi, n_rg = bootstrap_mean_ci(g["Regret"].to_numpy(float), g[unit_col].astype(str).tolist(), iters=iters, seed=22_000 + algo_rank(algo) + 101 * len(rows))
            rows.append(
                {
                    "Experiment": experiment,
                    "Algorithm": algo,
                    "OptGap (mean)": og_mean,
                    "OptGap 95% CI": fmt_ci(og_lo, og_hi),
                    "OptGap CI low": og_lo,
                    "OptGap CI high": og_hi,
                    "Regret (mean)": rg_mean,
                    "Regret 95% CI": fmt_ci(rg_lo, rg_hi),
                    "Regret CI low": rg_lo,
                    "Regret CI high": rg_hi,
                    "N": min(n_og, n_rg),
                }
            )
    table = pd.DataFrame(rows)
    table["__exp_order"] = table["Experiment"].map({e: i for i, e in enumerate(experiment_order)})
    table["__algo_order"] = table["Algorithm"].map({a: i for i, a in enumerate(ALGO_ORDER)})
    table = table.sort_values(["__exp_order", "__algo_order"]).drop(columns=["__exp_order", "__algo_order"])
    return table


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    display_cols = ["Experiment", "Algorithm", "OptGap (mean)", "OptGap 95% CI", "Regret (mean)", "Regret 95% CI"]
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\caption{Aerial ConverseQuad3D performance per algorithm and fixture. Metrics are means and 95\% confidence intervals via paired bootstrap over CRN trials.}")
    lines.append(r"\label{tab:aerial_overall_performance}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & Algorithm & OptGap (mean) & OptGap 95\% CI & Regret (mean) & Regret 95\% CI \\")
    lines.append(r"\midrule")
    prev_exp: Optional[str] = None
    for _, row in table.iterrows():
        exp = str(row["Experiment"])
        if prev_exp is not None and exp != prev_exp:
            lines.append(r"\midrule")
        cells = [
            exp.replace("_", r"\_"),
            str(row["Algorithm"]),
            sci(row["OptGap (mean)"]),
            str(row["OptGap 95% CI"]),
            sci(row["Regret (mean)"]),
            str(row["Regret 95% CI"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
        prev_exp = exp
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    display_cols = ["Experiment", "Algorithm", "OptGap (mean)", "OptGap 95% CI", "Regret (mean)", "Regret 95% CI", "N"]
    rows = []
    for _, r in table.iterrows():
        rows.append(
            {
                "Experiment": r["Experiment"],
                "Algorithm": r["Algorithm"],
                "OptGap (mean)": sci(r["OptGap (mean)"]),
                "OptGap 95% CI": r["OptGap 95% CI"],
                "Regret (mean)": sci(r["Regret (mean)"]),
                "Regret 95% CI": r["Regret 95% CI"],
                "N": int(r["N"]),
            }
        )
    # Avoid requiring pandas' optional ``tabulate`` dependency in the repo venv.
    lines = ["| " + " | ".join(display_cols) + " |", "| " + " | ".join(["---"] * len(display_cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in display_cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def savefig(fig: plt.Figure, base: Path, formats: Sequence[str]) -> List[str]:
    out_paths: List[str] = []
    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        fig.savefig(p, bbox_inches="tight", dpi=300 if fmt.lower() == "png" else None)
        out_paths.append(str(p))
    plt.close(fig)
    return out_paths


def sort_overall_for_metric(overall: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Order overall bars by the plotted metric, using labels only as ties."""
    if metric == "optgap":
        sort_col = "optgap_mean"
        ascending = False  # reward-signed: larger / closer to zero is better
    elif metric == "regret":
        sort_col = "regret_mean"
        ascending = True  # cost-convention regret: smaller is better
    else:
        raise ValueError(metric)

    out = overall.copy()
    out["__rank_value"] = pd.to_numeric(out[sort_col], errors="coerce")
    out["__algo_label"] = out["algo"].astype(str)
    return out.sort_values(
        ["__rank_value", "__algo_label"],
        ascending=[ascending, True],
        na_position="last",
    ).drop(columns=["__rank_value", "__algo_label"])


def plot_overall_bars(overall: pd.DataFrame, figures_dir: Path, formats: Sequence[str]) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []

    # Signed reward optimality gap.
    opt_overall = sort_overall_for_metric(overall, "optgap")
    x = np.arange(len(opt_overall))
    labels = opt_overall["algo"].astype(str).tolist()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(10, len(opt_overall))))[: len(opt_overall)]
    means = opt_overall["optgap_mean"].to_numpy(float)
    los = opt_overall["optgap_lo"].to_numpy(float)
    his = opt_overall["optgap_hi"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=True)
    ax.bar(x, means, color=colors, width=0.68, edgecolor="black", linewidth=0.35)
    yerr = np.vstack([np.maximum(means - los, 0.0), np.maximum(his - means, 0.0)])
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Signed reward OptGap $(V_r^\pi - V_r^*)/(|V_r^*|+\varepsilon)$")
    ax.set_title("ConverseQuad3D aerial overall optimality gap")
    ax.set_yscale("symlog", linthresh=SYMLIN)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    paths = savefig(fig, figures_dir / "fig_bar_optgap_overall", formats)
    # Aerial-explicit aliases requested in the plan.
    for src in paths:
        src_p = Path(src)
        shutil.copyfile(src_p, figures_dir / f"fig_bar_optgap_overall_aerial{src_p.suffix}")
    manifest.append({"figure": "fig_bar_optgap_overall", "paths": paths, "metric": "OptGap", "scope": "aerial_overall"})

    # Discounted regret.
    reg_overall = sort_overall_for_metric(overall, "regret")
    x = np.arange(len(reg_overall))
    labels = reg_overall["algo"].astype(str).tolist()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(10, len(reg_overall))))[: len(reg_overall)]
    means = np.maximum(reg_overall["regret_mean"].to_numpy(float), REGRET_MIN)
    los = np.maximum(reg_overall["regret_lo"].to_numpy(float), REGRET_MIN)
    his = np.maximum(reg_overall["regret_hi"].to_numpy(float), REGRET_MIN)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=True)
    ax.bar(x, means, color=colors, width=0.68, edgecolor="black", linewidth=0.35)
    lower = np.maximum(means - los, REGRET_MIN * 0.5)
    upper = np.maximum(his - means, REGRET_MIN * 0.5)
    ax.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", elinewidth=1.0, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Tail-bootstrapped cost regret $\hat J^\pi - J^*$")
    ax.set_title("ConverseQuad3D aerial overall regret")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    paths = savefig(fig, figures_dir / "fig_bar_regret_overall", formats)
    for src in paths:
        src_p = Path(src)
        shutil.copyfile(src_p, figures_dir / f"fig_bar_regret_overall_aerial{src_p.suffix}")
    manifest.append({"figure": "fig_bar_regret_overall", "paths": paths, "metric": "Regret", "scope": "aerial_overall"})
    return manifest


def annotate_heatmap(ax: plt.Axes, mat: np.ndarray, *, log_values: bool = False) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                text = "—"
            elif log_values:
                text = f"{val:.2f}"
            else:
                text = sci(val)
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")


def plot_heatmaps(by_p_algo: pd.DataFrame, figures_dir: Path, formats: Sequence[str]) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    p_values = sorted(by_p_algo["p"].astype(float).unique())
    algos = [a for a in ALGO_ORDER if a in set(by_p_algo["algo"].astype(str))]

    opt_mat = np.full((len(algos), len(p_values)), np.nan, dtype=float)
    reg_mat = np.full_like(opt_mat, np.nan)
    for i, algo in enumerate(algos):
        for j, p in enumerate(p_values):
            g = by_p_algo[(by_p_algo["algo"] == algo) & (np.isclose(by_p_algo["p"].astype(float), p))]
            if not g.empty:
                opt_mat[i, j] = float(g.iloc[0]["optgap_mean"])
                reg_mat[i, j] = float(g.iloc[0]["regret_mean"])

    # Signed OptGap heatmap.
    finite = opt_mat[np.isfinite(opt_mat)]
    if finite.size:
        vmin = float(min(np.min(finite), -SYMLIN))
        vmax = float(max(np.max(finite), 0.0))
    else:
        vmin, vmax = -1.0, 0.0
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    else:
        norm = None
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    im = ax.imshow(opt_mat, aspect="auto", cmap="coolwarm", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(np.arange(len(p_values)))
    ax.set_xticklabels([f"{p:.1f}" for p in p_values])
    ax.set_yticks(np.arange(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_xlabel("Control authority p")
    ax.set_ylabel("Algorithm")
    ax.set_title("Aerial signed reward OptGap by algorithm and p")
    annotate_heatmap(ax, opt_mat)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Mean OptGap")
    paths = savefig(fig, figures_dir / "fig_optgap_heatmap_aerial", formats)
    # Also provide the shorter requested-name variant.
    for src in paths:
        src_p = Path(src)
        shutil.copyfile(src_p, figures_dir / f"fig_optgap_heatmap{src_p.suffix}")
    manifest.append({"figure": "fig_optgap_heatmap_aerial", "paths": paths, "metric": "OptGap", "scope": "algorithm_by_p"})

    # Tail-bootstrapped cost regret heatmap on log10 scale.
    log_reg = np.log10(np.maximum(reg_mat, REGRET_MIN))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    im = ax.imshow(log_reg, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(p_values)))
    ax.set_xticklabels([f"{p:.1f}" for p in p_values])
    ax.set_yticks(np.arange(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_xlabel("Control authority p")
    ax.set_ylabel("Algorithm")
    ax.set_title(r"Aerial $\log_{10}$ tail cost regret by algorithm and p")
    annotate_heatmap(ax, log_reg, log_values=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(r"$\log_{10}$(mean tail cost regret)")
    paths = savefig(fig, figures_dir / "fig_regret_heatmap_aerial", formats)
    for src in paths:
        src_p = Path(src)
        shutil.copyfile(src_p, figures_dir / f"fig_regret_heatmap{src_p.suffix}")
    manifest.append({"figure": "fig_regret_heatmap_aerial", "paths": paths, "metric": "Regret", "scope": "algorithm_by_p"})
    return manifest


def plot_vs_p_lines(by_p_algo: pd.DataFrame, figures_dir: Path, formats: Sequence[str]) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    colors = {algo: plt.get_cmap("tab10")(i) for i, algo in enumerate(ALGO_ORDER)}

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    for algo in ALGO_ORDER:
        g = by_p_algo[by_p_algo["algo"] == algo].sort_values("p")
        if g.empty:
            continue
        ax.errorbar(
            g["p"].astype(float),
            g["optgap_mean"].astype(float),
            yerr=np.vstack([
                np.maximum(g["optgap_mean"].astype(float) - g["optgap_lo"].astype(float), 0.0),
                np.maximum(g["optgap_hi"].astype(float) - g["optgap_mean"].astype(float), 0.0),
            ]),
            marker="o",
            capsize=3,
            label=algo,
            color=colors.get(algo),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=SYMLIN)
    ax.set_xlabel("Control authority p")
    ax.set_ylabel("Signed reward OptGap")
    ax.set_title("Aerial OptGap versus control authority")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(frameon=False, ncol=3)
    paths = savefig(fig, figures_dir / "fig_optgap_vs_p_aerial", formats)
    manifest.append({"figure": "fig_optgap_vs_p_aerial", "paths": paths, "metric": "OptGap", "scope": "p_sweep"})

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    for algo in ALGO_ORDER:
        g = by_p_algo[by_p_algo["algo"] == algo].sort_values("p")
        if g.empty:
            continue
        means = np.maximum(g["regret_mean"].astype(float).to_numpy(), REGRET_MIN)
        los = np.maximum(g["regret_lo"].astype(float).to_numpy(), REGRET_MIN)
        his = np.maximum(g["regret_hi"].astype(float).to_numpy(), REGRET_MIN)
        ax.errorbar(
            g["p"].astype(float),
            means,
            yerr=np.vstack([np.maximum(means - los, REGRET_MIN * 0.5), np.maximum(his - means, REGRET_MIN * 0.5)]),
            marker="o",
            capsize=3,
            label=algo,
            color=colors.get(algo),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Control authority p")
    ax.set_ylabel("Tail-bootstrapped cost regret")
    ax.set_title("Aerial tail cost regret versus control authority")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(frameon=False, ncol=3)
    paths = savefig(fig, figures_dir / "fig_regret_vs_p_aerial", formats)
    manifest.append({"figure": "fig_regret_vs_p_aerial", "paths": paths, "metric": "Regret", "scope": "p_sweep"})
    return manifest


def make_stats(policy_df: pd.DataFrame, *, iters: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    opt_by_p = grouped_stats(policy_df, ["p", "algo"], "OptGap", "crn_unit_within_p", iters=iters, seed=31_000)
    reg_by_p = grouped_stats(policy_df, ["p", "algo"], "Regret", "crn_unit_within_p", iters=iters, seed=41_000)
    by_p_algo = opt_by_p.rename(columns={"mean": "optgap_mean", "lo": "optgap_lo", "hi": "optgap_hi", "n": "n_optgap"}).merge(
        reg_by_p.rename(columns={"mean": "regret_mean", "lo": "regret_lo", "hi": "regret_hi", "n": "n_regret"}),
        on=["p", "algo"],
        how="outer",
    )
    by_p_algo["algo_order"] = by_p_algo["algo"].map({a: i for i, a in enumerate(ALGO_ORDER)})
    by_p_algo = by_p_algo.sort_values(["p", "algo_order"]).drop(columns="algo_order")

    opt_overall = grouped_stats(policy_df, ["algo"], "OptGap", "crn_unit", iters=iters, seed=51_000)
    reg_overall = grouped_stats(policy_df, ["algo"], "Regret", "crn_unit", iters=iters, seed=61_000)
    overall = opt_overall.rename(columns={"mean": "optgap_mean", "lo": "optgap_lo", "hi": "optgap_hi", "n": "n_optgap"}).merge(
        reg_overall.rename(columns={"mean": "regret_mean", "lo": "regret_lo", "hi": "regret_hi", "n": "n_regret"}),
        on="algo",
        how="outer",
    )
    overall["algo_order"] = overall["algo"].map({a: i for i, a in enumerate(ALGO_ORDER)})
    overall = overall.sort_values("algo_order").drop(columns="algo_order")
    return by_p_algo, overall


def write_readme(
    out_dir: Path,
    runs: Sequence[RunInfo],
    *,
    episodes: int,
    eval_seed: int,
    bootstrap_iters: int,
    formats: Sequence[str],
) -> None:
    algos = ", ".join(a for a in ALGO_ORDER if any(r.algo == a for r in runs))
    p_values = ", ".join(f"{p:.1f}" for p in sorted({r.p for r in runs}))
    txt = f"""# New results arial

Generated paper-style results for the `ConverseQuad3D-v0` aerial converse benchmark.

## Evaluation scope

- Runs directory: `aerial_results/runs`
- Number of evaluated runs: **{len(runs)}**
- Algorithms: **{algos}**
- Control-authority fixtures: **p = {p_values}**
- Training seeds found: **{', '.join(map(str, sorted({r.seed for r in runs})))}**
- Evaluation episodes per run: **{episodes}**
- Shared CRN evaluation seed: **{eval_seed}**
- Bootstrap iterations: **{bootstrap_iters}**
- Figure formats: **{', '.join(formats)}**

Each policy was evaluated with the environment parameters stored in its own
`meta.json` (`gamma`, `tau`, `eta`, `sigma_noise`, `noise_type`, `gust_scale`,
`horizon`, `act_limit`, safety limits, and VecNormalize statistics when present).
The initial-state schedule and process-noise stream are shared across algorithms,
training seeds, and fixtures through the CRN seed/episode schedule.

## Main metric convention

The environment reports costs `c`, with reward `r=-c`.  The table and main plots
use the manuscript reward convention:

```text
OptGap(pi; s0) = (V_r^pi(s0) - V_r^*(s0)) / (|V_r^*(s0)| + eps)
Regret_T       = J_hat^pi(s0) - J*(s0)
               = sum_k gamma^k c_k + gamma^T J*(s_T) - J*(s0)
```

The analytic value is `V_r^*(s0) = -J^*(s0)`, where `J^*(s)=s^T P s + b`.
For the policy return used in the main OptGap, the script stores a tail-corrected
estimate:

```text
J_hat = sum_k gamma^k c(s_k, pi(s_k)) + gamma^T J^*(s_T)
V_r^pi_hat = -J_hat
```

The previous on-policy stage-action oracle gap is retained in the CSV as
`discounted_regret_stage_action` for diagnostics, but the main `Regret` column
uses the tail-bootstrapped cost gap above.

The finite-horizon non-tail OptGap is retained in the full trial CSV for audit.

## Key outputs

- `csv/aerial_policy_trials_full.csv`: one row per policy run/CRN episode with main and audit metrics.
- `csv/aerial_all_labels_summary_full.csv`: policy/oracle/zero summary rows.
- `csv/aerial_rollouts_all_labels_long.csv`: per-step rollout diagnostics for policy/oracle/zero.
- `csv/aerial_group_stats_by_p_algo.csv`: means and paired-bootstrap CIs by `(p, algorithm)`.
- `csv/aerial_overall_by_algorithm.csv`: aerial-only overall means and CIs by algorithm.
- `tables/overall_performance_aerial.tex`: LaTeX table for the aerial benchmark.
- `figures/fig_bar_optgap_overall.*`: aerial-only overall OptGap bars.
- `figures/fig_bar_regret_overall.*`: aerial-only overall regret bars.
- `figures/fig_optgap_heatmap_aerial.*`: aerial OptGap heatmap by algorithm and p.
- `figures/fig_regret_heatmap_aerial.*`: aerial regret heatmap by algorithm and p.
- `evaluations/<run>/figs/`: regenerated previous per-run aerial rollout diagnostic plots.
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", type=Path, default=ROOT / "aerial_results" / "runs")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "New results arial")
    ap.add_argument("--episodes", type=int, default=5, help="CRN episodes per trained run; matches the training EvalCallback n_eval_episodes.")
    ap.add_argument("--eval_seed", type=int, default=20260516)
    ap.add_argument("--bootstrap_iters", type=int, default=10_000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--algos", type=str, default=None, help="Comma-separated algorithm filter, e.g. SAC,TD3")
    ap.add_argument("--p_values", type=str, default=None, help="Comma-separated p filter, e.g. 0.6,0.7")
    ap.add_argument("--limit_runs", type=int, default=None, help="Optional smoke-test limit after sorting/filtering.")
    ap.add_argument("--formats", type=str, default="pdf,svg,png")
    ap.add_argument("--no_per_run_plots", action="store_true", help="Skip regenerating old per-run rollout diagnostic plots.")
    ap.add_argument("--reuse_existing", action="store_true", help="Reuse existing per-run summary/rollout CSVs in out_dir/evaluations when present.")
    args = ap.parse_args()

    out_dir = args.out_dir
    eval_root = out_dir / "evaluations"
    csv_dir = out_dir / "csv"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    ensure_dirs(out_dir, eval_root, csv_dir, tables_dir, figures_dir)

    algos = parse_str_list(args.algos)
    p_values = parse_float_list(args.p_values)
    runs = discover_runs(args.runs_dir, algos=algos, p_values=p_values)
    if args.limit_runs is not None:
        runs = runs[: int(args.limit_runs)]
    if not runs:
        raise SystemExit(f"No complete aerial runs found under {args.runs_dir}")

    formats = tuple(fmt.strip().lower() for fmt in str(args.formats).split(",") if fmt.strip())
    print(f"[info] found {len(runs)} runs")
    print(f"[info] algorithms: {sorted({r.algo for r in runs}, key=algo_rank)}")
    print(f"[info] p values: {sorted({r.p for r in runs})}")

    # Common CRN start states are generated from the first run's env parameters,
    # then injected into every run with options={s0: ...}.  Noise seeds are also
    # identical because rollout envs use eval_seed + episode.
    ref_args = namespace_from_run(runs[0], episodes=args.episodes, eval_seed=args.eval_seed, out_dir=eval_root / runs[0].run_name, device=args.device)
    s0_list = generate_common_s0(ref_args)
    np.savez(csv_dir / "crn_initial_states.npz", s0=np.asarray(s0_list, dtype=float), eval_seed=int(args.eval_seed))

    all_summary: List[Dict[str, Any]] = []
    all_rollouts: List[Dict[str, Any]] = []
    run_manifest: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs, start=1):
        eval_dir = eval_root / run.run_name
        eval_args = namespace_from_run(run, episodes=args.episodes, eval_seed=args.eval_seed, out_dir=eval_dir, device=args.device)
        metadata = run_metadata_columns(run, args.eval_seed, args.episodes)
        print(f"[eval {idx:02d}/{len(runs):02d}] {run.run_name}  algo={run.algo} p={run.p} seed={run.seed}")
        summary_rows, rollout_rows = evaluate_one_run(
            run,
            eval_args,
            s0_list,
            eval_dir,
            make_plots=not args.no_per_run_plots,
            reuse_existing=bool(args.reuse_existing),
        )
        summary_aug = add_metadata([dict(r) for r in summary_rows], metadata)
        rollout_aug = add_metadata([dict(r) for r in rollout_rows], metadata)
        # Rewrite augmented copies in the result directory so each per-run CSV is self-describing.
        write_csv(eval_dir / "summary_aerial_augmented.csv", summary_aug)
        write_csv(eval_dir / "rollouts_aerial_augmented.csv", rollout_aug)
        all_summary.extend(summary_aug)
        all_rollouts.extend(rollout_aug)
        run_manifest.append({**metadata, "eval_dir": str(eval_dir), "model_path": str(run.model_path), "vecnorm_path": str(run.vecnorm_path) if run.vecnorm_path else ""})

    summary_df = pd.DataFrame(all_summary)
    rollouts_df = pd.DataFrame(all_rollouts)
    policy_df = build_policy_trials(summary_df)

    # Save complete CSV outputs.
    summary_df.to_csv(csv_dir / "aerial_all_labels_summary_full.csv", index=False)
    policy_df.to_csv(csv_dir / "aerial_policy_trials_full.csv", index=False)
    rollouts_df.to_csv(csv_dir / "aerial_rollouts_all_labels_long.csv", index=False)
    pd.DataFrame(run_manifest).to_csv(csv_dir / "aerial_run_manifest.csv", index=False)

    by_p_algo, overall = make_stats(policy_df, iters=int(args.bootstrap_iters))
    by_p_algo.to_csv(csv_dir / "aerial_group_stats_by_p_algo.csv", index=False)
    overall.to_csv(csv_dir / "aerial_overall_by_algorithm.csv", index=False)

    table = make_table(policy_df, iters=int(args.bootstrap_iters))
    table.to_csv(tables_dir / "overall_performance_aerial.csv", index=False)
    write_latex_table(table, tables_dir / "overall_performance_aerial.tex")
    write_markdown_table(table, tables_dir / "overall_performance_aerial.md")

    fig_manifest: List[Dict[str, Any]] = []
    fig_manifest.extend(plot_overall_bars(overall, figures_dir, formats))
    fig_manifest.extend(plot_heatmaps(by_p_algo, figures_dir, formats))
    fig_manifest.extend(plot_vs_p_lines(by_p_algo, figures_dir, formats))
    pd.DataFrame(fig_manifest).to_csv(out_dir / "figure_manifest.csv", index=False)

    write_readme(
        out_dir,
        runs,
        episodes=int(args.episodes),
        eval_seed=int(args.eval_seed),
        bootstrap_iters=int(args.bootstrap_iters),
        formats=formats,
    )

    manifest = {
        "runs_dir": str(args.runs_dir),
        "out_dir": str(out_dir),
        "n_runs": len(runs),
        "algorithms": sorted({r.algo for r in runs}, key=algo_rank),
        "p_values": sorted({r.p for r in runs}),
        "train_seeds": sorted({r.seed for r in runs}),
        "episodes": int(args.episodes),
        "eval_seed": int(args.eval_seed),
        "bootstrap_iters": int(args.bootstrap_iters),
        "main_metric": "OptGap uses reward sign with oracle/value tail bootstrap; Regret is tail-bootstrapped cost gap J_hat - J*.",
        "outputs": {
            "policy_trials_csv": str(csv_dir / "aerial_policy_trials_full.csv"),
            "overall_table_tex": str(tables_dir / "overall_performance_aerial.tex"),
            "figures_dir": str(figures_dir),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Console summary.
    print("\n=== Aerial overall by algorithm ===")
    display = overall.copy()
    for col in ["optgap_mean", "optgap_lo", "optgap_hi", "regret_mean", "regret_lo", "regret_hi"]:
        display[col] = display[col].map(sci)
    print(display.to_string(index=False))
    print(f"\n[done] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()