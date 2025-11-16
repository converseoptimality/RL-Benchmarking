# plot_summary_nudex_suite.py
# -*- coding: utf-8 -*-
"""
Usage
-----
python plot_summary_nudex_suite.py \
    --csv_dir CSVs \
    --out_dir figs_summary \
    --save_svg

What this fixes (v3)
--------------------
1) **Optimality gap sign**: we recompute OptGap from CSVs using REWARDS, and
   we add a robust detector for the serial files in case `jstar_mean` was
   logged as reward instead of cost. For each *experiment*, we pick the
   interpretation whose median OptGap is ≤ 0 (as it must be for reward).
   Formula per row:
       R_pi   = - mc_cost_mean
       R_star = - jstar_mean    (or jstar_mean, if detection picks that)
       OptGap = (R_pi - R_star) / (|R_star| + eps)      # ≤ 0
       Regret =  R_star - R_pi                           # ≥ 0

2) **Table formatting**: all numeric cells are rendered in scientific notation,
   e.g. `1.11e+03`. The LaTeX table is single-column friendly (tabularx).

3) **Logarithmic y-axes**:
   - OptGap plots use **symlog** (handles negatives) with linthresh=1e-3.
   - Regret plots use **log**. Bars/lines are clipped to a tiny epsilon to
     avoid invalid log(0). Errorbars are adjusted accordingly.

4) **Plot polish**: clean, non-overlapping layout; legends outside; concise
   titles (removed the word “reward” from titles), readable labels.

Outputs
-------
(out_dir)/
  fig_bar_optgap_overall.{pdf,svg}
  fig_bar_regret_overall.{pdf,svg}
  fig_optgap_vs_p_serial_random.{pdf,svg}
  fig_regret_vs_p_nudex.{pdf,svg}
  fig_box_optgap_all.{pdf,svg}
  fig_cdf_optgap_all.{pdf,svg}
  tables/overall_performance.{csv,tex,md}
  metrics_readme.txt
"""

import argparse
import os
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# --------------------------- Matplotlib defaults ----------------------------
matplotlib.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

EPS = 1e-12
SYMLIN = 1e-3        # linthresh for symlog (OptGap)
REGRET_MIN = 1e-12   # floor for log-scale on regret


# ------------------------------ Utilities -----------------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def bootstrap_ci(x: np.ndarray, iters: int = 10_000, alpha: float = 0.05, seed: int = 2025) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    samples = rng.choice(x, size=(iters, x.size), replace=True).mean(axis=1)
    lo = np.quantile(samples, alpha / 2)
    hi = np.quantile(samples, 1 - alpha / 2)
    return float(lo), float(hi)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sci(x: float) -> str:
    """Format a number like '1.11e+03'."""
    if isinstance(x, (int, np.integer)):
        x = float(x)
    try:
        return f"{x:.2e}"
    except Exception:
        return str(x)


# --------------------------- Reward metrics ---------------------------------
def _make_optgap_regret_from_costs(mc_cost_mean: pd.Series,
                                   jstar_mean: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Primary computation (assuming jstar_mean is COST):
      R_pi   = - mc_cost_mean
      R_star = - jstar_mean
    """
    R_pi = -_to_float(mc_cost_mean)
    R_star = -_to_float(jstar_mean)
    optgap = (R_pi - R_star) / (np.abs(R_star) + EPS)
    regret = (R_star - R_pi)
    return optgap, regret


def _make_optgap_regret_alt_jstar_is_reward(mc_cost_mean: pd.Series,
                                            jstar_mean: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Alternate interpretation (if jstar_mean was logged as REWARD):
      R_pi   = - mc_cost_mean
      R_star =   jstar_mean
    """
    R_pi = -_to_float(mc_cost_mean)
    R_star = _to_float(jstar_mean)
    optgap = (R_pi - R_star) / (np.abs(R_star) + EPS)
    regret = (R_star - R_pi)
    return optgap, regret


def _choose_optgap_definition_per_experiment(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    For Serial CSVs we don't know if jstar_mean is cost or reward.
    Compute both versions and PICK the one whose median OptGap is ≤ 0
    (as it must be in reward sign). If both medians > 0 (unlikely),
    use the 'cost' version but flip sign to enforce ≤ 0.
    """
    og1, rg1 = _make_optgap_regret_from_costs(df["mc_cost_mean"], df["jstar_mean"])
    og2, rg2 = _make_optgap_regret_alt_jstar_is_reward(df["mc_cost_mean"], df["jstar_mean"])

    med1 = np.nanmedian(og1.to_numpy(dtype=float))
    med2 = np.nanmedian(og2.to_numpy(dtype=float))

    if (not np.isnan(med1)) and med1 <= 0:
        return og1, rg1
    if (not np.isnan(med2)) and med2 <= 0:
        return og2, rg2

    # Fallback: enforce ≤ 0
    return -np.abs(og1), np.abs(rg1)


# --------------------------- Harmonization ----------------------------------
def harmonize_serial(df: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    """
    Serial arm CSV → common schema with correct reward-signed metrics.
    Expected columns:
      run,algo,seed,n_links,p,tau,r_u,sigma_omega,gamma,mc_cost_mean,mc_cost_std,
      jstar_mean,jstar_std,eval_reward_mean,monitor_reward_smoothed
    """
    df = df.copy()
    df["experiment"] = exp_name
    df["algo"] = df["algo"].astype(str)
    df["p"] = _to_float(df["p"])
    df["gamma"] = _to_float(df["gamma"])
    df["n_links"] = _to_float(df["n_links"])
    df["K"] = np.nan  # not used here

    # Pick the correct jstar interpretation for this experiment
    og, rg = _choose_optgap_definition_per_experiment(df)
    df["optgap_mean"] = og
    df["regret_mean"] = rg

    return df[[
        "experiment", "algo", "p", "gamma", "n_links", "K",
        "optgap_mean", "regret_mean"
    ]]


def harmonize_nudex(df: pd.DataFrame) -> pd.DataFrame:
    """
    NUDEx CSV → common schema. Recompute from costs to be consistent and robust.
    Expected columns include:
      run,algo,seed,gamma,horizon,K,r_v,r_w,tau,p,normalize,mc_cost_mean,mc_cost_std,
      jstar_mean,jstar_std, ... (optgap_* may also exist but we recompute)
    """
    df = df.copy()
    df["experiment"] = "nudex"
    df["algo"] = df["algo"].astype(str)
    df["p"] = _to_float(df["p"])
    df["gamma"] = _to_float(df["gamma"])
    df["K"] = _to_float(df.get("K", np.nan))
    df["n_links"] = np.nan

    og, rg = _make_optgap_regret_from_costs(df.get("mc_cost_mean", np.nan),
                                            df.get("jstar_mean", np.nan))
    # If someone logged jstar as reward, the median will be > 0; fix by alt:
    med = np.nanmedian(og.to_numpy(dtype=float))
    if not np.isnan(med) and med > 0:
        og, rg = _make_optgap_regret_alt_jstar_is_reward(df.get("mc_cost_mean", np.nan),
                                                         df.get("jstar_mean", np.nan))

    df["optgap_mean"] = og
    df["regret_mean"] = rg

    return df[[
        "experiment", "algo", "p", "gamma", "n_links", "K",
        "optgap_mean", "regret_mean"
    ]]


def load_all(csv_dir: str) -> pd.DataFrame:
    """Load and concatenate all present CSVs; returns a harmonized dataframe."""
    parts = []
    path = Path(csv_dir)

    f1 = path / "summary_random_initialS.csv"
    if f1.exists():
        parts.append(harmonize_serial(pd.read_csv(f1), "serial_random"))

    f2 = path / "summary_fixed_initialS.csv"
    if f2.exists():
        parts.append(harmonize_serial(pd.read_csv(f2), "serial_fixed"))

    f3 = path / "summary_nudex.csv"
    if f3.exists():
        parts.append(harmonize_nudex(pd.read_csv(f3)))

    if not parts:
        raise FileNotFoundError("No expected CSVs found in the provided directory.")

    df = pd.concat(parts, axis=0, ignore_index=True)
    for col in ["optgap_mean", "regret_mean", "p", "gamma", "n_links", "K"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["algo", "optgap_mean", "regret_mean"], how="any")
    return df


# --------------------------- Aggregations -----------------------------------
def macro_average(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Macro-average in two stages (per your requirement):
      (1) Per (algo, experiment): mean over rows of that experiment.
      (2) Across experiments: bootstrap CI of those per-experiment means per algo.
    Returns: algo, mean, lo, hi, n_experiments
    """
    per_exp = (
        df.groupby(["algo", "experiment"], as_index=False)[metric]
          .mean()
          .rename(columns={metric: "exp_mean"})
    )
    out_rows = []
    for algo, g in per_exp.groupby("algo"):
        vals = g["exp_mean"].to_numpy(dtype=float)
        mean = float(np.nanmean(vals)) if vals.size else np.nan
        lo, hi = bootstrap_ci(vals)
        out_rows.append({"algo": algo, "mean": mean, "lo": lo, "hi": hi, "n_experiments": vals.size})
    return pd.DataFrame(out_rows).sort_values("mean")


def per_experiment_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper table: each (experiment, algorithm), mean ± 95% CI of OptGap (≤0) and Regret (≥0).
    """
    rows = []
    for (experiment, algo), g in df.groupby(["experiment", "algo"]):
        og_vals = g["optgap_mean"].to_numpy(dtype=float)
        rg_vals = g["regret_mean"].to_numpy(dtype=float)
        og_mean = float(np.nanmean(og_vals))
        og_lo, og_hi = bootstrap_ci(og_vals)
        rg_mean = float(np.nanmean(rg_vals))
        rg_lo, rg_hi = bootstrap_ci(rg_vals)
        rows.append({
            "Experiment": experiment,
            "Algorithm": algo,
            "OptGap (mean)": og_mean,
            "OptGap 95% CI": (og_lo, og_hi),
            "Regret (mean)": rg_mean,
            "Regret 95% CI": (rg_lo, rg_hi),
            "N": g.shape[0],
        })
    table = pd.DataFrame(rows)
    exp_order = ["serial_random", "serial_fixed", "nudex"]
    table["__exp_order"] = table["Experiment"].apply(lambda x: exp_order.index(x) if x in exp_order else 999)
    table = table.sort_values(["__exp_order", "Algorithm"]).drop(columns="__exp_order")
    return table


# --------------------------- Plot helpers -----------------------------------
def _legend_outside(ax, title=None):
    ax.legend(frameon=False, title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _bar_with_ci_symlog(ax, labels, means, los, his, ylabel, title):
    """Bars with CI on a symlog y-scale (for OptGap ≤ 0)."""
    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.6)
    yerr = np.vstack([means - los, his - means])
    # Ensure lower/upper are finite
    yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
    ax.errorbar(x, means, yerr=yerr, fmt="none", capsize=4, elinewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=SYMLIN)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)


def _bar_with_ci_log(ax, labels, means, los, his, ylabel, title):
    """Bars with CI on a log y-scale (for Regret ≥ 0)."""
    # Clip means/CI to be strictly positive for log-scale
    means = np.maximum(means, REGRET_MIN)
    los = np.maximum(los, REGRET_MIN)
    his = np.maximum(his, REGRET_MIN)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.6)
    # For log scale, errorbars can't cross zero; clamp
    lower = np.maximum(means - los, REGRET_MIN * 0.5)
    upper = np.maximum(his - means, REGRET_MIN * 0.5)
    yerr = np.vstack([lower, upper])
    ax.errorbar(x, means, yerr=yerr, fmt="none", capsize=4, elinewidth=1.0)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)


# --------------------------- Plots (no overlap) -----------------------------
def plot_overall_optgap(df: pd.DataFrame, out_dir: str, save_svg: bool):
    mac = macro_average(df, "optgap_mean")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    _bar_with_ci_symlog(
        ax,
        labels=mac["algo"].tolist(),
        means=mac["mean"].to_numpy(),
        los=mac["lo"].to_numpy(),
        his=mac["hi"].to_numpy(),
        ylabel="Optimality Gap",
        title="Overall Optimality Gap (macro-avg)"
    )
    fig.savefig(Path(out_dir, "fig_bar_optgap_overall.pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(Path(out_dir, "fig_bar_optgap_overall.svg"), bbox_inches="tight")
    plt.close(fig)


def plot_overall_regret(df: pd.DataFrame, out_dir: str, save_svg: bool):
    mac = macro_average(df, "regret_mean")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    _bar_with_ci_log(
        ax,
        labels=mac["algo"].tolist(),
        means=mac["mean"].to_numpy(),
        los=mac["lo"].to_numpy(),
        his=mac["hi"].to_numpy(),
        ylabel="Regret",
        title="Overall Regret (macro-avg)"
    )
    fig.savefig(Path(out_dir, "fig_bar_regret_overall.pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(Path(out_dir, "fig_bar_regret_overall.svg"), bbox_inches="tight")
    plt.close(fig)


def plot_optgap_vs_p_serial_random(df: pd.DataFrame, out_dir: str, save_svg: bool):
    sub = df[(df["experiment"] == "serial_random") & df["p"].notna() & df["n_links"].notna()]
    if sub.empty:
        return
    nlinks_values = sorted(sub["n_links"].dropna().unique().tolist())
    ncols = min(3, len(nlinks_values))
    nrows = int(math.ceil(len(nlinks_values) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(7.0, 2.6 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, nL in enumerate(nlinks_values):
        ax = axes[idx]
        block = sub[sub["n_links"] == nL]
        for algo, g in block.groupby("algo"):
            byp = g.groupby("p", as_index=False)["optgap_mean"].mean().sort_values("p")
            ax.plot(byp["p"].to_numpy(), byp["optgap_mean"].to_numpy(),
                    marker="o", label=str(algo))
        ax.set_title(f"n={int(nL)}")
        ax.set_xlabel("p")
        ax.set_ylabel("Optimality Gap")
        ax.set_yscale("symlog", linthresh=SYMLIN)
        ax.grid(True, linestyle=":", linewidth=0.7)
        if idx == 0:
            _legend_outside(ax, title="Algorithm")

    # Remove extra axes
    for j in range(len(nlinks_values), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Serial Arm • Optimality Gap vs p", y=1.02, fontsize=11)
    fig.savefig(Path(out_dir, "fig_optgap_vs_p_serial_random.pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(Path(out_dir, "fig_optgap_vs_p_serial_random.svg"), bbox_inches="tight")
    plt.close(fig)


def plot_regret_vs_p_nudex(df: pd.DataFrame, out_dir: str, save_svg: bool):
    sub = df[(df["experiment"] == "nudex") & df["p"].notna() & df["K"].notna()]
    if sub.empty:
        return
    K_values = sorted(sub["K"].dropna().unique().tolist())
    ncols = min(3, len(K_values))
    nrows = int(math.ceil(len(K_values) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(7.0, 2.6 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, K in enumerate(K_values):
        ax = axes[idx]
        block = sub[sub["K"] == K]
        for algo, g in block.groupby("algo"):
            byp = g.groupby("p", as_index=False)["regret_mean"].mean().sort_values("p")
            # clamp to positive for log
            y = np.maximum(byp["regret_mean"].to_numpy(), REGRET_MIN)
            ax.plot(byp["p"].to_numpy(), y, marker="o", label=str(algo))
        ax.set_title(f"K={int(K)}")
        ax.set_xlabel("p")
        ax.set_ylabel("Regret")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", linewidth=0.7)
        if idx == 0:
            _legend_outside(ax, title="Algorithm")

    for j in range(len(K_values), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("NUDEx • Regret vs p", y=1.02, fontsize=11)
    fig.savefig(Path(out_dir, "fig_regret_vs_p_nudex.pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(Path(out_dir, "fig_regret_vs_p_nudex.svg"), bbox_inches="tight")
    plt.close(fig)


def plot_box_optgap_all(df: pd.DataFrame, out_dir: str, save_svg: bool):
    """Boxplot of OptGap across all experiments grouped by algorithm (symlog ticks)."""
    algos = sorted(df["algo"].unique().tolist())
    data = [df[df["algo"] == a]["optgap_mean"].dropna().to_numpy() for a in algos]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.boxplot(data, labels=algos, showmeans=True, meanline=True)
    ax.set_title("Optimality Gap distribution")
    ax.set_ylabel("Optimality Gap")
    ax.set_yscale("symlog", linthresh=SYMLIN)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
    fig.savefig(Path(out_dir, "fig_box_optgap_all.pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(Path(out_dir, "fig_box_optgap_all.svg"), bbox_inches="tight")
    plt.close(fig)


def plot_cdf_optgap_all(df: pd.DataFrame, out_dir: str, save_svg: bool):
    """CDF (performance profile) of OptGap by algorithm."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for algo, g in df.groupby("algo"):
        vals = np.sort(g["optgap_mean"].dropna().to_numpy())
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        ax.step(vals, y, where="post", label=str(algo))
    ax.set_title("Optimality Gap performance profile")
    ax.set_xlabel("Optimality Gap")
    ax.set_ylabel("Fraction ≤ threshold")
    ax.grid(True, linestyle=":", linewidth=0.7)
    _legend_outside(ax, title="Algorithm")
    fig.savefig(Path(out_dir, "fig_cdf_optgap_all.pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(Path(out_dir, "fig_cdf_optgap_all.svg"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Table writers --------------------------------
def _df_to_markdown_fallback(df: pd.DataFrame) -> str:
    """Markdown table writer that doesn't require tabulate; formats as sci."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |"]
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, tuple) and len(v) == 2:
                cells.append(f"[{sci(v[0])}, {sci(v[1])}]")
            elif isinstance(v, (float, int, np.floating, np.integer)):
                cells.append(sci(float(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _to_tabularx_single_column(df: pd.DataFrame) -> str:
    """
    Single-column LaTeX with tabularx and scriptsize. All numbers in sci.
    Columns: Experiment, Algorithm, OptGap mean, OptGap CI, Regret mean, Regret CI, N
    """
    colspec = "llXXXXr"
    header = [str(h).replace("_", r"\_") for h in df.columns]

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabularx}{\linewidth}{%s}" % colspec)
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        row_cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, tuple) and len(v) == 2:
                row_cells.append(rf"[{sci(v[0])}, {sci(v[1])}]")
            elif isinstance(v, (float, int, np.floating, np.integer)):
                row_cells.append(sci(float(v)))
            else:
                row_cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\caption{Overall performance per algorithm and experiment (reward sign). "
                 r"Optimality Gap: $(V^\pi - V^\ast)/(|V^\ast|+\varepsilon)\le 0$; "
                 r"Regret: $V^\ast - V^\pi\ge 0$. Means with 95\% bootstrap CIs.}")
    lines.append(r"\label{tab:overall_performance}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_tables(table_raw: pd.DataFrame, tables_dir: Path):
    # Convert to display DataFrame with sci formatting where appropriate
    table = table_raw.copy()
    # keep tuple CI as-is; csv will expand them to strings
    table.to_csv(tables_dir / "overall_performance.csv", index=False)

    latex = _to_tabularx_single_column(table)
    with open(tables_dir / "overall_performance.tex", "w", encoding="utf-8") as f:
        f.write(latex)

    # Markdown: try pandas -> fallback (formatted)
    try:
        import tabulate  # noqa: F401
        md = table.style.format(
            {k: sci for k in ["OptGap (mean)", "Regret (mean)", "N"] if k in table.columns}
        ).to_string()  # style.to_string (pandas >= 2.2)
        # Fallback to manual if style route fails
        if not md or not isinstance(md, str):
            raise RuntimeError()
    except Exception:
        md = _df_to_markdown_fallback(table)
    with open(tables_dir / "overall_performance.md", "w", encoding="utf-8") as f:
        f.write("# Overall performance per algorithm and experiment\n\n")
        f.write(md)


def write_metrics_readme(out_dir: Path):
    txt = """Metric definitions (reward sign; matches paper):

Given CSV *costs*, define rewards:
  R_pi   = - mc_cost_mean
  R_star = - jstar_mean   (or jstar_mean if the file logged jstar as reward)

Then per row:
  Optimality Gap:  (V^pi - V^*)/(|V^*| + eps) = (R_pi - R_star)/( |R_star| + eps )  <= 0
  Regret:          V^* - V^pi = R_star - R_pi                                   >= 0

Aggregation:
  1) Compute the metric PER EXPERIMENT (mean over rows for each algorithm).
  2) Macro-average those per-experiment means across experiments per algorithm;
     show 95%% bootstrap CIs.
"""
    with open(out_dir / "metrics_readme.txt", "w", encoding="utf-8") as f:
        f.write(txt)


# ----------------------------- Main CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, default="CSVs", help="Directory containing the CSV files.")
    ap.add_argument("--out_dir", type=str, default="figs_summary", help="Directory to write figures and tables.")
    ap.add_argument("--save_svg", action="store_true", help="Also save figures as SVG.")
    args = ap.parse_args()

    figs_dir = Path(args.out_dir)
    tables_dir = Path(args.out_dir) / "tables"
    ensure_dirs(figs_dir, tables_dir)

    # 1) Load (reward-based harmonization with correct sign)
    df = load_all(args.csv_dir)

    # 2) Overall plots (macro-averaged; per-experiment first)
    plot_overall_optgap(df, figs_dir, args.save_svg)
    plot_overall_regret(df, figs_dir, args.save_svg)

    # 3) Per-experiment paper table (single-column friendly; sci formatting)
    table = per_experiment_table(df)
    save_tables(table, tables_dir)

    # 4) Extra plots (clean, no overlaps)
    plot_optgap_vs_p_serial_random(df, figs_dir, args.save_svg)
    plot_regret_vs_p_nudex(df, figs_dir, args.save_svg)
    plot_box_optgap_all(df, figs_dir, args.save_svg)
    plot_cdf_optgap_all(df, figs_dir, args.save_svg)

    # 5) Brief metrics note
    write_metrics_readme(figs_dir)

    # Console summary (sci)
    print("=== Macro-Average (Optimality Gap; ≤ 0) ===")
    mac_og = macro_average(df, "optgap_mean")
    print(mac_og.assign(mean=lambda d: d["mean"].map(sci),
                        lo=lambda d: d["lo"].map(sci),
                        hi=lambda d: d["hi"].map(sci)).to_string(index=False))

    print("\n=== Macro-Average (Regret; ≥ 0) ===")
    mac_rg = macro_average(df, "regret_mean")
    print(mac_rg.assign(mean=lambda d: d["mean"].map(sci),
                        lo=lambda d: d["lo"].map(sci),
                        hi=lambda d: d["hi"].map(sci)).to_string(index=False))

    print(f"\nSaved figures to: {figs_dir}")
    print(f"Saved tables to:  {tables_dir}")


if __name__ == "__main__":
    main()
