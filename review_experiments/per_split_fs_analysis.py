#!/usr/bin/env python
"""
Per-Fold Feature Selection: Results Compilation & Visualization (R2 #12)
========================================================================

读取 per-fold 实验的结果，生成：
1. 主汇总表（所有 cohort 的 Original vs Per-fold C-index, Δ）
2. 对比柱状图（每 cohort 双柱：Original / Per-fold）
3. Delta 柱状图（含误差棒）
4. 散点图（Original vs Per-fold，每点一个 cohort）
5. 每 cohort 的逐 seed 对比图

数据来源
---------
- Results/per_split_fs/all_cohorts/PRO_{cohort}_results.csv
- Results/per_split_fs/all_cohorts/RNA_{cohort}_results.csv
- Results/per_split_fs/comparison.csv  (单数据集实验旧结果)
"""

import os
import sys
import warnings
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Plotting
import matplotlib
matplotlib.use("Agg")  # No display backend needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Configuration
# ============================================================

RESULTS_DIRS = [
    "Results/per_split_fs/all_cohorts",
    "Results/per_split_fs",  # fallback for single-cohort results
]

OUTPUT_DIR = "Results/per_split_fs/analysis"

# Color scheme
COLOR_ORIGINAL = "#4C72B0"  # blue
COLOR_PERFOLD = "#DD8452"   # orange
COLOR_DELTA_POS = "#55A868" # green (improvement)
COLOR_DELTA_NEG = "#C44E52" # red (degradation)

# Order for display
PRO_COHORTS_ORDERED = ["CCRCC", "GBM", "HaNSCC", "HCC", "LA", "LSCC", "PDA", "UCEC"]
RNA_COHORTS_ORDERED = ["BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LIHC", "LUAD", "LUSC", "OV", "STAD"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ============================================================
# Data Loading
# ============================================================

def load_all_results() -> pd.DataFrame:
    """
    Load all per-cohort results from available result directories.

    Searches RESULTS_DIRS for PRO_*_results.csv and RNA_*_results.csv files,
    and also loads Results/per_split_fs/comparison.csv (single-dataset format).

    Returns:
        Combined DataFrame with columns:
        ['Omics', 'Cohort', 'Knowledge', 'Seed',
         'Original_CIndex', 'PerFold_CIndex', 'Delta']
    """
    all_dfs = []

    single_df = load_single_dataset_results()
    if not single_df.empty:
        def parse_dataset(ds):
            ds = str(ds)
            if 'PRO' in ds or 'pro' in ds or 'HCC' in ds:
                parts = ds.split()
                return ('PRO', parts[0] if len(parts) >= 1 else ds)
            elif 'RNA' in ds or 'rna' in ds or 'LIHC' in ds:
                parts = ds.split()
                return ('RNA', parts[0] if len(parts) >= 1 else ds)
            return ('UNKNOWN', ds)

        omics_cohort = single_df['Dataset'].apply(lambda x: parse_dataset(x))
        single_df.insert(0, 'Omics', omics_cohort.apply(lambda x: x[0]))
        single_df.insert(1, 'Cohort', omics_cohort.apply(lambda x: x[1]))
        all_dfs.append(single_df)

    for res_dir in RESULTS_DIRS:
        if not os.path.isdir(res_dir):
            continue

        for fn in os.listdir(res_dir):
            if not fn.endswith("_results.csv"):
                continue
            if fn.startswith("master"):
                continue

            fp = os.path.join(res_dir, fn)
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"  [WARN] Cannot read {fp}: {e}")
                continue

            # Validate required columns
            required = ['Omics', 'Cohort', 'Seed', 'PerFold_CIndex']
            if not all(c in df.columns for c in required):
                # Try old format (single-dataset script) without Omics column
                if 'Dataset' in df.columns and 'Knowledge' in df.columns:
                    # Map Dataset to Omics + Cohort
                    def parse_dataset(ds):
                        ds = str(ds)
                        if 'PRO' in ds or 'pro' in ds or 'HCC' in ds:
                            # Extract cohort name
                            parts = ds.split()
                            return ('PRO', parts[0] if len(parts) >= 1 else ds)
                        elif 'RNA' in ds or 'rna' in ds or 'LIHC' in ds:
                            parts = ds.split()
                            return ('RNA', parts[0] if len(parts) >= 1 else ds)
                        return ('UNKNOWN', ds)

                    omics_cohort = df['Dataset'].apply(lambda x: parse_dataset(x))
                    df.insert(0, 'Omics', omics_cohort.apply(lambda x: x[0]))
                    df.insert(1, 'Cohort', omics_cohort.apply(lambda x: x[1]))
                else:
                    print(f"  [SKIP] {fp}: missing required columns")
                    continue

            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by (Omics, Cohort, Seed) — keep the last entry
    combined = combined.drop_duplicates(
        subset=['Omics', 'Cohort', 'Seed'], keep='last'
    )

    return combined


def load_single_dataset_results() -> pd.DataFrame:
    """
    Load results from the original single-dataset experiment (comparison.csv).

    Returns:
        DataFrame with single-cohort results (compatible format), or empty.
    """
    fn = "Results/per_split_fs/comparison.csv"
    if os.path.isfile(fn):
        return pd.read_csv(fn)
    return pd.DataFrame()


# ============================================================
# Summary Computation
# ============================================================

def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cohort summary statistics.

    Args:
        df: Results DataFrame with columns
            ['Omics', 'Cohort', 'Seed', 'Original_CIndex', 'PerFold_CIndex', 'Delta'].

    Returns:
        Formatted summary DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []

    for (omics, cohort), group in df.groupby(['Omics', 'Cohort']):
        orig = group['Original_CIndex'].dropna()
        per_fold = group['PerFold_CIndex'].dropna()

        paired = group.dropna(subset=['Original_CIndex', 'PerFold_CIndex'])
        deltas = paired['PerFold_CIndex'].values - paired['Original_CIndex'].values

        knowledge = group['Knowledge'].iloc[0] if 'Knowledge' in group.columns else ''

        rows.append({
            'Omics': omics,
            'Cohort': cohort,
            'Knowledge': knowledge,
            '#Seeds': len(group),
            'Original (mean±std)': (f"{orig.mean():.4f}±{orig.std():.4f}"
                                     if len(orig) > 0 else "N/A"),
            'PerFold (mean±std)': (f"{per_fold.mean():.4f}±{per_fold.std():.4f}"
                                    if len(per_fold) > 0 else "N/A"),
            'Δ mean±std': (f"{deltas.mean():.4f}±{deltas.std():.4f}"
                            if len(deltas) > 0 else "N/A"),
            'Δ min/max': (f"[{deltas.min():.4f}, {deltas.max():.4f}]"
                           if len(deltas) > 0 else "N/A"),
        })

    summary = pd.DataFrame(rows)

    # Sort: PRO first, then RNA, by cohort name
    cohort_order = {c: i for i, c in enumerate(PRO_COHORTS_ORDERED + RNA_COHORTS_ORDERED)}
    summary['_sort'] = summary.apply(
        lambda r: (0 if r['Omics'] == 'PRO' else 1,
                   cohort_order.get(r['Cohort'], 999)),
        axis=1
    )
    summary = summary.sort_values('_sort').drop(columns=['_sort'])

    return summary


def compute_stats_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute numeric summary stats per cohort (for plotting).

    Returns:
        DataFrame with numeric columns: n_seeds, orig_mean, orig_std,
        pf_mean, pf_std, delta_mean, delta_std, delta_se.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []

    for (omics, cohort), group in df.groupby(['Omics', 'Cohort']):
        orig = group['Original_CIndex'].dropna().values
        pf = group['PerFold_CIndex'].dropna().values

        # Get overlapping seeds for paired comparison
        paired = group.dropna(subset=['Original_CIndex', 'PerFold_CIndex'])
        deltas = paired['PerFold_CIndex'].values - paired['Original_CIndex'].values

        rows.append({
            'Omics': omics,
            'Cohort': cohort,
            'n_seeds': len(group),
            'orig_mean': orig.mean() if len(orig) > 0 else np.nan,
            'orig_std': orig.std() if len(orig) > 0 else np.nan,
            'pf_mean': pf.mean() if len(pf) > 0 else np.nan,
            'pf_std': pf.std() if len(pf) > 0 else np.nan,
            'delta_mean': deltas.mean() if len(deltas) > 0 else np.nan,
            'delta_std': deltas.std() if len(deltas) > 0 else np.nan,
            'delta_se': (deltas.std() / np.sqrt(len(deltas))
                          if len(deltas) > 1 else 0),
        })

    stats = pd.DataFrame(rows)

    # Sort by omics then cohort
    cohort_order = {c: i for i, c in enumerate(PRO_COHORTS_ORDERED + RNA_COHORTS_ORDERED)}
    stats['_sort'] = stats.apply(
        lambda r: (0 if r['Omics'] == 'PRO' else 1,
                   cohort_order.get(r['Cohort'], 999)),
        axis=1
    )
    stats = stats.sort_values('_sort').reset_index(drop=True)

    return stats


# ============================================================
# Plotting Functions
# ============================================================

def plot_grouped_bar(
    stats: pd.DataFrame,
    title: str = "Original vs Per-Fold C-index by Cohort",
    fn_save: str = "grouped_bar.png",
):
    """
    Grouped bar chart: for each cohort, two bars (Original / Per-fold).

    Args:
        stats: DataFrame from compute_stats_for_plotting().
        title: Chart title.
        fn_save: Output filename.
    """
    n = len(stats)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 6))

    x = np.arange(n)
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width / 2, stats['orig_mean'].values, width,
                   yerr=stats['orig_std'].values, capsize=3,
                   label='Original (full-data FS)', color=COLOR_ORIGINAL,
                   error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x + width / 2, stats['pf_mean'].values, width,
                   yerr=stats['pf_std'].values, capsize=3,
                   label='Per-fold (train-only FS)', color=COLOR_PERFOLD,
                   error_kw={'linewidth': 1.5})

    # Labels
    xlabels = [f"{r['Omics']}-{r['Cohort']}" for _, r in stats.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("C-index")
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add value labels on top of bars
    for bar, val in zip(bars1, stats['orig_mean'].values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=7,
                    color=COLOR_ORIGINAL)
    for bar, val in zip(bars2, stats['pf_mean'].values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=7,
                    color=COLOR_PERFOLD)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp = os.path.join(OUTPUT_DIR, fn_save)
    fig.savefig(fp, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_delta_bar(
    stats: pd.DataFrame,
    title: str = "C-index Change (Δ) from Per-Fold Feature Selection",
    fn_save: str = "delta_bar.png",
):
    """
    Bar chart of Δ per cohort, with error bars (SE).
    Positive = per-fold better, Negative = original better.

    Args:
        stats: DataFrame from compute_stats_for_plotting().
        title: Chart title.
        fn_save: Output filename.
    """
    n = len(stats)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 6))

    x = np.arange(n)
    deltas = stats['delta_mean'].values
    errors = stats['delta_se'].values

    colors = [COLOR_DELTA_POS if d >= 0 else COLOR_DELTA_NEG for d in deltas]

    bars = ax.bar(x, deltas, yerr=errors, capsize=4, color=colors,
                  error_kw={'linewidth': 1.5})

    xlabels = [f"{r['Omics']}-{r['Cohort']}" for _, r in stats.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Δ C-index (Per-fold − Original)")
    ax.set_title(title, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add value labels
    for bar, val in zip(bars, deltas):
        if not np.isnan(val):
            y_pos = bar.get_height() + (0.005 if val >= 0 else -0.015)
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.4f}", ha='center', va=va, fontsize=7,
                    color='black')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp = os.path.join(OUTPUT_DIR, fn_save)
    fig.savefig(fp, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_scatter(
    stats: pd.DataFrame,
    title: str = "Original vs Per-Fold C-index",
    fn_save: str = "scatter.png",
):
    """
    Scatter plot: Original C-index vs Per-fold C-index, one point per cohort.
    Colors: PRO = blue, RNA = orange.

    Args:
        stats: DataFrame from compute_stats_for_plotting().
        title: Chart title.
        fn_save: Output filename.
    """
    if stats.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    for omics, color, marker in [('PRO', COLOR_ORIGINAL, 'o'),
                                  ('RNA', COLOR_PERFOLD, 's')]:
        subset = stats[stats['Omics'] == omics]
        if subset.empty:
            continue
        ax.scatter(subset['orig_mean'], subset['pf_mean'],
                   c=color, marker=marker, s=80, label=omics,
                   zorder=5, edgecolors='white', linewidth=0.5)

        # Label each point
        for _, row in subset.iterrows():
            ax.annotate(row['Cohort'],
                        (row['orig_mean'], row['pf_mean']),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, alpha=0.8)

    # Diagonal line: y = x
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='y = x')

    ax.set_xlabel("Original C-index (full-data FS)")
    ax.set_ylabel("Per-fold C-index (train-only FS)")
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp = os.path.join(OUTPUT_DIR, fn_save)
    fig.savefig(fp, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_per_seed_comparison(
    df: pd.DataFrame,
    fn_save: str = "per_seed_comparison.png",
):
    """
    Per-seed line plot for each (Omics, Cohort): original vs per-fold C-index.

    Args:
        df: Full results DataFrame.
        fn_save: Output filename.
    """
    groups = list(df.groupby(['Omics', 'Cohort']))
    n_groups = len(groups)
    if n_groups == 0:
        return

    n_cols = 4
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5),
                             squeeze=False)

    for idx, ((omics, cohort), group) in enumerate(groups):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        group = group.sort_values('Seed')
        seeds = group['Seed'].values

        ax.plot(seeds, group['Original_CIndex'].values, 'o-',
                color=COLOR_ORIGINAL, label='Original', markersize=4, linewidth=1)
        ax.plot(seeds, group['PerFold_CIndex'].values, 's-',
                color=COLOR_PERFOLD, label='Per-fold', markersize=4, linewidth=1)

        ax.set_title(f"{omics}-{cohort}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Seed", fontsize=8)
        ax.set_ylabel("C-index", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_groups, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Per-Seed C-index: Original vs Per-Fold Feature Selection",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp = os.path.join(OUTPUT_DIR, fn_save)
    fig.savefig(fp, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {fp}")


# ============================================================
# Main
# ============================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("Per-Fold Feature Selection Analysis")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")

    # Load data
    print("\n[1] Loading results...")
    df = load_all_results()
    print(f"  Loaded {len(df)} rows across {df[['Omics', 'Cohort']].drop_duplicates().shape[0]} cohorts")

    if df.empty:
        print("  No results found. Run per_split_fs_all_cohorts.py first.")
        return

    # Print summary table
    print("\n[2] Summary table:")
    summary = compute_summary_table(df)
    print(summary.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn_table = os.path.join(OUTPUT_DIR, "summary_table.csv")
    summary.to_csv(fn_table, index=False)
    print(f"\n  Table saved: {fn_table}")

    # Compute numeric stats for plotting
    stats = compute_stats_for_plotting(df)

    # Generate plots
    print("\n[3] Generating plots...")
    plot_grouped_bar(stats, fn_save="grouped_bar.png")
    plot_delta_bar(stats, fn_save="delta_bar.png")
    plot_scatter(stats, fn_save="scatter.png")
    plot_per_seed_comparison(df, fn_save="per_seed_comparison.png")

    # Per-omics breakdown plots
    for omics in ['PRO', 'RNA']:
        sub_stats = stats[stats['Omics'] == omics]
        if not sub_stats.empty:
            plot_grouped_bar(
                sub_stats,
                title=f"{omics}: Original vs Per-Fold C-index",
                fn_save=f"{omics.lower()}_grouped_bar.png",
            )
            plot_delta_bar(
                sub_stats,
                title=f"{omics}: C-index Change (Δ) from Per-Fold FS",
                fn_save=f"{omics.lower()}_delta_bar.png",
            )

    # Paired t-test summary
    print("\n[4] Paired t-test analysis (per cohort):")
    for (omics, cohort), group in df.groupby(['Omics', 'Cohort']):
        paired = group.dropna(subset=['Original_CIndex', 'PerFold_CIndex'])
        if len(paired) < 3:
            print(f"  {omics}-{cohort}: insufficient paired data (n={len(paired)})")
            continue

        from scipy import stats as scipy_stats

        t_stat, p_val = scipy_stats.ttest_rel(
            paired['PerFold_CIndex'].values,
            paired['Original_CIndex'].values,
        )
        delta_mean = paired['PerFold_CIndex'].mean() - paired['Original_CIndex'].mean()
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {omics}-{cohort}: Δ={delta_mean:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
