#!/usr/bin/env python3
"""
HCC Multi-Cohort Batch Effect PCA Analysis
============================================
Generates PCA scatter plots for HCC multi-cohort proteomic data to
demonstrate that after Z-score normalization, batch effects between
cohorts are significantly reduced.

Outputs:
    Results/batch_effect_pca/pca_comparison.png
        Side-by-side: raw data (left) vs Z-scored (right)
    Results/batch_effect_pca/pca_after_normalization.png
        Single larger plot of Z-scored PCA for detailed inspection

Usage:
    python other_experiments/hcc_batch_effect_pca.py
    python other_experiments/hcc_batch_effect_pca.py --smoke_test

=============================================================================
脚本流程注解:
1. 读取 HCC 多队列蛋白质组数据 (ProteomicsCohorts_clinic.csv):
   - 1386 samples, 4412 基因, 6 个队列 (FZ, GZ, Gao, Jiang, SH, xing)
2. 数据清洗:
   - 剔除 >50% NaN 的基因列, 剩余 NaN 用列均值填充
3. PCA 降维 (n_components=2) 分别在原始数据和 Z-score 后计算:
   - 原始数据 PCA → PC1/PC2 坐标 + 方差占比
   - Z-score 标准化 (StandardScaler)
   - Z-score 后 PCA → PC1/PC2 坐标 + 方差占比
4. 绘图:
   - pca_comparison.png: 左右子图对比（原始 vs Z-score）
   - pca_after_normalization.png: Z-score 后 PCA 单独大图
   - 点按 cohort 用不同颜色标记
=============================================================================
"""

import sys
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 使用与模型训练一致的数据源：per-cohort 预处理好（80% NaN 过滤 + Cox 排序）
COHORT_FILES = {
    "Gao":  "/Backup/home/chenyupeng/DATA/HCC_MultiCohorts/ProteinCohorts_0.8nafilter_CoxSort/ALL/Gao/dataset.csv",
    "Jiang": "/Backup/home/chenyupeng/DATA/HCC_MultiCohorts/ProteinCohorts_0.8nafilter_CoxSort/ALL/Jiang/dataset.csv",
    "Xing":  "/Backup/home/chenyupeng/DATA/HCC_MultiCohorts/ProteinCohorts_0.8nafilter_CoxSort/ALL/Xing/dataset.csv",
}
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Results", "batch_effect_pca")

# Cohort display mapping (internal → pretty label)
COHORT_DISPLAY = {
    "Gao": "Gao",
    "Jiang": "Jiang",
    "Xing": "Xing",
}

# Qualitative color palette for 3 cohorts (colorblind-friendly)
COHORT_PALETTE = {
    "Gao":  "#2ca02c",   # green
    "Jiang": "#ff7f0e", # orange
    "Xing": "#d62728",  # red
}


def load_and_preprocess() -> tuple:
    """
    按照实际 pipeline 加载 HCC 三队列蛋白质组数据：
    1. 分别读取每个队列的 dataset.csv（已做 80% NaN 过滤 + Cox 排序）
    2. 按行合并（concat axis=0）
    3. 剔除 OS time / death 标签列，仅保留基因表达特征
    4. 剔除 >50% NaN 的基因列
    5. 剩余 NaN 用列均值填充

    Returns
    -------
    expression : pd.DataFrame  shape (n_samples, n_genes)
    cohorts    : list[str]     shape (n_samples,)
    """
    TARGET_COHORTS = {"Gao", "Jiang", "Xing"}
    df_list = []
    label_list = []

    for cohort in TARGET_COHORTS:
        fn = COHORT_FILES[cohort]
        df_cohort = pd.read_csv(fn, index_col=0)
        print(f"[Data] {cohort}: {df_cohort.shape[0]} samples, {df_cohort.shape[1]} genes")
        df_list.append(df_cohort)
        label_list.extend([cohort] * df_cohort.shape[0])

    # 按行合并
    df_all = pd.concat(df_list, axis=0)
    cohorts = label_list
    print(f"\n[Data] Merged: {df_all.shape[0]} samples, {df_all.shape[1]} columns")
    for cohort in TARGET_COHORTS:
        cnt = df_all.shape[0] if cohort == list(TARGET_COHORTS)[-1] else 0
    for cohort in sorted(TARGET_COHORTS):
        cnt = sum(1 for c in cohorts if c == cohort)
        print(f"         {COHORT_DISPLAY[cohort]:>8s}: {cnt:4d}")

    # 剔除最后两列（OS time, death），仅保留基因表达
    expr = df_all.iloc[:, :-2]
    print(f"[Data] Expression columns (genes): {expr.shape[1]}")

    # ── 剔除 >50% NaN 的基因列 ──
    max_nan_frac = 0.5
    n_before = expr.shape[1]
    keep_cols = expr.columns[expr.isnull().mean() < max_nan_frac]
    expr = expr[keep_cols]
    dropped_cols = n_before - expr.shape[1]
    if dropped_cols:
        print(f"[Clean] Dropped {dropped_cols} gene columns (>50 % NaN)")

    # ── 填充剩余 NaN ──
    nan_count = expr.isnull().sum().sum()
    if nan_count > 0:
        expr = expr.fillna(expr.mean())
        print(f"[Clean] Filled {nan_count} NaN values with column mean")

    print(f"[Clean] Final shape: {expr.shape[0]} samples, {expr.shape[1]} genes")
    return expr, np.array(cohorts)


def run_pca(data: np.ndarray) -> tuple:
    """Fit PCA(2) and return (coordinates, explained_variance_ratio)."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)
    return coords, pca.explained_variance_ratio_


def plot_pca_subplot(ax, coords, var_ratio, batch_labels, title: str):
    """
    Draw a PCA scatter on *ax* with points coloured by cohort.

    Parameters
    ----------
    ax          : matplotlib Axes
    coords      : (n, 2) PCA coordinates
    var_ratio   : (2,) explained variance fractions
    batch_labels: 1-D array of cohort string labels
    title       : subplot title
    """
    unique_batches = list(dict.fromkeys(batch_labels))  # preserve order
    for cohort in unique_batches:
        mask = batch_labels == cohort
        colour = COHORT_PALETTE.get(cohort, "#333333")
        display = COHORT_DISPLAY.get(cohort, cohort)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colour,
            label=display,
            alpha=0.65,
            edgecolor="w",
            linewidth=0.4,
            s=18,
        )
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(title="Cohort", fontsize=8, title_fontsize=9, loc="best",
              framealpha=0.85, edgecolor="#cccccc")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_aspect("equal", adjustable="datalim")


def make_comparison_figure(coords_raw, var_raw, coords_z, var_z, cohorts):
    """
    Side-by-side figure: raw PCA (left) vs Z-scored PCA (right).
    Saved to pca_comparison.png.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.suptitle(
        "HCC Proteomic Data: PCA Before and After Z-score Normalization",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plot_pca_subplot(ax1, coords_raw, var_raw, cohorts,
                     "Before Z-score Normalization (Raw)")
    plot_pca_subplot(ax2, coords_z, var_z, cohorts,
                     "After Z-score Normalization")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[Output] Saved: {path}")
    plt.close(fig)


def make_after_figure(coords_z, var_z, cohorts):
    """
    Single larger plot of PCA after Z-score normalisation.
    Saved to pca_after_normalization.png.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_pca_subplot(ax, coords_z, var_z, cohorts,
                     "HCC Proteomic Data: PCA After Z-score Normalization")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_after_normalization.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[Output] Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="HCC multi-cohort batch-effect PCA visualisation")
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Only load data and print shape/cohort distribution, skip PCA")
    args = parser.parse_args()

    print("=" * 65)
    print("  HCC Multi-Cohort Batch Effect PCA Analysis")
    print("=" * 65)

    # ── 1. Load & preprocess ──────────────────────────────────────────
    # 按照实际 pipeline 加载：per-cohort 预处理数据 → 合并 → 50% NaN 过滤
    expr, cohorts = load_and_preprocess()

    if args.smoke_test:
        print("\n[Smoke test] Data loaded successfully — skipping PCA.")
        return

    raw_values = expr.values.astype(np.float64)

    # ── 2. PCA on raw (preprocessed but NOT yet Z-scored) data ─────────
    print("\n[PCA] Computing PCA on raw (preprocessed) expression data ...")
    coords_raw, var_raw = run_pca(raw_values)
    print(f"       Explained variance: PC1={var_raw[0]:.2%}, PC2={var_raw[1]:.2%}")

    # ── 3. Per-sample Z-score normalisation ────────────────────────────
    # 与 pipeline 一致：zscore(axis=1) = 每个样本的所有基因 profile 做标准化
    # 先转置使 axis=1 对应基因维度，scipy.stats.zscore 按行标准化
    print("[PCA] Applying per-sample Z-score normalisation (zscore, axis=1) ...")
    z_values = zscore(raw_values, axis=1)
    # 处理可能的 NaN（zscore 对全零行返回 NaN）
    nan_rows = np.isnan(z_values).any(axis=1)
    if nan_rows.sum() > 0:
        z_values[nan_rows] = 0
        print(f"       Fixed {nan_rows.sum()} all-constant rows (set Z-score to 0)")
    print(f"       Z-scored shape: {z_values.shape}")

    # ── 4. PCA on Z-scored data ───────────────────────────────────────
    print("[PCA] Computing PCA on Z-scored data ...")
    coords_z, var_z = run_pca(z_values)
    print(f"       Explained variance: PC1={var_z[0]:.2%}, PC2={var_z[1]:.2%}")

    # ── 5. Output directory ───────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 6. Figures ────────────────────────────────────────────────────
    print("\n[Plot] Generating figures ...")
    make_comparison_figure(coords_raw, var_raw, coords_z, var_z, cohorts)
    make_after_figure(coords_z, var_z, cohorts)
    print("\nDone.")


if __name__ == "__main__":
    main()
