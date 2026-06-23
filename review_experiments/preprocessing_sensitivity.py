#!/usr/bin/env python3
"""
Preprocessing Sensitivity Analysis for HGS-BioAbductor.

Part 1: Cox p-value threshold sensitivity on feature selection and C-index.
Part 2: Z-score before/after PCA comparison for HCC proteomic data with cohort coloring.

Usage:
    python other_experiments/preprocessing_sensitivity.py
    python other_experiments/preprocessing_sensitivity.py --smoke_test

=============================================================================
脚本流程注解:
Part 1 — Cox p-value 阈值敏感性分析:
  1. 加载 HCC PRO 数据 (data/PRO/HCC/dataset.csv, 2257 基因)
  2. 多进程计算所有基因的单变量 Cox p-value (lifelines.CoxPHFitter)
  3. 对每个阈值 [0.01, 0.05, 0.1, top400]:
     a. 根据 p-value 筛选特征
     b. Z-score 归一化
     c. 5 折交叉验证训练 CoxPH 模型
     d. 记录 mean±std C-index
  4. 输出 sensitivity_results.csv

Part 2 — Z-score 前后 PCA 对比:
  1. 读取 HCC 多队列临床数据 (ProteomicsCohorts_clinic.csv)
  2. 提取基因表达列 + cohort 标签, 处理 NaN
  3. PCA (n_components=2) on 原始数据 vs Z-score 后
  4. 绘制对比图, 点按 cohort 着色
=============================================================================
"""

import os
import sys
import argparse
import warnings
import time
from functools import partial
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# add project root to path for potential imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ===========================================================================
# constants
# ===========================================================================

HCC_PRO_DATASET = os.path.join(
    _project_root, "data", "PRO", "HCC", "dataset.csv"
)
CLINIC_DATA = os.path.join(
    "/Backup/home/chenyupeng/DATA/HCC_MultiCohorts",
    "ProteomicsCohorts_clinic.csv",
)
OUTPUT_DIR = os.path.join(_project_root, "Results", "preprocessing_sensitivity")

THRESHOLDS: List[Optional[float]] = [0.01, 0.05, 0.1, None]
TOP_FEATURES = 400
N_FOLDS = 5
RANDOM_SEED = 42

COHORT_NAMES = {
    "Jiang": "Jiang",
    "Gao": "Gao",
    "xing": "Xing",
    "FZ": "FZ",
    "GZ": "GZ",
    "SH": "SH",
}
COHORT_COLORS = {
    "Jiang": "#1f77b4",
    "Gao": "#ff7f0e",
    "xing": "#2ca02c",
    "FZ": "#d62728",
    "GZ": "#9467bd",
    "SH": "#8c564b",
}


# ===========================================================================
# Part 1 helpers – Cox univariate p-value computation
# ===========================================================================

def _univariate_cox_fit(
    data: pd.DataFrame, duration_col: str, event_col: str, feature: str
) -> Tuple[float, str]:
    """Run univariate CoxPH and return (p_value, feature_name)."""
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.0001)
    try:
        cph.fit(df=data, duration_col=duration_col, event_col=event_col,
                formula=feature)
        p = cph.summary.loc[feature, "p"]
    except Exception:
        p = 1.0
    return p, feature


def compute_cox_pvalues(
    feature_matrix: pd.DataFrame,
    time_col: str,
    event_col: str,
    process_num: int = 10,
    max_features: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute univariate Cox p-values for every gene column via multiprocessing.

    Returns a DataFrame with columns ``feature``, ``p_value``, sorted
    ascending by p-value.
    """
    import multiprocessing as mul
    from tqdm import tqdm

    features = feature_matrix.columns.tolist()
    if max_features is not None:
        features = features[:max_features]

    # build combined DataFrame with all features + labels for the fitter
    label_df = pd.DataFrame({
        time_col: feature_matrix.index.map(
            lambda _: None  # placeholder, replaced below
        ),
        event_col: None,
    }, index=feature_matrix.index)

    # ---- fill labels from the original dataset file ----
    full = pd.read_csv(HCC_PRO_DATASET, index_col=0)
    label_df[time_col] = full.loc[feature_matrix.index, time_col]
    label_df[event_col] = full.loc[feature_matrix.index, event_col]

    data = pd.concat([feature_matrix[features], label_df], axis=1)

    reg = partial(_univariate_cox_fit, data, time_col, event_col)
    pvals: List[float] = []

    print(f"  Running univariate Cox on {len(features)} features "
          f"({process_num} processes) ...")
    t0 = time.time()
    with mul.Pool(processes=process_num) as pool:
        with tqdm(total=len(features), desc="Cox univariate") as pbar:
            for p, feat in pool.imap(reg, features):
                pvals.append(p)
                pbar.update()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    result = pd.DataFrame({"feature": features, "p_value": pvals})
    result = result.sort_values("p_value").reset_index(drop=True)
    return result


# ===========================================================================
# Part 1 helpers – train / evaluate multivariate CoxPH
# ===========================================================================

def _train_evaluate_coxph(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> float:
    """Fit CoxPH on train, return C-index on test."""
    from lifelines import CoxPHFitter
    from sksurv.metrics import concordance_index_censored

    # drop any constant columns that cause singular matrix
    gene_cols = [c for c in train_df.columns if c not in (time_col, event_col)]
    keep = [c for c in gene_cols if train_df[c].nunique() > 1]
    if len(keep) < len(gene_cols):
        dropped = set(gene_cols) - set(keep)
        if dropped:
            print(f"    Dropped {len(dropped)} constant-features")

    try:
        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(train_df[keep + [time_col, event_col]],
                duration_col=time_col, event_col=event_col)
        risk = cph.predict_partial_hazard(test_df[keep + [time_col, event_col]])
        ci = concordance_index_censored(
            test_df[event_col].astype(bool).values,
            test_df[time_col].values,
            risk.values,
        )[0]
    except Exception as exc:
        print(f"    CoxPH failed: {exc}")
        ci = 0.5
    return ci


# ===========================================================================
# Part 1 – threshold sensitivity
# ===========================================================================

def _run_sensitivity(smoke_test: bool = False) -> pd.DataFrame:
    """Run threshold sensitivity analysis, return results DataFrame."""
    from sklearn.model_selection import KFold
    from scipy.stats import zscore

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- load data ----
    print("=" * 60)
    print("Loading HCC PRO dataset ...")
    df = pd.read_csv(HCC_PRO_DATASET, index_col=0)
    time_col, event_col = "OS time", "death"
    gene_cols = df.columns[:-2].tolist()
    print(f"  Patients: {df.shape[0]},  Genes: {len(gene_cols)}")
    print(f"  Event rate: {df[event_col].mean():.2%}")

    # ---- compute Cox p-values ----
    if smoke_test:
        max_feat = 200
        n_proc = 4
        thresholds: List[Optional[float]] = [0.01, 0.05]
        n_folds_local = 1
    else:
        max_feat = None
        n_proc = min(16, os.cpu_count() or 8)
        thresholds = list(THRESHOLDS)
        n_folds_local = N_FOLDS

    pval_df = compute_cox_pvalues(
        df[gene_cols], time_col, event_col,
        process_num=n_proc, max_features=max_feat,
    )

    # ---- iterate thresholds ----
    rows: list = []

    for thresh in thresholds:
        if thresh is None:
            selected = pval_df.head(TOP_FEATURES)
            label = "top400"
        else:
            selected = pval_df[pval_df["p_value"] < thresh]
            label = str(thresh)

        n_feat = len(selected)
        print(f"\nThreshold {label}: {n_feat} features selected")

        if n_feat == 0:
            rows.append({"threshold": label, "num_features": 0,
                         "mean_c_index": np.nan, "std_c_index": np.nan})
            continue

        if smoke_test:
            # just record counts, skip CoxPH training
            rows.append({"threshold": label, "num_features": n_feat,
                         "mean_c_index": np.nan, "std_c_index": np.nan})
            continue

        # prepare data with selected features
        sel_genes = selected["feature"].tolist()
        sub = df[sel_genes + [time_col, event_col]].copy()
        kf = KFold(n_splits=n_folds_local, shuffle=True,
                   random_state=RANDOM_SEED)

        # Z-score normalise
        zvals = zscore(sub[sel_genes].values, axis=0, nan_policy="omit")
        sub[sel_genes] = np.nan_to_num(zvals, nan=0.0)
        sub = sub.dropna(subset=[time_col, event_col])

        # 5-fold CV
        ci_scores: List[float] = []
        for train_idx, test_idx in kf.split(sub):
            train_data = sub.iloc[train_idx]
            test_data = sub.iloc[test_idx]
            ci = _train_evaluate_coxph(train_data, test_data,
                                       time_col, event_col)
            ci_scores.append(ci)

        mean_ci = float(np.mean(ci_scores))
        std_ci = float(np.std(ci_scores))
        print(f"  C-index: {mean_ci:.4f} ± {std_ci:.4f}")

        rows.append({"threshold": label, "num_features": n_feat,
                     "mean_c_index": mean_ci, "std_c_index": std_ci})

    return pd.DataFrame(rows)


# ===========================================================================
# Part 2 – PCA before / after Z-score
# ===========================================================================

def _run_pca(smoke_test: bool = False) -> str:
    """Generate side-by-side PCA plot, return path to saved figure."""
    from sklearn.decomposition import PCA
    from scipy.stats import zscore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading multi-cohort clinic data ...")
    clinic = pd.read_csv(CLINIC_DATA, index_col=0)
    gene_cols = clinic.columns[8:].tolist()
    cohort_col = "cohort"

    print(f"  Samples: {clinic.shape[0]},  Genes: {len(gene_cols)}")
    cohorts_present = clinic[cohort_col].unique()
    print(f"  Cohorts: {list(cohorts_present)}")
    for c in cohorts_present:
        print(f"    {COHORT_NAMES.get(c, c)}: {(clinic[cohort_col]==c).sum()}")

    # expression matrix
    expr_raw = clinic[gene_cols].copy()

    # fill NaN with column mean
    expr = expr_raw.fillna(expr_raw.mean())

    # smoke-test: subset
    if smoke_test:
        expr = expr.iloc[:60, :200]

    cohort_labels = clinic.loc[expr.index, cohort_col].values
    unique_cohorts = pd.unique(cohort_labels)

    # ---- PCA helpers ----
    def _do_pca(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(mat)
        return coords, pca.explained_variance_ratio_

    # before Z-score
    print("\nPCA before Z-score ...")
    coords_before, var_before = _do_pca(expr.values)

    # Z-score
    print("Applying Z-score ...")
    mat_z = zscore(expr.values, axis=0)
    mat_z = np.nan_to_num(mat_z, nan=0.0, posinf=0.0, neginf=0.0)

    # after Z-score
    print("PCA after Z-score ...")
    coords_after, var_after = _do_pca(mat_z)

    # ---- plot ----
    print("Generating comparison plot ...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, (coords, var_ratio, subtitle) in enumerate([
        (coords_before, var_before, "Before Z-score"),
        (coords_after, var_after, "After Z-score"),
    ]):
        ax = axes[idx]
        for cohort in unique_cohorts:
            mask = cohort_labels == cohort
            color = COHORT_COLORS.get(cohort, "#333333")
            label = COHORT_NAMES.get(cohort, cohort)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, label=label, alpha=0.65,
                       edgecolor="w", linewidth=0.3, s=28)

        # cohort centroids
        for cohort in unique_cohorts:
            mask = cohort_labels == cohort
            center = coords[mask].mean(axis=0)
            ax.scatter(center[0], center[1], marker="x",
                       c="black", s=120, linewidths=2, zorder=5)

        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)", fontsize=11)
        ax.set_title(f"HCC Proteomics  {subtitle}",
                     fontsize=13, fontweight="bold")
        ax.legend(title="Cohort", fontsize=9, title_fontsize=10,
                  loc="upper right", framealpha=0.85)
        ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pca_before_after.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocessing Sensitivity Analysis – HGS-BioAbductor")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick verification: limited thresholds, folds, "
                             "and features")
    args = parser.parse_args()

    print("=" * 60)
    print("HGS-BioAbductor  Preprocessing Sensitivity Analysis")
    print(f"Smoke test = {args.smoke_test}")
    print(f"Output     = {OUTPUT_DIR}")
    print("=" * 60)

    # ---- Part 1 ----
    print("\n[Part 1] Cox p-value threshold sensitivity")
    res = _run_sensitivity(smoke_test=args.smoke_test)
    csv_path = os.path.join(OUTPUT_DIR, "sensitivity_results.csv")
    res.to_csv(csv_path, index=False)
    print(f"\nThreshold sensitivity results:\n{res.to_string(index=False)}")
    print(f"Saved → {csv_path}")

    # ---- Part 2 ----
    print("\n[Part 2] PCA before/after Z-score")
    _run_pca(smoke_test=args.smoke_test)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
