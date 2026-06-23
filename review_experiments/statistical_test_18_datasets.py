"""
Paired t-test comparing HGS variants against baseline models across 18 datasets.

Reads per-seed C-index results from existing benchmark files, computes paired
t-tests for each (HGS_variant, baseline_model, dataset) triple, and outputs
detailed + summary CSVs to Results/statistical_tests/.

Usage:
    python other_experiments/statistical_test_18_datasets.py
    python other_experiments/statistical_test_18_datasets.py --smoke_test

=============================================================================
脚本流程注解:
1. 定义 18 个数据集(8 PRO + 10 RNA)、3 种 HGS 变体(STRING/Reactome/hcluster)、
   4 种基线模型(DeepHit/DeepSurv/DRSA/Pnet)
2. 遍历每个 (HGS变体, 基线模型, 数据集) 三元组:
   a. 读取 HGS 的 Results-nt20.csv 和基线的 Results.csv
   b. 解析 CSV 中的首个有效数据段，提取 seeds 0-9 的 test_ci
   c. 对齐 10 对 C-index 值，用 scipy.stats.ttest_rel 做 paired t-test
3. 输出 full_results.csv（每行一个 triple）和 summary.csv（每个模型对汇总）
4. 支持 --smoke_test 模式：限制到 2 个数据集 + 1 个模型对
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import ttest_rel

# ---------------------------------------------------------------------------
# Paths & dataset definitions
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path("Results/Benchmark")

PRO_DATASETS = ["CCRCC", "GBM", "HaNSCC", "HCC", "LA", "LSCC", "PDA", "UCEC"]
RNA_DATASETS = ["BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LIHC", "LUAD", "LUSC", "OV", "STAD"]

HGS_VARIANTS = ["STRING", "Reactome", "hcluster"]
BASELINE_MODELS = ["DeepHit", "DeepSurv", "DRSA", "Pnet"]

OUTPUT_DIR = Path("Results/statistical_tests")
N_SEEDS = 10

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _read_raw_lines(filepath: Path) -> list[str] | None:
    """Return non-empty stripped lines from *filepath*, or *None*."""
    if not filepath.exists():
        return None
    with open(filepath, "r") as fh:
        return [line.rstrip("\n") for line in fh]


def _smart_split(line: str) -> list[str]:
    """Split a CSV line, respecting brackets so commas inside ``[...]`` are
    preserved."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in line:
        if ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def _find_csv_header(lines: list[str], target_col: str = "test_ci") -> int | None:
    """Return index of the first line that looks like a CSV header containing
    *target_col* and the word ``seed``."""
    for idx, line in enumerate(lines):
        if not line:
            continue
        parts = _smart_split(line)
        if "seed" in parts and target_col in parts:
            return idx
    return None


def _extract_data_section(
    lines: list[str], header_idx: int
) -> pd.DataFrame | None:
    """Parse CSV data starting at *header_idx* until a non-data line is
    encountered.
    
    Uses a bracket-aware split to handle list-valued cells such as ``[200,100]``
    that appear in HGS result files.
    """
    header = _smart_split(lines[header_idx])
    data_rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        # Stop on summary / section-delimiter lines
        first_word = stripped.split(",")[0]
        if first_word in (
            "valid",
            "test",
            "final",
            "--UPDATE",
            "---",
        ) or first_word.endswith("Cindex"):
            break
        # Skip bare dict-looking lines (HGS json fragments)
        if stripped.startswith("{"):
            break
        data_rows.append(_smart_split(stripped))

    if not data_rows:
        return None

    # Build DataFrame, coerce numeric columns
    df = pd.DataFrame(data_rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _get_test_ci_series(df: pd.DataFrame) -> pd.Series | None:
    """Extract ``test_ci`` for seeds 0-9 from a parsed data section, keeping
    the first occurrence per seed."""
    if "test_ci" not in df.columns or "seed" not in df.columns:
        return None
    sub = df.loc[:, ["seed", "test_ci"]].dropna(subset=["seed"])
    sub["seed"] = sub["seed"].astype(int)
    sub = sub.drop_duplicates(subset="seed")
    sub = sub[sub["seed"].isin(range(N_SEEDS))].set_index("seed")
    # Ensure all 10 seeds are present
    if len(sub) < N_SEEDS:
        return None
    return sub["test_ci"].sort_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_test_ci(filepath: Path) -> pd.Series | None:
    """Return ``test_ci`` values for seeds 0-9 from the first data section of
    a benchmark result CSV (HGS or baseline).  Returns *None* when the file
    does not exist or cannot be parsed."""
    lines = _read_raw_lines(filepath)
    if lines is None:
        return None

    header_idx = _find_csv_header(lines)
    if header_idx is None:
        warnings.warn(f"No CSV header found in {filepath}")
        return None

    df = _extract_data_section(lines, header_idx)
    if df is None:
        warnings.warn(f"No data rows found in {filepath}")
        return None

    series = _get_test_ci_series(df)
    if series is None:
        warnings.warn(
            f"Insufficient seed coverage (need {N_SEEDS}) in {filepath}"
        )
    return series


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------


def hgs_path(data_type: str, variant: str, dataset: str) -> Path:
    return (
        BENCHMARK_DIR
        / data_type
        / f"HGS-{variant}"
        / "Auto"
        / dataset
        / "Results-nt20.csv"
    )


def baseline_path(data_type: str, model: str, dataset: str) -> Path:
    return BENCHMARK_DIR / data_type / model / dataset / "Results.csv"


# ---------------------------------------------------------------------------
# T-test logic
# ---------------------------------------------------------------------------


def paired_ttest(
    hgs_vals: pd.Series, baseline_vals: pd.Series
) -> dict:
    """Paired t-test between two 10-element series.  Returns a dict with
    ``mean_hgs``, ``mean_baseline``, ``p_value``, ``significant`` (at 0.05),
    and ``mean_diff``."""
    # Align by seed index
    common = hgs_vals.index.intersection(baseline_vals.index)
    if len(common) < 3:
        return {"mean_hgs": float("nan"), "mean_baseline": float("nan"),
                "p_value": float("nan"), "significant": False,
                "mean_diff": float("nan")}

    h = hgs_vals.loc[common].to_numpy(dtype=float)
    b = baseline_vals.loc[common].to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stat, p_val = ttest_rel(h, b, nan_policy="omit")

    return {
        "mean_hgs": float(h.mean()),
        "mean_baseline": float(b.mean()),
        "mean_diff": float(h.mean() - b.mean()),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _collect_results(
    data_type: str,
    datasets: list[str],
    hgs_variants: list[str],
    baseline_models: list[str],
) -> list[dict]:
    """Iterate over all combinations and collect t-test results into a list of
    dicts (one per (hgs, baseline, dataset) triple)."""
    rows: list[dict] = []
    for ds in datasets:
        for hgs_var in hgs_variants:
            hgs_file = hgs_path(data_type, hgs_var, ds)
            hgs_ci = load_test_ci(hgs_file)
            if hgs_ci is None:
                continue

            for bm in baseline_models:
                bl_file = baseline_path(data_type, bm, ds)
                bl_ci = load_test_ci(bl_file)
                if bl_ci is None:
                    continue

                result = paired_ttest(hgs_ci, bl_ci)
                rows.append(
                    {
                        "data_type": data_type,
                        "dataset": ds,
                        "hgs_variant": hgs_var,
                        "baseline_model": bm,
                        **result,
                    }
                )
    return rows


def build_summary(full: pd.DataFrame) -> pd.DataFrame:
    """Aggregate *full* results into per-(hgs_variant, baseline_model)
    summary."""
    if full.empty:
        return pd.DataFrame(
            columns=[
                "hgs_variant",
                "baseline_model",
                "mean_p_value",
                "num_significant",
                "total_datasets",
            ]
        )

    grp = full.groupby(["hgs_variant", "baseline_model"], dropna=False)
    summary = grp.agg(
        mean_p_value=("p_value", "mean"),
        num_significant=("significant", "sum"),
        total_datasets=("significant", "count"),
    ).reset_index()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired t-test: HGS vs baselines across 18 datasets"
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Limit to 2 datasets, 1 HGS variant (STRING), 1 baseline (DeepHit)",
    )
    args = parser.parse_args()

    # ---- smoke-test overrides -----------------------------------------------
    pro_ds = PRO_DATASETS
    rna_ds = RNA_DATASETS
    hgs_variants = HGS_VARIANTS
    baseline_models = BASELINE_MODELS

    if args.smoke_test:
        pro_ds = PRO_DATASETS[:2]
        rna_ds = RNA_DATASETS[:2]
        hgs_variants = ["STRING"]
        baseline_models = ["DeepHit"]
        print("=== SMOKE TEST MODE ===")

    # ---- collect ------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows: list[dict] = []
    for data_type, datasets in [("PRO", pro_ds), ("RNA", rna_ds)]:
        print(f"\nProcessing {data_type} ({len(datasets)} datasets) ...")
        rows = _collect_results(
            data_type, datasets, hgs_variants, baseline_models
        )
        print(f"  Collected {len(rows)} (hgs, baseline, dataset) results.")
        all_rows.extend(rows)

    if not all_rows:
        print("No results collected — nothing to write.")
        sys.exit(1)

    # ---- full results -------------------------------------------------------
    full_df = pd.DataFrame(all_rows)
    full_path = OUTPUT_DIR / "full_results.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\nFull results written to {full_path}")

    # ---- summary ------------------------------------------------------------
    summary_df = build_summary(full_df)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")

    # ---- console preview ----------------------------------------------------
    print("\n--- Summary (first 20 rows) ---")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(summary_df.head(20))

    print("\nDone.")


if __name__ == "__main__":
    main()
