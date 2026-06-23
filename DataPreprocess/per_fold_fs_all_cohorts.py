#!/usr/bin/env python
"""
All-Cohorts Per-Fold Feature Selection Validation Experiment (R2 #12)
=====================================================================

Extends the per-fold feature selection validation from individual datasets
to all 18 cohorts (8 proteomic + 10 transcriptomic).

核心问题 (R2 #12)
------------------
原始流程中 Cox 回归在全 cohort 上计算 p-value → 排序 → 取 top 400 → 再 split，
导致 test set 的生存信息间接参与了特征选择，造成数据泄漏。

本脚本修复该问题：对每个 cohort，先 split，再仅在 TRAINING SET 上做 Cox 排序。

关键设计
---------
1. 自动从 benchmark Results-nt20.csv 中解析每个 cohort 的最优超参数（AutoML 搜索结果）
2. 复用单数据集脚本 (per_fold_feature_selection_experiment.py) 的核心函数
3. 每 cohort 独立保存结果，支持断点续跑
4. 基因选择缓存：每 seed 的 Cox 选出基因保存到文件，中断后自动跳过已完成 seed

数据流
-------
PRO (STRING):
  data/PRO/{cohort}/dataset.csv
    → split train/valid/test (60/20/20)
    → Cox on TRAIN → rank → top 400
    → STRING gene filter → STRING H → generate_G → HGS train → test C-index

RNA (Reactome):
  data/RNA/{cohort}/feature_matrix.csv  (genes × patients)
  data/RNA/ClinicalDataFrame_DiscreteTime-Cut15Years.csv  (shared survival)
    → Reactome gene intersection
    → transpose → split train/valid/test
    → Cox on TRAIN → rank → top 400
    → Reactome H filter → generate_G → HGS train → test C-index

用法
-----
    # Full experiment on all cohorts
    python DataPreprocess/per_fold_fs_all_cohorts.py

    # PRO only
    python DataPreprocess/per_fold_fs_all_cohorts.py --omics PRO

    # RNA only
    python DataPreprocess/per_fold_fs_all_cohorts.py --omics RNA

    # Smoke test (1 seed, 5 epochs)
    python DataPreprocess/per_fold_fs_all_cohorts.py --smoke_test

    # Resume a specific cohort that was interrupted
    python DataPreprocess/per_fold_fs_all_cohorts.py --omics PRO --cohort HCC

    # Limit Cox genes (for fast testing / avoiding hang)
    python DataPreprocess/per_fold_fs_all_cohorts.py --max_cox_genes 2000

输出
-----
Results/per_fold_fs/all_cohorts/
├── PRO_{cohort}_results.csv       # Per-seed results per cohort
├── PRO_{cohort}_summary.csv       # Summary stats per cohort
├── RNA_{cohort}_results.csv
├── RNA_{cohort}_summary.csv
├── selected_genes/                 # Per-seed gene selection cache
│   ├── PRO_{cohort}_seed{n}.csv
│   └── RNA_{cohort}_seed{n}.csv
└── master_results.csv             # All cohorts combined (final)
"""

import os
import sys
import re
import ast
import time
import json
import warnings
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Models import HGS
from utils.hg_ops import generate_G_from_H, construct_H_STRING
from utils.data_utils import build_hiddens
from Preprocess.DATA_preprocess import cox_feature_selection

# Reuse shared functions from single-dataset experiment script
from DataPreprocess.per_fold_feature_selection_experiment import (
    build_STRING_H,
    build_Reactome_H,
    save_selected_genes,
    load_selected_genes,
    setup_logger,
)


# ============================================================
# Constants
# ============================================================

OUT_DIR = "Results/per_fold_fs/all_cohorts"
GENES_CACHE_DIR = os.path.join(OUT_DIR, "selected_genes")

PRO_COHORTS = ["CCRCC", "GBM", "HaNSCC", "HCC", "LA", "LSCC", "PDA", "UCEC"]
RNA_COHORTS = ["BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LIHC", "LUAD", "LUSC", "OV", "STAD"]

RNA_SURVIVAL_FILE = "data/RNA/ClinicalDataFrame_DiscreteTime-Cut15Years.csv"
REACTOME_H1_TPL = "data/PriorKnow/Reactome/reactome_P{layer}/H1.csv"

BENCHMARK_PRO_TPL = "Results/Benchmark/PRO/HGS-STRING/Auto/{cohort}/Results-nt20.csv"
BENCHMARK_RNA_TPL = "Results/Benchmark/RNA/HGS-Reactome/Auto/{cohort}/Results-nt20.csv"

# Default GPU memory threshold (fraction of total memory)
GPU_MEM_FRACTION = 0.9


# ============================================================
# Section 1: Best HP Config Parser
# ============================================================

def parse_best_config(fn_results: str) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Parse the best hyperparameter + PK config from a benchmark Results-nt20.csv.

    The file contains multiple blocks separated by '------------------------------'.
    Each block has:
        {'n_hid': ..., ...}       ← hyperparameter dict
        {'method': ..., ...}      ← prior knowledge config
        dataset,seed,...           ← data rows

    Returns (hp_dict, pk_dict) from the first valid block.

    Args:
        fn_results: Path to Results-nt20.csv from original benchmark.

    Returns:
        Tuple of (hp_dict, pk_dict), or (None, None) if not found.
    """
    if not os.path.isfile(fn_results):
        return None, None

    with open(fn_results, 'r') as f:
        content = f.read()

    # Split by 30-dash separator
    blocks = content.split('-' * 30)

    for block in blocks:
        lines = block.strip().split('\n')
        hp_dict = None
        pk_dict = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Handle "Hyperparameters: {'n_hid': ...}" format (first block)
            if line.startswith('Hyperparameters:'):
                dict_str = line[len('Hyperparameters:'):].strip()
                try:
                    d = ast.literal_eval(dict_str)
                    if isinstance(d, dict) and 'n_hid' in d:
                        hp_dict = d
                        continue
                except (SyntaxError, ValueError):
                    pass

            # Handle bare "{'n_hid': ...}" format (subsequent blocks)
            if hp_dict is None and line.startswith('{') and '\'n_hid\'' in line:
                try:
                    d = ast.literal_eval(line)
                    if isinstance(d, dict) and 'n_hid' in d:
                        hp_dict = d
                        continue
                except (SyntaxError, ValueError):
                    pass

            # Handle PK config: "{'method': 'layer_range', ...}"
            if line.startswith('{') and '\'method\'' in line and '\'type_know\'' in line:
                try:
                    d = ast.literal_eval(line)
                    if isinstance(d, dict) and 'method' in d:
                        pk_dict = d
                        continue
                except (SyntaxError, ValueError):
                    pass

        if hp_dict is not None and pk_dict is not None:
            return hp_dict, pk_dict

    return None, None


def extract_experiment_config(hp_dict: dict, pk_dict: dict) -> dict:
    """
    Extract experiment-relevant parameters from full HP and PK dicts.

    Args:
        hp_dict: Full hyperparameter dict from benchmark results.
        pk_dict: Full prior-knowledge config dict.

    Returns:
        Config dict with all necessary values for the per-fold experiment.
    """
    cfg = {}

    # --- Model architecture ---
    cfg['n_hid'] = hp_dict.get('n_hid', 200)
    cfg['MLP_hiddens'] = hp_dict.get('MLP_hiddens', [cfg['n_hid']])
    cfg['predict_hiddens'] = hp_dict.get('predict_hiddens',
                                          [cfg['n_hid'], cfg['n_hid'] // 2])
    cfg['depth'] = hp_dict.get('depth', 2)
    cfg['AGG'] = hp_dict.get('AGG', 'noAGG')
    cfg['ACT'] = hp_dict.get('ACT', 'LeakyReLU')
    cfg['num_hgat'] = hp_dict.get('num_hgat', 'single')
    cfg['type_atten'] = hp_dict.get('type_atten', 'additive')
    cfg['edge_pooling'] = hp_dict.get('edge_pooling', True)
    cfg['pooling_method'] = hp_dict.get('pooling_method', 'linear')
    cfg['num_min'] = hp_dict.get('num_min', 25)
    cfg['HG_BN'] = hp_dict.get('HG_BN', True)
    cfg['dropout'] = hp_dict.get('dropout', 0.5)
    cfg['RESNET'] = hp_dict.get('RESNET', False)

    # --- Training ---
    cfg['lr'] = hp_dict.get('lr', 0.01)
    cfg['l2'] = hp_dict.get('l2', 0.005)
    cfg['glr'] = hp_dict.get('glr', 0)
    cfg['batch_size'] = hp_dict.get('batch_size', 16)
    cfg['epochs'] = hp_dict.get('epochs', 50)
    cfg['print_freq'] = hp_dict.get('print_freq', 10)
    cfg['update_freq'] = hp_dict.get('update_freq', 1)
    cfg['gamma'] = hp_dict.get('gamma', 0.99)
    cfg['patience'] = hp_dict.get('patience', 5)
    cfg['score'] = hp_dict.get('score', 'ci')
    cfg['metric_update'] = hp_dict.get('metric_update', 'score')
    cfg['train_scheme'] = hp_dict.get('train_scheme', 'final_epoch')
    cfg['loss_w'] = hp_dict.get('loss_w', {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1})

    # --- Data ---
    cfg['cox_num'] = hp_dict.get('cox_num', 400)
    cfg['end_point'] = hp_dict.get('end_point', 'OS_60')

    # --- Model metadata ---
    cfg['model'] = hp_dict.get('model', 'HGS')

    # --- Prior knowledge ---
    cfg['type_know'] = pk_dict.get('type_know', 'STRING')
    cfg['knowledge_method'] = pk_dict.get('method', 'layer_range')
    cfg['divisor'] = pk_dict.get('divisor', 8)
    cfg['layer_STRING'] = pk_dict.get('layer_STRING', None)
    cfg['layer_Reactome'] = pk_dict.get('layer_Reactome', None)

    return cfg


# ============================================================
# Section 2: Original Results Parser (wraps the shared version)
# ============================================================

def parse_original_seed_results(fn_results: str, cfg: dict) -> Dict[int, float]:
    """
    Parse original benchmark per-seed results for matching HP config.

    Uses the shared parse_original_seed_results from the single-dataset script,
    constructing the target_hp dict from the experiment config.

    Args:
        fn_results: Path to Results-nt20.csv.
        cfg: Experiment config dict (from extract_experiment_config).

    Returns:
        Dict of {seed: test_c_index}.
    """
    if not os.path.isfile(fn_results):
        return {}

    # Build target HP match dict (keys shared by parse_original_seed_results)
    target_hp = {
        'n_hid': cfg['n_hid'],
        'lr': cfg['lr'],
        'l2': cfg['l2'],
        'glr': cfg['glr'],
        'AGG': cfg['AGG'],
    }

    # Import the shared parser
    from DataPreprocess.per_fold_feature_selection_experiment import \
        parse_original_seed_results as _parse_func

    return _parse_func(fn_results, target_hp)


# ============================================================
# Section 3: PRO Experiment Runner
# ============================================================

def run_pro_cohort(
    cohort: str,
    cfg: dict,
    device: str = "cuda:0",
    cox_processes: int = 60,
    num_seeds: int = 10,
    epochs: int = 50,
    max_cox_genes: int = 0,
    smoke_test: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Run per-fold feature selection experiment for a single PRO cohort.

    Args:
        cohort: Cohort name (e.g., "HCC", "CCRCC").
        cfg: Experiment config dict.
        device: CUDA device string.
        cox_processes: Parallel workers for Cox regression.
        num_seeds: Number of seeds (0 to num_seeds-1).
        epochs: Training epochs per seed.
        max_cox_genes: If >0, limit Cox to N random genes for speed.
        smoke_test: If True, run only seed=0 with 5 epochs.
        logger: Logger instance.

    Returns:
        DataFrame with per-seed comparison results.
    """
    if logger is None:
        logger = setup_logger(f"pro_{cohort.lower()}")

    # Override epochs for smoke test
    actual_epochs = 5 if smoke_test else epochs
    cfg = dict(cfg)  # shallow copy to avoid mutating original
    cfg['epochs'] = actual_epochs

    logger.info("=" * 60)
    logger.info(f"PRO {cohort} STRING — Per-fold Feature Selection")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Load full gene expression data
    # ---------------------------------------------------------------
    fn_data = f"data/PRO/{cohort}/dataset.csv"
    if not os.path.isfile(fn_data):
        logger.error(f"Data file not found: {fn_data}")
        return pd.DataFrame()

    data_df = pd.read_csv(fn_data, index_col=0)
    logger.info(f"Loaded full data: {data_df.shape}")
    logger.info(f"Total genes: {data_df.shape[1] - 2}, Patients: {data_df.shape[0]}")

    # ---------------------------------------------------------------
    # Step 2: Parse original benchmark results for comparison
    # ---------------------------------------------------------------
    fn_original = BENCHMARK_PRO_TPL.format(cohort=cohort)
    if not os.path.isfile(fn_original):
        logger.warning(f"Original results not found: {fn_original}")
        original_results = {}
    else:
        original_results = parse_original_seed_results(fn_original, cfg)
        logger.info(f"Parsed {len(original_results)} original seed results")

    # ---------------------------------------------------------------
    # Step 3: Run per-fold experiment for each seed
    # ---------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GENES_CACHE_DIR, exist_ok=True)

    # PK config for STRING H construction
    pk_config = {
        'method': cfg['knowledge_method'],
        'type_know': 'STRING',
        'divisor': cfg['divisor'],
        'layer_STRING': cfg['layer_STRING'],
    }

    results = []
    seeds_to_run = [0] if smoke_test else range(num_seeds)

    for seed in seeds_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Seed {seed}/{max(seeds_to_run)}")

        # --- Step 3a: Split ---
        label = data_df.iloc[:, -1]  # "death" column
        data_train_val, data_test, y_train_val, y_test = train_test_split(
            data_df, label, test_size=0.2, random_state=seed,
            shuffle=True, stratify=label,
        )
        data_train, data_valid, _, _ = train_test_split(
            data_train_val, y_train_val, test_size=0.25,
            random_state=seed, shuffle=True, stratify=y_train_val,
        )
        logger.info(f"Train: {data_train.shape[0]}, Valid: {data_valid.shape[0]}, "
                     f"Test: {data_test.shape[0]}")

        # --- Step 3b: Per-fold Cox on TRAINING ONLY ---
        seed_label = f"PRO_{cohort}_seed{seed}"
        selected_genes = load_selected_genes(seed_label)

        if selected_genes is not None:
            logger.info(f"Loaded {len(selected_genes)} cached genes (skipping Cox)")
            top_400 = selected_genes
            has_cox_result = False  # already cached
        else:
            feat_train = data_train.iloc[:, :-2]
            te_train = data_train.iloc[:, -2:]

            # Optional gene sampling for speed
            if max_cox_genes > 0 and feat_train.shape[1] > max_cox_genes:
                rng = np.random.default_rng(42)
                sampled = rng.choice(feat_train.columns, max_cox_genes, replace=False)
                feat_train = feat_train[sampled]
                logger.info(f"Limited to {max_cox_genes} random genes for Cox")

            logger.info(f"Cox regression: {feat_train.shape[1]} genes × {feat_train.shape[0]} patients")
            start_time = time.time()
            _, p_values = cox_feature_selection(
                time_col="OS time", event_col="death",
                feature_matrix=feat_train, label_matrix=te_train,
                process_num=cox_processes,
            )
            elapsed = time.time() - start_time
            logger.info(f"Cox completed in {elapsed:.1f}s")

            gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
            top_400 = gene_ranking.index[:400].tolist()
            logger.info("Selected top 400 genes from training-set Cox ranking")
            has_cox_result = True

        # --- Step 3c: Filter splits to selected genes ---
        data_train_filt = data_train[top_400 + ["OS time", "death"]]
        data_valid_filt = data_valid[top_400 + ["OS time", "death"]]
        data_test_filt = data_test[top_400 + ["OS time", "death"]]

        # --- Step 3d: Build STRING H ---
        try:
            H, valid_genes = build_STRING_H(data_train_filt, cohort, pk_config)
        except Exception as e:
            logger.error(f"STRING H construction failed: {e}")
            continue

        logger.info(f"Built STRING H: {H.shape}")

        if has_cox_result and max_cox_genes == 0:
            save_selected_genes(seed_label, valid_genes)

        n_dropped = len(top_400) - len(valid_genes)
        if n_dropped > 0:
            logger.info(f"Filtering to {len(valid_genes)} STRING-available genes")
            data_train_filt = data_train_filt[valid_genes + ["OS time", "death"]]
            data_valid_filt = data_valid_filt[valid_genes + ["OS time", "death"]]
            data_test_filt = data_test_filt[valid_genes + ["OS time", "death"]]

        # --- Step 3e: Build graph G ---
        G = generate_G_from_H(H.T) if cfg["edge_pooling"] else generate_G_from_H(H)

        # --- Step 3f: t_obs ---
        t_obs = data_train_filt["OS time"].max() + 2

        # --- Step 3g: pooling_hiddens ---
        cfg['pooling_hiddens'] = build_hiddens(H.shape[1], pk_config['divisor'])
        logger.info(f"pooling_hiddens: {cfg['pooling_hiddens']}")

        # --- Step 3h: Train HGS ---
        fn_ckpt = os.path.join(OUT_DIR, f"PRO_{cohort}_seed{seed}")
        logger.info(f"Training HGS (epochs={cfg['epochs']})...")

        try:
            model = HGS(
                cfg,
                data_train=data_train_filt.values,
                data_eval=data_valid_filt.values,
                data_test=data_test_filt.values,
                H=H, fn_ckpt=fn_ckpt, t_obs=t_obs, G=G, seed=seed,
            )
            model = model.cuda()

            optimizer = optim.Adam(
                model.parameters(), lr=cfg['lr'], weight_decay=cfg['l2'],
            )

            model.fit(
                optimizer=optimizer, logger=logger,
                num_epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                loss_dict=cfg['loss_w'],
            )
        except Exception as e:
            logger.error(f"Training failed for seed {seed}: {e}")
            per_fold_ci = float('nan')
        else:
            # --- Extract test C-index from checkpoint ---
            ckpt_path = f'{fn_ckpt}.ckpt'
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                per_fold_ci = ckpt['final_test_ci']
            else:
                logger.error(f"Checkpoint not found: {ckpt_path}")
                per_fold_ci = float('nan')

        original_ci = original_results.get(seed, None)
        delta = (per_fold_ci - original_ci) if original_ci else None

        logger.info(f"  Original C-index:  {original_ci:.4f}" if original_ci
                     else "  Original: N/A")
        logger.info(f"  Per-fold C-index:  {per_fold_ci:.4f}")
        if delta is not None:
            logger.info(f"  Delta:             {delta:+.4f}")

        results.append({
            'Omics': 'PRO',
            'Cohort': cohort,
            'Knowledge': 'STRING',
            'Seed': seed,
            'Original_CIndex': original_ci,
            'PerFold_CIndex': per_fold_ci,
            'Delta': delta,
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Save per-cohort results
    fn_out = os.path.join(OUT_DIR, f"PRO_{cohort}_results.csv")
    results_df.to_csv(fn_out, index=False)
    logger.info(f"Results saved to {fn_out}")

    return results_df


# ============================================================
# Section 4: RNA Experiment Runner
# ============================================================

def run_rna_cohort(
    cohort: str,
    cfg: dict,
    device: str = "cuda:0",
    cox_processes: int = 60,
    num_seeds: int = 10,
    epochs: int = 50,
    max_cox_genes: int = 0,
    smoke_test: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Run per-fold feature selection experiment for a single RNA cohort.

    Args:
        cohort: Cohort name (e.g., "LIHC", "BLCA").
        cfg: Experiment config dict.
        device: CUDA device string.
        cox_processes: Parallel workers for Cox regression.
        num_seeds: Number of seeds (0 to num_seeds-1).
        epochs: Training epochs per seed.
        max_cox_genes: If >0, limit Cox to N random genes for speed.
        smoke_test: If True, run only seed=0 with 5 epochs.
        logger: Logger instance.

    Returns:
        DataFrame with per-seed comparison results.
    """
    if logger is None:
        logger = setup_logger(f"rna_{cohort.lower()}")

    actual_epochs = 5 if smoke_test else epochs
    cfg = dict(cfg)
    cfg['epochs'] = actual_epochs

    logger.info("=" * 60)
    logger.info(f"RNA {cohort} Reactome — Per-fold Feature Selection")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Load RNA expression and survival data
    # ---------------------------------------------------------------
    fn_data = f"data/RNA/{cohort}/feature_matrix.csv"
    if not os.path.isfile(fn_data):
        logger.error(f"Data file not found: {fn_data}")
        return pd.DataFrame()

    if not os.path.isfile(RNA_SURVIVAL_FILE):
        logger.error(f"Survival file not found: {RNA_SURVIVAL_FILE}")
        return pd.DataFrame()

    feature_matrix = pd.read_csv(fn_data, index_col=0)
    logger.info(f"Loaded feature matrix: {feature_matrix.shape}")

    survival_df = pd.read_csv(RNA_SURVIVAL_FILE, index_col=0)
    survival_df = survival_df.loc[
        survival_df["PatientID"].isin(feature_matrix.columns)
    ]
    survival_df = survival_df.loc[~survival_df.duplicated()]
    logger.info(f"Survival data: {len(survival_df)} patients")

    # ---------------------------------------------------------------
    # Step 2: Load Reactome H1 and intersect genes
    # ---------------------------------------------------------------
    layer_r = cfg.get('layer_Reactome')
    if layer_r is None:
        # Fallback: use a reasonable default
        layer_r = 8
        cfg['layer_Reactome'] = 8

    fn_H_path = REACTOME_H1_TPL.format(layer=layer_r)
    if not os.path.isfile(fn_H_path):
        logger.error(f"Reactome H1 not found: {fn_H_path}")
        return pd.DataFrame()

    H_reactome = pd.read_csv(fn_H_path, index_col=0)
    logger.info(f"Full Reactome H1: {H_reactome.shape}")

    # Intersect genes present in both expression data and Reactome
    common_genes = feature_matrix.index.intersection(H_reactome.index)
    feature_matrix = feature_matrix.loc[common_genes]
    logger.info(f"After Reactome intersection: {feature_matrix.shape[0]} genes")

    # ---------------------------------------------------------------
    # Step 3: Parse original benchmark results
    # ---------------------------------------------------------------
    fn_original = BENCHMARK_RNA_TPL.format(cohort=cohort)
    if not os.path.isfile(fn_original):
        logger.warning(f"Original results not found: {fn_original}")
        original_results = {}
    else:
        original_results = parse_original_seed_results(fn_original, cfg)
        logger.info(f"Parsed {len(original_results)} original seed results")

    # ---------------------------------------------------------------
    # Step 4: Run per-fold experiment for each seed
    # ---------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GENES_CACHE_DIR, exist_ok=True)

    pk_config = {
        'method': cfg['knowledge_method'],
        'type_know': 'Reactome',
        'divisor': cfg['divisor'],
        'layer_Reactome': layer_r,
    }

    results = []
    seeds_to_run = [0] if smoke_test else range(num_seeds)

    for seed in seeds_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Seed {seed}/{max(seeds_to_run)}")

        # --- Step 4a: Build patient DataFrame ---
        data_df = feature_matrix.T.copy()
        surv_subset = survival_df.set_index("PatientID")
        surv_subset = surv_subset.loc[surv_subset.index.isin(data_df.index)]
        data_df = data_df.loc[data_df.index.isin(surv_subset.index)]
        data_df["time"] = surv_subset.loc[data_df.index, cfg["end_point"]].values
        data_df["event"] = surv_subset.loc[data_df.index, "OS Status"].values

        logger.info(f"Patient DataFrame: {data_df.shape}")

        # --- Step 4b: Split ---
        label = data_df["event"]
        data_train_val, data_test, y_train_val, y_test = train_test_split(
            data_df, label, test_size=0.2, random_state=seed,
            shuffle=True, stratify=label,
        )
        data_train, data_valid, _, _ = train_test_split(
            data_train_val, y_train_val, test_size=0.25,
            random_state=seed, shuffle=True, stratify=y_train_val,
        )
        logger.info(f"Train: {data_train.shape[0]}, Valid: {data_valid.shape[0]}, "
                     f"Test: {data_test.shape[0]}")

        # --- Step 4c: Per-fold Cox on TRAINING ONLY ---
        seed_label = f"RNA_{cohort}_seed{seed}"
        selected_genes = load_selected_genes(seed_label)

        if selected_genes is not None:
            logger.info(f"Loaded {len(selected_genes)} cached genes (skipping Cox)")
            top_400 = selected_genes
        else:
            feat_train = data_train.iloc[:, :-2]
            te_train = data_train.iloc[:, -2:]

            if max_cox_genes > 0 and feat_train.shape[1] > max_cox_genes:
                rng = np.random.default_rng(42)
                sampled = rng.choice(feat_train.columns, max_cox_genes, replace=False)
                feat_train = feat_train[sampled]
                logger.info(f"Limited to {max_cox_genes} random genes for Cox")

            logger.info(f"Cox regression: {feat_train.shape[1]} genes × {feat_train.shape[0]} patients")
            start_time = time.time()
            _, p_values = cox_feature_selection(
                time_col="time", event_col="event",
                feature_matrix=feat_train, label_matrix=te_train,
                process_num=cox_processes,
            )
            elapsed = time.time() - start_time
            logger.info(f"Cox completed in {elapsed:.1f}s")

            gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
            top_400 = gene_ranking.index[:400].tolist()
            logger.info("Selected top 400 genes from training-set Cox ranking")

            if max_cox_genes == 0:
                save_selected_genes(seed_label, top_400)

        # --- Step 4d: Filter splits ---
        data_train_filt = data_train[top_400 + ["time", "event"]]
        data_valid_filt = data_valid[top_400 + ["time", "event"]]
        data_test_filt = data_test[top_400 + ["time", "event"]]

        # --- Step 4e: Build Reactome H ---
        H = build_Reactome_H(top_400, H_reactome)
        logger.info(f"Built Reactome H: {H.shape}")

        if H.shape[1] == 0:
            logger.error("Empty Reactome H — no pathways contain selected genes. Skipping seed.")
            continue

        # --- Step 4f: Build graph G ---
        G = generate_G_from_H(H.T) if cfg["edge_pooling"] else generate_G_from_H(H)
        t_obs = data_train_filt["time"].max() + 2

        # --- Step 4g: pooling_hiddens ---
        cfg['pooling_hiddens'] = build_hiddens(H.shape[1], pk_config['divisor'])
        logger.info(f"pooling_hiddens: {cfg['pooling_hiddens']}")

        # --- Step 4h: Train HGS ---
        fn_ckpt = os.path.join(OUT_DIR, f"RNA_{cohort}_seed{seed}")
        logger.info(f"Training HGS (epochs={cfg['epochs']})...")

        try:
            model = HGS(
                cfg,
                data_train=data_train_filt.values,
                data_eval=data_valid_filt.values,
                data_test=data_test_filt.values,
                H=H, fn_ckpt=fn_ckpt, t_obs=t_obs, G=G, seed=seed,
            )
            model = model.cuda()

            optimizer = optim.Adam(
                model.parameters(), lr=cfg['lr'], weight_decay=cfg['l2'],
            )

            model.fit(
                optimizer=optimizer, logger=logger,
                num_epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                loss_dict=cfg['loss_w'],
            )
        except Exception as e:
            logger.error(f"Training failed for seed {seed}: {e}")
            per_fold_ci = float('nan')
        else:
            ckpt_path = f'{fn_ckpt}.ckpt'
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                per_fold_ci = ckpt['final_test_ci']
            else:
                logger.error(f"Checkpoint not found: {ckpt_path}")
                per_fold_ci = float('nan')

        original_ci = original_results.get(seed, None)
        delta = (per_fold_ci - original_ci) if original_ci else None

        logger.info(f"  Original C-index:  {original_ci:.4f}" if original_ci
                     else "  Original: N/A")
        logger.info(f"  Per-fold C-index:  {per_fold_ci:.4f}")
        if delta is not None:
            logger.info(f"  Delta:             {delta:+.4f}")

        results.append({
            'Omics': 'RNA',
            'Cohort': cohort,
            'Knowledge': 'Reactome',
            'Seed': seed,
            'Original_CIndex': original_ci,
            'PerFold_CIndex': per_fold_ci,
            'Delta': delta,
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    fn_out = os.path.join(OUT_DIR, f"RNA_{cohort}_results.csv")
    results_df.to_csv(fn_out, index=False)
    logger.info(f"Results saved to {fn_out}")

    return results_df


# ============================================================
# Section 5: Summary Computation
# ============================================================

def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics (mean ± std) per (Omics, Cohort).

    Args:
        results_df: Per-seed results DataFrame with columns
                    ['Omics', 'Cohort', 'Seed', 'Original_CIndex',
                     'PerFold_CIndex', 'Delta'].

    Returns:
        Summary DataFrame with mean/std per cohort.
    """
    if results_df.empty:
        return pd.DataFrame()

    summary_rows = []

    for (omics, cohort), group in results_df.groupby(['Omics', 'Cohort']):
        orig = group['Original_CIndex'].dropna()
        per_fold = group['PerFold_CIndex'].dropna()

        paired = group.dropna(subset=['Original_CIndex', 'PerFold_CIndex'])
        deltas = paired['PerFold_CIndex'].values - paired['Original_CIndex'].values

        knowledge = group['Knowledge'].iloc[0]

        summary_rows.append({
            'Omics': omics,
            'Cohort': cohort,
            'Knowledge': knowledge,
            'N_Seeds': len(group),
            'Original_Mean': f"{orig.mean():.4f}" if len(orig) > 0 else "N/A",
            'Original_Std': f"{orig.std():.4f}" if len(orig) > 0 else "N/A",
            'PerFold_Mean': f"{per_fold.mean():.4f}" if len(per_fold) > 0 else "N/A",
            'PerFold_Std': f"{per_fold.std():.4f}" if len(per_fold) > 0 else "N/A",
            'Delta_Mean': f"{deltas.mean():.4f}" if len(deltas) > 0 else "N/A",
            'Delta_Std': f"{deltas.std():.4f}" if len(deltas) > 0 else "N/A",
            'Delta_Min': f"{deltas.min():.4f}" if len(deltas) > 0 else "N/A",
            'Delta_Max': f"{deltas.max():.4f}" if len(deltas) > 0 else "N/A",
        })

    return pd.DataFrame(summary_rows)


def save_master_results(all_results: List[pd.DataFrame],
                        logger: logging.Logger) -> str:
    """
    Combine all cohort results into a master file and summary.

    Args:
        all_results: List of per-cohort DataFrames.
        logger: Logger instance.

    Returns:
        Path to master results file.
    """
    if not all_results:
        logger.warning("No results to save.")
        return ""

    master_df = pd.concat(all_results, ignore_index=True)
    fn_master = os.path.join(OUT_DIR, "master_results.csv")
    master_df.to_csv(fn_master, index=False)
    logger.info(f"\nMaster results saved to {fn_master}")

    # Summary
    summary_df = compute_summary(master_df)
    fn_summary = os.path.join(OUT_DIR, "master_summary.csv")
    summary_df.to_csv(fn_summary, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("MASTER SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info(f"\nSummary saved to {fn_summary}")

    return fn_master


# ============================================================
# Section 6: CLI
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="All-cohorts per-fold feature selection validation (R2 #12)"
    )
    parser.add_argument(
        "--omics", type=str, choices=["PRO", "RNA", "all"], default="all",
        help="Run PRO, RNA, or both (default: all)",
    )
    parser.add_argument(
        "--cohort", type=str, default=None,
        help="Specific cohort to run (e.g. 'HCC', 'LIHC'). Runs all if not specified.",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=10,
        help="Number of random seeds (default: 10)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--cox_processes", type=int, default=60,
        help="Parallel workers for Cox regression (default: 60)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per model (default: 50)",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run only seed=0 with 5 epochs",
    )
    parser.add_argument(
        "--max_cox_genes", type=int, default=0,
        help="Limit Cox to N random genes (0=all, default: 0)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip cohorts whose results file already exists (resume mode)",
    )
    return parser.parse_args()


def is_cohort_complete(omics: str, cohort: str) -> bool:
    """Check if a cohort's full results file exists."""
    fn = os.path.join(OUT_DIR, f"{omics}_{cohort}_results.csv")
    return os.path.isfile(fn)


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logger("per_fold_fs_all", log_dir="logs")
    logger.info(f"Arguments: {args}")
    logger.info(f"Output directory: {OUT_DIR}")

    all_results = []

    # Determine which cohorts to run
    target_omics = []
    if args.omics in ("PRO", "all"):
        target_omics.append("PRO")
    if args.omics in ("RNA", "all"):
        target_omics.append("RNA")

    for omics in target_omics:
        cohorts = PRO_COHORTS if omics == "PRO" else RNA_COHORTS
        if args.cohort is not None:
            if args.cohort not in cohorts:
                logger.warning(f"Cohort '{args.cohort}' not found in {omics}")
                continue
            cohorts = [args.cohort]

        for cohort in cohorts:
            # Skip if already completed (resume mode)
            if args.skip_existing and is_cohort_complete(omics, cohort):
                logger.info(f"[SKIP] {omics} {cohort} — results file exists")
                fn_existing = os.path.join(OUT_DIR, f"{omics}_{cohort}_results.csv")
                df_existing = pd.read_csv(fn_existing)
                if not df_existing.empty:
                    all_results.append(df_existing)
                continue

            # Parse best config from benchmark results
            if omics == "PRO":
                fn_benchmark = BENCHMARK_PRO_TPL.format(cohort=cohort)
            else:
                fn_benchmark = BENCHMARK_RNA_TPL.format(cohort=cohort)

            hp_dict, pk_dict = parse_best_config(fn_benchmark)

            if hp_dict is None or pk_dict is None:
                logger.warning(f"[SKIP] {omics} {cohort} — no benchmark config found at {fn_benchmark}")
                continue

            cfg = extract_experiment_config(hp_dict, pk_dict)
            logger.info(f"[CONFIG] {omics} {cohort}: "
                        f"n_hid={cfg['n_hid']}, lr={cfg['lr']}, l2={cfg['l2']}, "
                        f"glr={cfg['glr']}, AGG={cfg['AGG']}, "
                        f"div={cfg['divisor']}, "
                        f"layer={cfg.get('layer_STRING') or cfg.get('layer_Reactome')}")

            # Run experiment
            try:
                if omics == "PRO":
                    df_cohort = run_pro_cohort(
                        cohort=cohort, cfg=cfg,
                        device=args.device,
                        cox_processes=args.cox_processes,
                        num_seeds=args.num_seeds,
                        epochs=args.epochs,
                        max_cox_genes=args.max_cox_genes,
                        smoke_test=args.smoke_test,
                        logger=logger,
                    )
                else:
                    df_cohort = run_rna_cohort(
                        cohort=cohort, cfg=cfg,
                        device=args.device,
                        cox_processes=args.cox_processes,
                        num_seeds=args.num_seeds,
                        epochs=args.epochs,
                        max_cox_genes=args.max_cox_genes,
                        smoke_test=args.smoke_test,
                        logger=logger,
                    )
            except Exception as e:
                logger.error(f"Experiment failed for {omics} {cohort}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

            if not df_cohort.empty:
                all_results.append(df_cohort)

            # Save per-cohort summary
            summary_df = compute_summary(df_cohort)
            fn_summary = os.path.join(OUT_DIR, f"{omics}_{cohort}_summary.csv")
            summary_df.to_csv(fn_summary, index=False)
            logger.info(f"Summary saved to {fn_summary}")

    # Save master combined results
    if all_results:
        save_master_results(all_results, logger)
    else:
        logger.warning("No results generated.")

    logger.info("\n Experiment completed.")


if __name__ == "__main__":
    main()
