#!/usr/bin/env python
"""
All-Models Per-Fold Feature Selection Validation — HCC Only (R2 #12)
=====================================================================

对 HCC PRO 和 LIHC RNA 两个案例，在所有 benchmark 模型 × 知识类型上执行
per-fold 特征选择验证实验。

覆盖组合
---------
HCC PRO:
  - HGS-STRING   (HGS + STRING 超图)
  - HGS-Reactome (HGS + Reactome 超图)
  - HGS-hcluster (HGS + hcluster 超图)
  - DeepSurv
  - DeepHit
  - DRSA
  - Pnet

LIHC RNA:
  - HGS-STRING   (HGS + STRING 超图)
  - HGS-Reactome (HGS + Reactome 超图)
  - HGS-hcluster (HGS + hcluster 超图)
  - DeepSurv
  - DeepHit
  - DRSA
  - Pnet

核心逻辑
---------
对所有模型共用同一个 per-fold Cox 特征选择结果（同一 seed 下相同的
training-set top 400 基因），之后各模型用各自的超参和架构独立训练。

输出
-----
Results/per_split_fs/hcc_all_models/
├── {Modality}_{Knowledge}_{Model}_results.csv    # Per-seed 结果
├── hcc_all_models_summary.csv                    # 汇总表
└── selected_genes/                               # 基因选择缓存

用法
-----
    # 全量实验
    python review_experiments/per_split_fs_all_models.py

    # 只跑特定模型
    python review_experiments/per_split_fs_all_models.py --models HGS-STRING DeepSurv

    # 只跑 PRO 或 RNA
    python review_experiments/per_split_fs_all_models.py --omics PRO

    # 冒烟测试
    python review_experiments/per_split_fs_all_models.py --smoke_test
"""

import os
import sys
import re
import ast
import copy
import time
import json
import warnings
import logging
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Models import HGS, DeepSurv, DeepHit, DRSA, Pnet
from utils.hg_ops import generate_G_from_H, construct_H_STRING
from utils.data_utils import build_hiddens, data_split
from Preprocess.DATA_preprocess import cox_feature_selection

from review_experiments.per_split_feature_selection_experiment import (
    save_selected_genes,
    load_selected_genes,
    setup_logger,
)


# ============================================================
# Constants
# ============================================================

OUT_DIR = "Results/per_split_fs/hcc_all_models"
GENES_CACHE_DIR = os.path.join(OUT_DIR, "selected_genes")

# Available models and knowledge types
HGS_KNOWLEDGE_TYPES = ["STRING", "Reactome", "hcluster"]
BASELINE_MODELS = ["DeepSurv", "DeepHit", "DRSA", "Pnet"]
ALL_MODELS = [f"HGS-{k}" for k in HGS_KNOWLEDGE_TYPES] + BASELINE_MODELS

# RNA
RNA_SURVIVAL_FILE = "data/RNA/ClinicalDataFrame_DiscreteTime-Cut15Years.csv"
REACTOME_H1_TPL = "data/PriorKnow/Reactome/reactome_P{layer}/H1.csv"

# Benchmark paths
BENCHMARK_HGS_TPL = "Results/Benchmark/{omics}/HGS-{knowledge}/Auto/{cohort}/Results-nt20.csv"
BENCHMARK_DL_TPL = "Results/Benchmark/{omics}/{model}/{cohort}/Results.csv"

# hcluster: weighted, divisor = ?
HCLUSTER_DIVISOR = 8  # default, check from benchmark config

GPU_MEM_FRACTION = 0.9


# ============================================================
# Section 1: Best HP Parser (shared across models)
# ============================================================

def parse_hgs_best_config(fn_results: str) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Parse HGS best HP + PK config from Results-nt20.csv.
    Same logic as in per_split_fs_all_cohorts.py.
    """
    if not os.path.isfile(fn_results):
        return None, None
    with open(fn_results) as f:
        content = f.read()

    blocks = content.split('-' * 30)
    for block in blocks:
        lines = block.strip().split('\n')
        hp_dict = None
        pk_dict = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Hyperparameters:'):
                try:
                    d = ast.literal_eval(line[len('Hyperparameters:'):].strip())
                    if isinstance(d, dict) and 'n_hid' in d:
                        hp_dict = d
                        continue
                except:
                    pass
            if hp_dict is None and line.startswith('{') and '\'n_hid\'' in line:
                try:
                    d = ast.literal_eval(line)
                    if isinstance(d, dict) and 'n_hid' in d:
                        hp_dict = d
                        continue
                except:
                    pass
            if line.startswith('{') and '\'method\'' in line and '\'type_know\'' in line:
                try:
                    d = ast.literal_eval(line)
                    if isinstance(d, dict) and 'method' in d:
                        pk_dict = d
                        continue
                except:
                    pass
        if hp_dict and pk_dict:
            return hp_dict, pk_dict
    return None, None


def parse_hgs_config(hp_dict: dict, pk_dict: dict) -> dict:
    """Extract HGS experiment config from HP + PK dicts."""
    cfg = {}
    cfg['n_hid'] = hp_dict.get('n_hid', 200)
    cfg['MLP_hiddens'] = hp_dict.get('MLP_hiddens', [cfg['n_hid']])
    cfg['predict_hiddens'] = hp_dict.get('predict_hiddens', [cfg['n_hid'], cfg['n_hid'] // 2])
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
    cfg['lr'] = hp_dict.get('lr', 0.01)
    cfg['l2'] = hp_dict.get('l2', 0.005)
    cfg['glr'] = hp_dict.get('glr', 0)
    cfg['batch_size'] = hp_dict.get('batch_size', 16)
    cfg['epochs'] = 50
    cfg['print_freq'] = hp_dict.get('print_freq', 10)
    cfg['update_freq'] = hp_dict.get('update_freq', 1)
    cfg['gamma'] = hp_dict.get('gamma', 0.99)
    cfg['patience'] = hp_dict.get('patience', 5)
    cfg['score'] = hp_dict.get('score', 'ci')
    cfg['metric_update'] = hp_dict.get('metric_update', 'score')
    cfg['train_scheme'] = hp_dict.get('train_scheme', 'final_epoch')
    cfg['loss_w'] = hp_dict.get('loss_w', {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1})
    cfg['cox_num'] = hp_dict.get('cox_num', 400)
    cfg['end_point'] = hp_dict.get('end_point', 'OS_60')
    cfg['model'] = hp_dict.get('model', 'HGS')
    cfg['type_know'] = pk_dict.get('type_know', 'STRING')
    cfg['knowledge_method'] = pk_dict.get('method', 'layer_range')
    cfg['divisor'] = pk_dict.get('divisor', 8)
    cfg['layer_STRING'] = pk_dict.get('layer_STRING', None)
    cfg['layer_Reactome'] = pk_dict.get('layer_Reactome', None)
    return cfg


def parse_baseline_best_hp(fn_results: str, model_name: str) -> Tuple[float, float, int]:
    """
    Parse baseline model's Results.csv and find best HP config (by mean val_ci).

    Args:
        fn_results: Path to Results.csv.
        model_name: Model name (for identifying cohort column).

    Returns:
        (best_l2, best_lr, best_nl) — hyperparameters with highest mean val_ci.
    """
    if not os.path.isfile(fn_results):
        return None, None, None

    df = pd.read_csv(fn_results, comment='G')
    if df.empty:
        return None, None, None

    # Group by (L2, lr, nl) and compute mean val_ci
    hp_groups = df.groupby(['L2', 'lr', 'nl'])['val_ci'].mean()
    if hp_groups.empty:
        return None, None, None

    best = hp_groups.idxmax()
    return best[0], best[1], best[2]


def parse_baseline_original_results(fn_results: str, target_l2: float,
                                     target_lr: float, target_nl: int) -> Dict[int, float]:
    """
    Parse baseline Results.csv for per-seed test_ci of a specific HP config.

    Returns {seed: test_ci}.
    """
    if not os.path.isfile(fn_results):
        return {}
    df = pd.read_csv(fn_results, comment='G')
    if df.empty:
        return {}

    # Filter to target HP
    mask = (df['L2'] == target_l2) & (df['lr'] == target_lr) & (df['nl'] == target_nl)
    subset = df[mask]
    return dict(zip(subset['seed'], subset['test_ci']))


# ============================================================
# Section 2: Hypergraph Construction
# ============================================================

def build_STRING_H_from_genes(selected_genes: List[str], cohort: str,
                               pk_config: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Build STRING H from selected gene list (used with per-fold Cox results).
    Replicates build_STRING_H() from the single-dataset script.
    """
    genes_STRING = pd.read_csv("data/PriorKnow/STRING/clusters.protein.ensg.csv"
                               )['protein_id'].to_list()
    genes_STRING = sorted(list(set(genes_STRING)))

    gene_series = pd.Series(selected_genes)
    gene_set = gene_series[gene_series.isin(genes_STRING)]

    fn_H = (f"data/PriorKnow/STRING/sorted/{pk_config['method']}/"
            f"{cohort}-Level{pk_config['layer_STRING']}-H.csv")

    if os.path.isfile(fn_H):
        H = pd.read_csv(fn_H, index_col=0)
    else:
        H = construct_H_STRING(gene_set, layer_STRING=pk_config['layer_STRING'])
        os.makedirs(os.path.dirname(fn_H), exist_ok=True)
        H.to_csv(fn_H)

    edges_sorted = H.columns.sort_values()
    valid_genes = gene_set.intersection(H.index)
    H = H.loc[valid_genes, edges_sorted]
    H = H.loc[:, H.sum(axis=0) != 0]
    return H.values, valid_genes.tolist()


def build_Reactome_H_from_genes(selected_genes: List[str],
                                 H_reactome: pd.DataFrame) -> np.ndarray:
    """Build Reactome H from selected genes (same as original)."""
    H = H_reactome.loc[H_reactome.index.isin(selected_genes), :]
    H = H.loc[:, H.sum(axis=0) != 0]
    return H.values


def build_hcluster_H(selected_genes: List[str], cohort: str,
                     omics: str) -> np.ndarray:
    """
    Load hcluster (hierarchical clustering) H matrix.
    hcluster H is pre-computed and stored per cohort.
    Format: data/PriorKnow/hcluster/{omics}/{cohort}/H.csv
    """
    fn_h = f"data/PriorKnow/hcluster/{omics}/{cohort}/H.csv"
    if not os.path.isfile(fn_h):
        logger = logging.getLogger("build_hcluster")
        logger.error(f"hcluster H not found: {fn_h}")
        return None

    H_full = pd.read_csv(fn_h, index_col=0)

    # Filter to selected genes
    valid_genes = [g for g in selected_genes if g in H_full.index]
    if len(valid_genes) == 0:
        return None

    H = H_full.loc[valid_genes, :]
    H = H.loc[:, H.sum(axis=0) != 0]
    return H.values


# ============================================================
# Section 3: Per-Fold Cox (shared across models)
# ============================================================

def run_per_split_cox(data_train: pd.DataFrame, data_valid: pd.DataFrame,
                     data_test: pd.DataFrame, cox_processes: int,
                     max_cox_genes: int = 0,
                     time_col: str = "OS time", event_col: str = "death",
                     is_rna: bool = False) -> Tuple:
    """
    Run per-fold Cox on training set, return filtered data and selected genes.

    For RNA: time_col="time", event_col="event".
    For PRO: time_col="OS time", event_col="death".

    Returns:
        (data_train_filt, data_valid_filt, data_test_filt, top_400, p_values_series)
    """
    feat_train = data_train.iloc[:, :-2]
    te_train = data_train.iloc[:, -2:]

    if max_cox_genes > 0 and feat_train.shape[1] > max_cox_genes:
        rng = np.random.default_rng(42)
        sampled = rng.choice(feat_train.columns, max_cox_genes, replace=False)
        feat_train = feat_train[sampled]

    _, p_values = cox_feature_selection(
        time_col=time_col, event_col=event_col,
        feature_matrix=feat_train, label_matrix=te_train,
        process_num=cox_processes,
    )
    gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
    top_400 = gene_ranking.index[:400].tolist()

    # Filter splits
    extra_cols = [time_col, event_col] if not is_rna else ["time", "event"]
    data_train_filt = data_train[top_400 + extra_cols]
    data_valid_filt = data_valid[top_400 + extra_cols]
    data_test_filt = data_test[top_400 + extra_cols]

    return data_train_filt, data_valid_filt, data_test_filt, top_400, gene_ranking


# ============================================================
# Section 4: Model Training
# ============================================================

def train_hgs(cfg: dict, data_train, data_valid, data_test,
              H: np.ndarray, G: np.ndarray, t_obs: float,
              device: str, fn_ckpt: str, seed: int,
              logger: logging.Logger) -> float:
    """Train HGS model, return test C-index."""
    if H.shape[1] == 0:
        logger.error("Empty H matrix, cannot train HGS")
        return float('nan')

    cfg['pooling_hiddens'] = build_hiddens(H.shape[1], cfg.get('divisor', 8))

    model = HGS(
        cfg,
        data_train=data_train.values,
        data_eval=data_valid.values,
        data_test=data_test.values,
        H=H, fn_ckpt=fn_ckpt, t_obs=t_obs, G=G, seed=seed,
    )
    model = model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['l2'],
    )
    model.fit(
        optimizer=optimizer, logger=logger,
        num_epochs=cfg.get("epochs", 50),
        batch_size=cfg["batch_size"],
        loss_dict=cfg['loss_w'],
    )

    ckpt_path = f'{fn_ckpt}.ckpt'
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        return ckpt['final_test_ci']
    return float('nan')


def train_baseline(model_name: str, data_train, data_valid, data_test,
                   num_layers: int, lr: float, l2: float, batch_size: int,
                   device: str, fn_ckpt: str, seed: int,
                   t_obs: float = None, pathway_mask=None,
                   logger: logging.Logger = None) -> float:
    """
    Train a baseline DL model (DeepSurv/DeepHit/DRSA/Pnet), return test C-index.

    All baseline models share the same .fit() interface.
    """
    model_params = {
        'data_train': data_train.values,
        'data_eval': data_valid.values,
        'data_test': data_test.values,
        'nn_seed': seed,
        'dropout': 0.5,
        'fn': fn_ckpt,
    }

    if model_name == "DeepSurv":
        model = DeepSurv(**model_params, ACT="LeakyReLU", NumLayers=num_layers)
    elif model_name == "DeepHit":
        if t_obs is None:
            t_obs = data_train.iloc[:, -2].max() + 2
        model = DeepHit(**model_params, ACT="LeakyReLU", NumLayers=num_layers, t_obs=t_obs)
    elif model_name == "DRSA":
        if t_obs is None:
            t_obs = data_train.iloc[:, -2].max() + 2
        model = DRSA(**model_params, lstm_layers=num_layers, t_obs=t_obs)
    elif model_name == "Pnet":
        if t_obs is None:
            t_obs = data_train.iloc[:, -2].max() + 2
        model = Pnet(**model_params, pathway_mask=pathway_mask, ACT="tanh", t_obs=t_obs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # All baseline models use .fit() with the same signature
    model.fit(optimizer=optimizer, logger=logger, num_epochs=50, batch_size=batch_size)

    ckpt_path = f'{fn_ckpt}.ckpt'
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        return ckpt['final_test_ci']
    return float('nan')


def _get_pnet_pathway_mask(data_df: pd.DataFrame, omics: str) -> np.ndarray:
    """
    Build PNET pathway mask from data.
    Replicates logic from baseline_DL_GS.py.
    """
    from utils.data_utils import get_BINN_Pathways
    data_tmp = data_df.copy()
    # Rename columns for get_BINN_Pathways compatibility
    pathway_mask, _ = get_BINN_Pathways(data_tmp, 4)
    return pathway_mask


# ============================================================
# Section 5: Experiment Runner
# ============================================================

def run_hcc_experiment(
    omics: str,
    models: List[str],
    device: str = "cuda:0",
    cox_processes: int = 60,
    num_seeds: int = 10,
    epochs: int = 50,
    smoke_test: bool = False,
    max_cox_genes: int = 0,
    logger: Optional[logging.Logger] = None,
) -> List[pd.DataFrame]:
    """
    Run per-fold experiment for HCC PRO or LIHC RNA across all requested models.

    Args:
        omics: "PRO" or "RNA".
        models: List of model names (e.g., ["HGS-STRING", "DeepSurv"]).
        device: CUDA device.
        cox_processes: Parallel workers for Cox.
        num_seeds: Number of seeds.
        epochs: Training epochs.
        smoke_test: Run only seed=0 with 5 epochs.
        max_cox_genes: Limit Cox genes (for quick testing).
        logger: Logger instance.

    Returns:
        List of per-model result DataFrames.
    """
    cohort = "HCC" if omics == "PRO" else "LIHC"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{omics} {cohort} — All-Models Per-Fold Experiment")
    logger.info(f"Models: {models}")
    logger.info(f"{'=' * 60}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GENES_CACHE_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Pre-load data (varies by omics type)
    # ---------------------------------------------------------------
    if omics == "PRO":
        fn_data = f"data/PRO/HCC/dataset.csv"
        data_all = pd.read_csv(fn_data, index_col=0)
        time_col, event_col = "OS time", "death"
        logger.info(f"PRO HCC data: {data_all.shape}")
    else:
        fn_data = "data/RNA/LIHC/feature_matrix.csv"
        feature_matrix = pd.read_csv(fn_data, index_col=0)
        survival_df = pd.read_csv(RNA_SURVIVAL_FILE, index_col=0)
        survival_df = survival_df.loc[survival_df["PatientID"].isin(feature_matrix.columns)]
        survival_df = survival_df.loc[~survival_df.duplicated()]
        time_col, event_col = "time", "event"
        logger.info(f"RNA LIHC data: {feature_matrix.shape}")

        # Pre-load Reactome H1 for HGS-Reactome
        H_reactome_full = None
        # Will be loaded lazily when needed for HGS-Reactome

    # ---------------------------------------------------------------
    # For each model, prepare and run experiment
    # ---------------------------------------------------------------
    all_results = []
    actual_epochs = 5 if smoke_test else epochs

    for model_name in models:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Model: {model_name}")

        # Determine model type and knowledge
        if model_name.startswith("HGS-"):
            base_model = "HGS"
            knowledge = model_name[len("HGS-"):]
            model_label = f"HGS-{knowledge}"
        else:
            base_model = model_name
            knowledge = None
            model_label = model_name

        # ---------------------------------------------------------------
        # Prepare model-specific configs
        # ---------------------------------------------------------------
        if base_model == "HGS":
            # Parse benchmark results for this HGS config
            fn_benchmark = BENCHMARK_HGS_TPL.format(
                omics=omics, knowledge=knowledge, cohort=cohort
            )
            hp_dict, pk_dict = parse_hgs_best_config(fn_benchmark)
            if hp_dict is None:
                logger.warning(f"No benchmark config for {model_label}, skipping")
                continue
            cfg = parse_hgs_config(hp_dict, pk_dict)
            cfg['epochs'] = actual_epochs
        else:
            # Baseline model — parse best HP from results
            fn_benchmark = BENCHMARK_DL_TPL.format(
                omics=omics, model=model_name, cohort=cohort
            )
            best_l2, best_lr, best_nl = parse_baseline_best_hp(fn_benchmark, model_name)
            if best_l2 is None:
                logger.warning(f"No benchmark config for {model_name}, skipping")
                continue

        # Parse original per-seed results for comparison
        if base_model == "HGS":
            from review_experiments.per_split_feature_selection_experiment import \
                parse_original_seed_results as parse_hgs_seeds
            target_hp = {'n_hid': cfg['n_hid'], 'lr': cfg['lr'],
                         'l2': cfg['l2'], 'glr': cfg['glr'], 'AGG': cfg['AGG']}
            original_results = parse_hgs_seeds(fn_benchmark, target_hp)
        else:
            original_results = parse_baseline_original_results(
                fn_benchmark, best_l2, best_lr, best_nl
            )
        logger.info(f"Parsed {len(original_results)} original seed results")

        # ---------------------------------------------------------------
        # Data loading for this model
        # ---------------------------------------------------------------
        if omics == "PRO":
            # PRO: always from data_all
            pass  # data_all already loaded
        else:
            # RNA: build patient DataFrame for this model
            survival_df_sub = survival_df.set_index("PatientID")
            data_df = feature_matrix.T.copy()
            common_idx = data_df.index.intersection(survival_df_sub.index)
            data_df = data_df.loc[common_idx]
            surv_sub = survival_df_sub.loc[common_idx]
            data_df["time"] = surv_sub["OS_60"].values
            data_df["event"] = surv_sub["OS Status"].values
            data_all = data_df

        # ---------------------------------------------------------------
        # Per-seed loop
        # ---------------------------------------------------------------
        results = []
        seeds_to_run = [0] if smoke_test else range(num_seeds)

        for seed in seeds_to_run:
            logger.info(f"\n  Seed {seed}/{max(seeds_to_run)}")

            # --- Step A: Split ---
            if omics == "PRO":
                label = data_all.iloc[:, -1]  # "death"
            else:
                label = data_all["event"]

            data_train_val, data_test, y_train_val, y_test = train_test_split(
                data_all, label, test_size=0.2, random_state=seed,
                shuffle=True, stratify=label,
            )
            data_train, data_valid, _, _ = train_test_split(
                data_train_val, y_train_val, test_size=0.25,
                random_state=seed, shuffle=True, stratify=y_train_val,
            )
            logger.info(f"    Train: {data_train.shape[0]}, Valid: {data_valid.shape[0]}, "
                        f"Test: {data_test.shape[0]}")

            # --- Step B: Per-fold Cox (shared, cached by seed_label) ---
            seed_label = f"{omics}_{cohort}_seed{seed}"
            needs_cache_save = False

            if base_model != "Pnet":
                selected_genes = load_selected_genes(seed_label)

                if selected_genes is not None:
                    logger.info(f"    Loaded {len(selected_genes)} cached genes")
                    top_400 = selected_genes
                    tr_f, va_f, te_f = data_train, data_valid, data_test
                else:
                    tr_f, va_f, te_f, top_400, _ = run_per_split_cox(
                        data_train, data_valid, data_test, cox_processes,
                        max_cox_genes=max_cox_genes,
                        time_col="OS time" if omics == "PRO" else "time",
                        event_col="death" if omics == "PRO" else "event",
                        is_rna=(omics == "RNA"),
                    )
                    if max_cox_genes == 0:
                        needs_cache_save = True
                    logger.info(f"    Selected top 400 genes from training-set Cox")
            else:
                tr_f, va_f, te_f = data_train, data_valid, data_test
                top_400 = data_train.columns[:-2].tolist()

            if needs_cache_save:
                save_selected_genes(seed_label, top_400)
                logger.info(f"    Saved {len(top_400)} selected genes to cache")

            # --- Step C: Build H (only for HGS) ---
            if base_model == "HGS":
                if knowledge == "STRING":
                    # Build STRING H from selected genes
                    pk_hgs = {
                        'method': cfg['knowledge_method'],
                        'type_know': 'STRING',
                        'divisor': cfg['divisor'],
                        'layer_STRING': cfg['layer_STRING'],
                    }
                    # Use existing build_STRING_H function
                    H, valid_genes = build_STRING_H_from_genes(
                        top_400, cohort, pk_hgs
                    )
                    logger.info(f"    STRING H: {H.shape}")

                    # Filter data splits to STRING-available genes
                    n_dropped = len(top_400) - len(valid_genes)
                    if n_dropped > 0:
                        extra = ["OS time", "death"] if omics == "PRO" else ["time", "event"]
                        tr_f = tr_f[valid_genes + extra]
                        va_f = va_f[valid_genes + extra]
                        te_f = te_f[valid_genes + extra]
                        logger.info(f"    Filtered to {len(valid_genes)} STRING genes")

                    G = generate_G_from_H(H.T) if cfg["edge_pooling"] else generate_G_from_H(H)

                elif knowledge == "Reactome":
                    # Load Reactome H1 matrix (lazy)
                    layer_r = cfg.get('layer_Reactome', 8)
                    fn_h1 = REACTOME_H1_TPL.format(layer=layer_r)
                    if not os.path.isfile(fn_h1):
                        logger.error(f"Reactome H1 not found: {fn_h1}, skipping seed")
                        continue
                    H_reactome = pd.read_csv(fn_h1, index_col=0)

                    # Intersect with selected genes
                    common = [g for g in top_400 if g in H_reactome.index]
                    H = build_Reactome_H_from_genes(common, H_reactome)
                    logger.info(f"    Reactome H: {H.shape} (from {len(common)} common genes)")
                    if H.shape[1] == 0:
                        logger.error("Empty Reactome H, skipping seed")
                        continue
                    G = generate_G_from_H(H.T) if cfg["edge_pooling"] else generate_G_from_H(H)

                elif knowledge == "hcluster":
                    H = build_hcluster_H(top_400, cohort, omics)
                    if H is None or H.shape[0] == 0 or H.shape[1] == 0:
                        logger.error(f"Empty hcluster H for {omics} {cohort}, skipping seed")
                        continue
                    logger.info(f"    hcluster H: {H.shape}")
                    G = generate_G_from_H(H.T) if cfg["edge_pooling"] else generate_G_from_H(H)

                else:
                    logger.error(f"Unknown HGS knowledge: {knowledge}")
                    continue

                # Build graph G
                t_obs = tr_f.iloc[:, -2].max() + 2
                fn_ckpt = os.path.join(OUT_DIR, f"{model_label}_{seed}")

                per_fold_ci = train_hgs(
                    cfg, tr_f, va_f, te_f, H, G, t_obs,
                    device, fn_ckpt, seed, logger,
                )

            else:
                # Baseline model training
                if base_model == "DeepSurv":
                    batch_size = 64
                elif base_model == "DeepHit":
                    batch_size = 64
                elif base_model == "DRSA":
                    batch_size = 64
                elif base_model == "Pnet":
                    batch_size = 64

                t_obs_baseline = tr_f.iloc[:, -2].max() + 2
                fn_ckpt = os.path.join(OUT_DIR, f"{model_label}_{seed}")

                # For Pnet, build pathway mask
                pathway_mask = None
                if base_model == "Pnet":
                    try:
                        from utils.data_utils import get_BINN_Pathways
                        # Build pathway mask from training data
                        # Pnet uses its own pathway structure defined in PriorKnow/PNET/
                        if omics == "PRO":
                            data_tmp = tr_f.copy()
                            # Pnet requires pathway file at data/PriorKnow/PNET/
                            pathway_mask, _ = get_BINN_Pathways(
                                data_tmp, 4,
                            )
                            pathway_mask = pathway_mask[:best_nl]
                        else:
                            data_tmp = tr_f.copy()
                            pathway_mask, _ = get_BINN_Pathways(
                                data_tmp, 4,
                            )
                            pathway_mask = pathway_mask[:best_nl]
                    except Exception as e:
                        logger.error(f"Failed to build PNET pathways: {e}, skipping")
                        continue

                per_fold_ci = train_baseline(
                    base_model, tr_f, va_f, te_f,
                    num_layers=best_nl, lr=best_lr, l2=best_l2,
                    batch_size=batch_size,
                    device=device, fn_ckpt=fn_ckpt, seed=seed,
                    t_obs=t_obs_baseline, pathway_mask=pathway_mask,
                    logger=logger,
                )

            # --- Step D: Compare with original ---
            original_ci = original_results.get(seed, None)
            delta = (per_fold_ci - original_ci) if original_ci else None

            logger.info(f"    Original C-index:  {original_ci:.4f}" if original_ci
                        else "    Original: N/A")
            logger.info(f"    Per-fold C-index:  {per_fold_ci:.4f}")
            if delta is not None:
                logger.info(f"    Delta:             {delta:+.4f}")

            results.append({
                'Omics': omics,
                'Cohort': cohort,
                'Model': model_label,
                'Knowledge': knowledge if knowledge else '-',
                'Seed': seed,
                'Original_CIndex': original_ci,
                'PerFold_CIndex': per_fold_ci,
                'Delta': delta,
            })

        # Save per-model results
        if results:
            df_model = pd.DataFrame(results)
            fn_out = os.path.join(OUT_DIR, f"{omics}_{cohort}_{model_label}_results.csv")
            df_model.to_csv(fn_out, index=False)
            logger.info(f"Results saved to {fn_out}")
            all_results.append(df_model)
        else:
            logger.warning(f"No results for {model_label}")

    return all_results


# ============================================================
# Section 6: Summary & CLI
# ============================================================

def compute_summary(results_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Compute per (Omics, Cohort, Model, Knowledge) summary."""
    if not results_list:
        return pd.DataFrame()
    df = pd.concat(results_list, ignore_index=True)

    rows = []
    for (omics, cohort, model, knowledge), group in df.groupby(
            ['Omics', 'Cohort', 'Model', 'Knowledge']):
        paired = group.dropna(subset=['Original_CIndex', 'PerFold_CIndex'])
        deltas = paired['PerFold_CIndex'].values - paired['Original_CIndex'].values
        orig = group['Original_CIndex'].dropna()
        pf = group['PerFold_CIndex'].dropna()

        rows.append({
            'Omics': omics,
            'Cohort': cohort,
            'Model': model,
            'Knowledge': knowledge,
            'N_Seeds': len(group),
            'Original_Mean±Std': f"{orig.mean():.4f}±{orig.std():.4f}" if len(orig) > 0 else "N/A",
            'PerFold_Mean±Std': f"{pf.mean():.4f}±{pf.std():.4f}" if len(pf) > 0 else "N/A",
            'Δ_Mean±Std': f"{deltas.mean():.4f}±{deltas.std():.4f}" if len(deltas) > 0 else "N/A",
            'Δ_Min': f"{deltas.min():.4f}" if len(deltas) > 0 else "N/A",
            'Δ_Max': f"{deltas.max():.4f}" if len(deltas) > 0 else "N/A",
        })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="All-models per-fold feature selection validation (HCC, R2 #12)"
    )
    parser.add_argument(
        "--omics", type=str, choices=["PRO", "RNA", "both"], default="both",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help=f"Models to run (default: all). Options: {ALL_MODELS}",
    )
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cox_processes", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_cox_genes", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip (Omics, Model) combos with existing results")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("per_split_fs_all_models", log_dir="logs")
    logger.info(f"Output: {OUT_DIR}")
    logger.info(f"Arguments: {args}")

    target_omics = []
    if args.omics in ("PRO", "both"):
        target_omics.append("PRO")
    if args.omics in ("RNA", "both"):
        target_omics.append("RNA")

    target_models = args.models or ALL_MODELS
    # Validate model names
    for m in target_models:
        if m not in ALL_MODELS:
            logger.error(f"Unknown model: {m}. Valid: {ALL_MODELS}")
            return

    all_results = []
    for omics in target_omics:
        cohort = "HCC" if omics == "PRO" else "LIHC"

        models_to_run = []
        for model_name in target_models:
            if args.skip_existing:
                fn_check = os.path.join(
                    OUT_DIR, f"{omics}_{cohort}_{model_name}_results.csv"
                )
                if os.path.isfile(fn_check):
                    logger.info(f"[SKIP] {omics} {model_name} — results exist")
                    df_existing = pd.read_csv(fn_check)
                    if not df_existing.empty:
                        all_results.append(df_existing)
                    continue
            models_to_run.append(model_name)

        if not models_to_run:
            logger.info(f"[SKIP] {omics}: all models have existing results")
            continue

        # Run experiment for this omics
        df_list = run_hcc_experiment(
            omics=omics, models=models_to_run,
            device=args.device, cox_processes=args.cox_processes,
            num_seeds=args.num_seeds, epochs=args.epochs,
            smoke_test=args.smoke_test, max_cox_genes=args.max_cox_genes,
            logger=logger,
        )
        all_results.extend(df_list)

    # Save combined summary
    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        fn_master = os.path.join(OUT_DIR, "hcc_all_models_results.csv")
        master_df.to_csv(fn_master, index=False)

        summary_df = compute_summary(all_results)
        fn_summary = os.path.join(OUT_DIR, "hcc_all_models_summary.csv")
        summary_df.to_csv(fn_summary, index=False)

        logger.info(f"\n{'=' * 60}")
        logger.info("MASTER SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        logger.info(f"\nMaster results: {fn_master}")
        logger.info(f"Summary: {fn_summary}")

    logger.info("\nExperiment completed.")


if __name__ == "__main__":
    main()
