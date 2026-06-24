#!/usr/bin/env python
"""
Per-Fold Feature Selection Validation Experiment (R2 #12)
========================================================

对比实验：per-fold（仅在训练集上）Cox 特征选择 vs. 原始全数据特征选择。

背景（R2 #12 审稿意见）
-----------------------
原始流程中，Cox 回归在全 cohort 上排序基因 p-value → 取 top 400 → 再 split，
导致 test set 的生存信息间接参与了特征选择，造成数据泄漏。

本脚本修复该问题：先 split，再仅在 TRAINING SET 上做 Cox 排序。

数据集信息
-----------
Dataset 1: HCC PRO + STRING knowledge
  - 数据文件: data/PRO/HCC/dataset.csv
  - 样本: 412 位 HCC 患者 (train 246 / valid 83 / test 83, 60/20/20 split)
  - 特征: 2257 个蛋白编码基因的表达值 (ENSG ID)
  - 标签: "OS time" (总生存时间), "death" (事件指示)
  - 先验知识: STRING 蛋白质交互网络 (layer=42, 构建 21 个超边)
  - 最优超参数: n_hid=100, lr=0.01, l2=0.005, glr=0.1, agg='noAGG'

Dataset 2: LIHC RNA + Reactome knowledge
  - 数据文件: data/RNA/LIHC/feature_matrix.csv + ClinicalDataFrame_DiscreteTime-Cut15Years.csv
  - 样本: ~321 位 LIHC 患者 (train ~192 / valid ~64 / test ~65)
  - 特征: 8559 个 Reactome 通路基因的表达值 (ENSG ID, 已与 Reactome H1 交集)
  - 标签: "OS_60" (离散化总生存时间), "OS Status" (事件指示)
  - 先验知识: Reactome 通路数据库 (level=8, 1859 条通路)
  - 最优超参数: n_hid=200, lr=0.01, l2=0.1, glr=0, agg='noAGG'

实验流程（每个数据集独立执行）
-------------------------------
Step 1: 加载全量基因（跳过原始 Cox 预过滤），加载最优超参数
Step 2: 解析原始 benchmark 结果文件，提取每个 seed 的 C-index（用于对比）
Step 3: 对每个 seed (0-9)：

  a. 数据划分
     - 分层抽样 (stratified by event)，60% train / 20% valid / 20% test
     - 保证各 split 中 event 比例与原数据一致

  b. Per-fold Cox 特征选择（只在 TRAINING SET 上）
     - 输入: training set 的基因表达矩阵 + 生存标签
     - 方法: 单变量 CoxPHFitter 回归，每个基因独立计算 p-value
     - 并行: multiprocessing.Pool (默认 60 进程)
     - 输出: 按 p-value 升序排列的基因排名
     - 选择: p-value 最小的前 400 个基因

  c. 同步过滤所有数据分片
     - 将 train/valid/test 的基因列限制为 Step 3b 选出的 top 400
     - PRO STRING: 还需进一步过滤（部分基因不在 STRING H 矩阵中），
       取 400 基因与 STRING H index 的交集（实际约 308 个），
       同步更新数据分片和 H 矩阵

  d. 构建超图 H
     - PRO STRING: 加载/构建 STRING H，过滤到选定基因，删除空通路
     - RNA Reactome: 从预加载的 Reactome H1 中筛选选定基因，删除空通路

  e. 构建图投影 G (hypergraph → graph via generate_G_from_H)
     - edge_pooling=True 时使用 H.T（边 × 节点）
     - edge_pooling=False 时使用 H（节点 × 边）

  f. 计算池化层维度 pooling_hiddens = build_hiddens(H.shape[1], divisor)

  g. 训练 HGS 模型
     - 使用已知最优超参数（不做 grid search）
     - Adam 优化器 + L2 正则化
     - 学习率调度: gamma=0.99, patience=5
     - 评价指标: C-index (concordance index)
     - early stopping: 基于 validation C-index

  h. 记录结果: 从 checkpoint 提取 final_test_ci

Step 4: 汇总与对比
  - 每个 seed 输出: Original C-index vs Per-fold C-index vs Δ
  - 汇总: 每个数据集输出均值 ± 标准差
  - 保存: Results/per_split_fs/comparison.csv (per-seed) + summary.csv (汇总)

数据流概要
-----------
PRO STRING:
  2257 genes
    → Cox on TRAIN (246 patients) → rank → top 400
    → STRING gene filter → ~308 genes in STRING
    → STRING H (308 genes × 21 edges) → generate_G → HGS train → test C-index

RNA Reactome:
  8559 genes (Reactome-intersected)
    → Cox on TRAIN (192 patients) → rank → top 400
    → Reactome H filter (全部 400 基因都在 Reactome 中) → 0 drop
    → Reactome H (400 genes × N pathways) → generate_G → HGS train → test C-index

用法
-----
    # Full experiment (all seeds, both datasets)
    python review_experiments/per_split_feature_selection_experiment.py

    # Smoke test (1 seed with fast epochs)
    python review_experiments/per_split_feature_selection_experiment.py --smoke_test

    # Skip one dataset
    python review_experiments/per_split_feature_selection_experiment.py --skip_rna

    # Custom seeds / epochs
    python review_experiments/per_split_feature_selection_experiment.py --num_seeds 3 --epochs 20

    # 限制 Cox 基因数（RNA 8559 基因可能挂起时用）
    python review_experiments/per_split_feature_selection_experiment.py --max_cox_genes 2000
"""

import os
import sys
import argparse
import logging
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# Import project modules (must be runnable from project root)
# ============================================================
# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Models import HGS
from utils.hg_ops import generate_G_from_H, construct_H_STRING
from utils.data_utils import build_hiddens
from Preprocess.DATA_preprocess import cox_feature_selection


# ============================================================
# Section 1: Best Hyperparameters (from existing benchmark results)
# ============================================================
# These are extracted from:
#   Results/Benchmark/PRO/HGS-STRING/Auto/Final_Results-nt20.csv  (HCC PRO)
#   Results/Benchmark/RNA/HGS-Reactome/Auto/Final_Results-nt20.csv  (LIHC RNA)

def get_best_hparams_pro_string() -> dict:
    """
    Best hyperparameters for HCC PRO STRING.
    
    Source: Results/Benchmark/PRO/HGS-STRING/Auto/HCC/Results-nt20.csv
    Best config (selected by val_ci):
      n_hid=100, lr=0.01, l2=0.005, glr=0.1, agg='noAGG'
      pred_hids=[100,50], depth=2, H_shape depends on layer_STRING
    """
    return {
        # Architecture
        'n_hid': 100,
        'MLP_hiddens': [100],
        'predict_hiddens': [100, 50],
        'pooling_hiddens': [],     # Will be computed from H.shape[1] / divisor
        'depth': 2,
        'AGG': 'noAGG',
        'ACT': 'LeakyReLU',
        'num_hgat': 'single',
        'type_atten': 'additive',
        'edge_pooling': True,
        'pooling_method': 'linear',
        'num_min': 25,
        'HG_BN': True,
        'dropout': 0.5,
        'RESNET': False,
        # Training
        'lr': 0.01,
        'l2': 0.005,
        'glr': 0.1,
        'batch_size': 16,
        'epochs': 50,
        'print_freq': 10,
        'update_freq': 1,
        'gamma': 0.99,
        'patience': 5,
        'score': 'ci',
        'metric_update': 'score',
        'train_scheme': 'final_epoch',
        'loss_w': {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1},
        # Data
        'cox_num': 400,
        'end_point': 'OS_60',
        'repeat_time': 10,
        # Model metadata
        'model': 'HGS',
        'dataset': 'HCC',
    }


def get_best_hparams_rna_reactome() -> dict:
    """
    Best hyperparameters for LIHC RNA Reactome.
    
    Source: Results/Benchmark/RNA/HGS-Reactome/Auto/Final_Results-nt20.csv
    Best config:
      n_hid=200, lr=0.01, l2=0.1, glr=0, agg='noAGG'
      pred_hids=[200,100], depth=2
    """
    return {
        # Architecture
        'n_hid': 200,
        'MLP_hiddens': [200],
        'predict_hiddens': [200, 100],
        'pooling_hiddens': [],
        'depth': 2,
        'AGG': 'noAGG',
        'ACT': 'LeakyReLU',
        'num_hgat': 'single',
        'type_atten': 'additive',
        'edge_pooling': True,
        'pooling_method': 'linear',
        'num_min': 25,
        'HG_BN': True,
        'dropout': 0.5,
        'RESNET': False,
        # Training
        'lr': 0.01,
        'l2': 0.1,
        'glr': 0,
        'batch_size': 32,
        'epochs': 50,
        'print_freq': 10,
        'update_freq': 1,
        'gamma': 0.99,
        'patience': 5,
        'score': 'ci',
        'metric_update': 'score',
        'train_scheme': 'final_epoch',
        'loss_w': {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1},
        # Data
        'cox_num': 400,
        'end_point': 'OS_60',
        'repeat_time': 10,
        # Model metadata
        'model': 'HGS',
        'dataset': 'LIHC',
    }


# ============================================================
# Section 2: Hypergraph Construction Helpers
# ============================================================

def build_STRING_H(data_train: pd.DataFrame, cohort: str,
                   pk_config: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Build STRING hypergraph incidence matrix H from training-set gene list.
    
    This replicates the STRING H construction logic found in
    utils/data_utils.py → data_split().
    
    Args:
        data_train: Training DataFrame (patients × [genes + time + event]).
                    Last 2 columns are time and event.
        cohort: Dataset name (e.g., "HCC") for STRING H file naming.
        pk_config: Dict with 'type_know', 'layer_STRING', 'method'.
    
    Returns:
        Tuple of (H, valid_genes) where:
            H: Incidence matrix as numpy array, shape (n_genes, n_hyperedges).
            valid_genes: List of gene names that survived STRING filtering.
    """
    # Step 1: Extract gene names from training data (exclude time+event cols)
    gene_set = data_train.columns[:-2]
    
    # Step 2: Load STRING gene list (all ~11K protein-coding ENSGs in STRING)
    genes_STRING = pd.read_csv(
        "data/PriorKnow/STRING/clusters.protein.ensg.csv"
    )['protein_id'].to_list()
    genes_STRING = sorted(list(set(genes_STRING)))
    
    # Step 3: Keep only genes that exist in STRING
    gene_set = gene_set[gene_set.isin(genes_STRING)]
    
    # Step 4: Path to cached STRING H file
    fn_H = (
        f"data/PriorKnow/STRING/sorted/{pk_config['method']}/"
        f"{cohort}-Level{pk_config['layer_STRING']}-H.csv"
    )
    
    # Step 5: Load or construct STRING H
    if os.path.isfile(fn_H):
        # Load pre-computed STRING H matrix from file
        H = pd.read_csv(fn_H, index_col=0)
    else:
        # Construct STRING H from scratch using cluster hierarchy
        H = construct_H_STRING(gene_set, layer_STRING=pk_config['layer_STRING'])
        os.makedirs(os.path.dirname(fn_H), exist_ok=True)
        H.to_csv(fn_H)
    
    # Step 6: Sort edge columns and filter rows to our gene set
    edges_sorted = H.columns.sort_values()
    # Some genes from gene_set may not exist in H.index (pruned at this STRING level)
    valid_genes = gene_set.intersection(H.index)
    if len(valid_genes) < len(gene_set):
        n_missing = len(gene_set) - len(valid_genes)
        logger = logging.getLogger("per_split_fs")
        logger.warning(f"{n_missing}/{len(gene_set)} genes not in STRING H, dropping them")
    H = H.loc[valid_genes, edges_sorted]
    H = H.loc[:, H.sum(axis=0) != 0]
    
    return H.values, valid_genes.tolist()


def build_Reactome_H(selected_genes: List[str],
                     H_reactome: pd.DataFrame) -> np.ndarray:
    """
    Build Reactome hypergraph incidence matrix H from selected genes.
    
    Filters a pre-loaded Reactome H1 matrix to only include the selected
    genes, then removes empty pathways.
    
    Args:
        selected_genes: List of ENSG IDs after per-fold Cox selection.
        H_reactome: Full Reactome H1 DataFrame (genes × pathways).
    
    Returns:
        H: Filtered incidence matrix, shape (n_selected_genes, n_active_pathways).
    """
    # Step 1: Filter to selected genes
    H = H_reactome.loc[H_reactome.index.isin(selected_genes), :]
    
    # Step 2: Remove pathways that have no genes in our selection
    H = H.loc[:, H.sum(axis=0) != 0]
    
    return H.values


# ============================================================
# Section 3: Gene Selection Cache (save/load per seed)
# ============================================================
# Cox 回归是最耗时步骤（PRO ~163s/seed, RNA 可能更久）。
# 将每 seed 选出的基因列表保存到文件，方便：
#   - 中断后恢复（跳过 Cox 直接训练）
#   - 跨 seed 分析哪些基因被稳定选中

GENES_CACHE_DIR = "Results/per_split_fs/selected_genes"


def save_selected_genes(seed_label: str, genes: List[str]) -> None:
    """Save per-seed selected gene list to CSV.

    Args:
        seed_label: Unique identifier, e.g. "HCC_PRO_STRING_seed0".
        genes: List of gene ENSG IDs selected by Cox (after STRING filtering).
    """
    os.makedirs(GENES_CACHE_DIR, exist_ok=True)
    fn = os.path.join(GENES_CACHE_DIR, f"{seed_label}.csv")
    pd.DataFrame({"gene": genes}).to_csv(fn, index=False)


def load_selected_genes(seed_label: str) -> Optional[List[str]]:
    """Load per-seed selected gene list if previously saved.

    Args:
        seed_label: Unique identifier, e.g. "HCC_PRO_STRING_seed0".

    Returns:
        List of gene ENSG IDs, or None if no saved file found.
    """
    fn = os.path.join(GENES_CACHE_DIR, f"{seed_label}.csv")
    if not os.path.isfile(fn):
        return None
    df = pd.read_csv(fn)
    return df["gene"].tolist()


# ============================================================
# Section 4: Original Results Parser
# ============================================================

def parse_original_seed_results(fn_results: str,
                                target_hp: dict) -> Dict[int, float]:
    """
    Parse existing benchmark Results-nt20.csv for per-seed C-index values.
    
    The file has multiple grid search blocks separated by '------...'.
    Each block has the format:
        {'n_hid': ...}           ← hyperparameter dict
        {'method': ...}           ← prior knowledge config  
        dataset,seed,L2,lr,...     ← header
        HCC,0,0.005,...            ← seed data rows
    
    WARNING: pred_hids and pool_hids columns contain commas inside brackets
    (e.g. "[100, 50]"), so naive .split(',') will give wrong indices.
    We use a regex to parse seed data robustly.
    
    Args:
        fn_results: Path to Results-nt20.csv from original benchmark.
        target_hp: Dict with keys to match (lr, l2, glr, AGG, n_hid).
    
    Returns:
        Dict: {seed: test_ci} for the matching hyperparameter block.
    """
    # Read the entire results file
    with open(fn_results, 'r') as f:
        content = f.read()
    
    # Split by the 30-dash separator line
    blocks = content.split('-' * 30)
    
    for block_idx, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) < 4:
            continue
        
        # First line is the hyperparameter dict (e.g. "{'n_hid': 100, ...}")
        hp_line = lines[0].strip()
        if not hp_line.startswith("{'n_hid'"):
            continue
        
        # Parse HP dict and check match
        try:
            hp_dict = eval(hp_line)
        except Exception:
            continue
        
        # Match target hyperparameters (all specified keys must match)
        matches = all(
            str(hp_dict.get(k)) == str(target_hp[k])
            for k in ['lr', 'l2', 'glr', 'AGG', 'n_hid']
            if k in target_hp
        )
        if not matches:
            continue
        
        # Found matching block! Parse seed data rows.
        # Seed line format:
        #   Dataset,Seed,L2,lr,glr,AGG,[PredHids...],Depth,[PoolHids...],Val_ci,Test_ci,Val_loss
        #   e.g. "HCC,0,0.005,0.01,0.1,noAGG,[100, 50],2,[25, 3],0.7170,0.6690,76.9753"
        #
        # Since pred_hids and pool_hids contain commas inside brackets,
        # we parse by extracting the last 3 floats (val_ci, test_ci, val_loss)
        # and the 2nd field (seed).
        results = {}
        for line in lines:
            line = line.strip()
            if not line.startswith(('HCC,', 'LIHC,')):
                continue
            
            # Extract seed (2nd field, always before any brackets)
            seed_str = line.split(',')[1]
            
            # Extract test_ci: it's the 2nd-to-last float value.
            # Find all float-like tokens from the end of the line.
            # The line ends with ",val_ci,test_ci,val_loss"
            # pred_hids and pool_hids also contain numbers but they're before.
            # Strategy: the last 3 comma-separated values are always val_ci, test_ci, val_loss.
            # We can extract them by finding the last 3 floats.
            tokens = line.split(',')
            # Walk backwards from the end to find the last 3 numeric values
            numeric_end = []
            for token in reversed(tokens):
                token = token.strip()
                try:
                    numeric_end.append(float(token))
                except ValueError:
                    break
                if len(numeric_end) >= 3:
                    break
            
            if len(numeric_end) >= 3:
                # numeric_end = [val_loss, test_ci, val_ci]
                test_ci = numeric_end[1]
                
                try:
                    seed = int(seed_str)
                    results[seed] = test_ci
                except (ValueError, IndexError):
                    continue
        
        if results:
            return results
    
    # Fallback: no matching block found
    return {}


# ============================================================
# Section 4: Logging Helper
# ============================================================

def setup_logger(name: str = "per_split_fs",
                 log_dir: str = "logs") -> logging.Logger:
    """
    Configure a logger that writes to both file and console.
    
    Args:
        name: Logger name.
        log_dir: Directory for log files.
    
    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"per_split_fs-{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # File handler (records everything)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(fh)
    
    # Console handler (progress updates)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s')
    )
    logger.addHandler(ch)
    
    logger.info(f"Log file: {log_file}")
    return logger


# ============================================================
# Section 5: HCC PRO STRING Experiment
# ============================================================

def run_pro_string_experiment(
    device: str = "cuda:0",
    cox_processes: int = 60,
    num_seeds: int = 10,
    epochs: int = 50,
    smoke_test: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Run per-fold feature selection experiment for HCC PRO STRING.
    
    Flow:
      1. Load all 2259 genes from dataset.csv (no pre-filtering)
      2. For each seed:
         a. Split via train_test_split (60/20/20)
         b. Cox regression on TRAINING SET ONLY → rank → select top 400
         c. Filter train/val/test to those 400 genes
         d. Build STRING H from those 400 genes
         e. Train HGS (best hyperparams) → record test C-index
      3. Compare with original per-seed results
    
    Args:
        device: CUDA device string.
        cox_processes: Parallel workers for Cox univariate regression.
        num_seeds: Number of seeds (0 to num_seeds-1).
        epochs: Training epochs per seed.
        smoke_test: If True, run only seed=0 with fast settings.
        logger: Logger instance.
    
    Returns:
        DataFrame with per-seed comparison results.
    """
    if logger is None:
        logger = setup_logger("pro_string")
    
    # ---------------------------------------------------------------
    # Step 1: Load best hyperparameters and prior knowledge config
    # ---------------------------------------------------------------
    hyper_params = get_best_hparams_pro_string()
    if smoke_test:
        hyper_params['epochs'] = 5  # Fast smoke test
    
    # Prior knowledge config (optimized by Optuna in original run)
    pk_config = {
        'method': 'layer_range',
        'type_know': 'STRING',
        'divisor': 8,
        'layer_STRING': 42,
    }
    
    # ---------------------------------------------------------------
    # Step 2: Load FULL raw data (ALL 2259 genes, not just top 400)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("HCC PRO STRING — Per-fold Feature Selection Experiment")
    logger.info("=" * 60)
    
    fn_data = "data/PRO/HCC/dataset.csv"
    data_df = pd.read_csv(fn_data, index_col=0)
    # data_df shape: (412 patients, 2261 columns) = 2259 genes + "OS time" + "death"
    logger.info(f"Loaded full data: {data_df.shape}")
    logger.info(f"Total genes (before Cox filter): {data_df.shape[1] - 2}")
    logger.info(f"Patients: {data_df.shape[0]}")
    
    # ---------------------------------------------------------------
    # Step 3: Parse original benchmark seed results for comparison
    # ---------------------------------------------------------------
    fn_original = "Results/Benchmark/PRO/HGS-STRING/Auto/HCC/Results-nt20.csv"
    original_results = parse_original_seed_results(fn_original, {
        'n_hid': 100, 'lr': 0.01, 'l2': 0.005, 'glr': 0.1, 'AGG': 'noAGG',
    })
    logger.info(f"Parsed {len(original_results)} original seed results")
    
    # ---------------------------------------------------------------
    # Step 4: Create output directory
    # ---------------------------------------------------------------
    out_dir = "Results/per_split_fs"
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------------------------------------------------------
    # Step 5: Run per-fold experiment for each seed
    # ---------------------------------------------------------------
    results = []
    seeds_to_run = [0] if smoke_test else range(num_seeds)
    
    for seed in seeds_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Seed {seed}/{max(seeds_to_run)}")
        
        # --- Step 5a: Split data into train/val/test ---
        # Same split strategy as utils/data_utils.py → data_split():
        #   1st split: 80% train_val, 20% test
        #   2nd split: 75% of train_val → train, 25% → valid
        # Result ratio: 60% train, 20% valid, 20% test
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
        
        # --- Step 5b: Per-fold Cox feature selection on TRAINING ONLY ---
        seed_label = f"HCC_PRO_STRING_seed{seed}"
        selected_genes = load_selected_genes(seed_label)
        
        if selected_genes is not None:
            logger.info(f"Loaded {len(selected_genes)} cached genes (skipping Cox)")
            top_400 = selected_genes
            has_cox_result = False
        else:
            # Use all genes from training patients to compute Cox p-values
            feat_train = data_train.iloc[:, :-2]   # patients × 2259 genes (expression)
            te_train = data_train.iloc[:, -2:]      # patients × ["OS time", "death"]
            
            logger.info(f"Cox regression on training set: "
                        f"{feat_train.shape[1]} genes × {feat_train.shape[0]} patients")
            start_time = time.time()
            _, p_values = cox_feature_selection(
                time_col="OS time",
                event_col="death",
                feature_matrix=feat_train,
                label_matrix=te_train,
                process_num=cox_processes,
            )
            elapsed = time.time() - start_time
            logger.info(f"Cox completed in {elapsed:.1f}s")
            
            # Sort genes by p-value (ascending → most significant first)
            gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
            top_400 = gene_ranking.index[:400].tolist()
            logger.info(f"Selected top 400 genes (lowest Cox p-values from training set)")
            has_cox_result = True
        
        # --- Step 5c: Filter all splits to the selected top 400 genes ---
        data_train_filt = data_train[top_400 + ["OS time", "death"]]
        data_valid_filt = data_valid[top_400 + ["OS time", "death"]]
        data_test_filt  = data_test[top_400 + ["OS time", "death"]]
        logger.info(f"Filtered data: train {data_train_filt.shape}, "
                    f"valid {data_valid_filt.shape}, test {data_test_filt.shape}")
        
        # --- Step 5d: Build STRING H from training-set-selected genes ---
        H, valid_genes = build_STRING_H(data_train_filt, "HCC", pk_config)
        logger.info(f"Built STRING H: {H.shape}")
        
        if has_cox_result:
            save_selected_genes(seed_label, valid_genes)
        
        n_dropped = len(top_400) - len(valid_genes)
        if n_dropped > 0:
            logger.info(f"Filtering data splits to {len(valid_genes)} STRING-available genes")
            data_train_filt = data_train_filt[valid_genes + ["OS time", "death"]]
            data_valid_filt = data_valid_filt[valid_genes + ["OS time", "death"]]
            data_test_filt  = data_test_filt[valid_genes + ["OS time", "death"]]
        
        # --- Step 5e: Build graph G from H (hypergraph → graph projection) ---
        G = generate_G_from_H(H.T) if hyper_params["edge_pooling"] else generate_G_from_H(H)
        
        # --- Step 5f: Compute t_obs (max observed time + buffer for discrete bins) ---
        t_obs = data_train_filt["OS time"].max() + 2
        logger.info(f"t_obs: {t_obs}")
        
        # --- Step 5g: Compute pooling_hiddens from H's number of hyperedges ---
        # Replicates HGS_GS.py line 291-292:
        #   if pk_config.get('divisor'):
        #       hyper_params['pooling_hiddens'] = build_hiddens(H.shape[1], div)
        hyper_params['pooling_hiddens'] = build_hiddens(
            H.shape[1], pk_config['divisor']
        )
        logger.info(f"pooling_hiddens: {hyper_params['pooling_hiddens']}")
        
        # --- Step 5h: Train HGS model ---
        fn_ckpt = f"{out_dir}/HCC_PRO_STRING_seed{seed}"
        logger.info(f"Training HGS model (epochs={hyper_params['epochs']})...")
        
        model = HGS(
            hyper_params,
            data_train=data_train_filt.values,
            data_eval=data_valid_filt.values,
            data_test=data_test_filt.values,
            H=H,
            fn_ckpt=fn_ckpt,
            t_obs=t_obs,
            G=G,
            seed=seed,
        )
        model = model.cuda()
        
        # Adam optimizer with weight decay (= L2 regularization)
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['l2'],
        )
        
        # Train the model (see models/Models.py → HGS.fit())
        model.fit(
            optimizer=optimizer,
            logger=logger,
            num_epochs=hyper_params["epochs"],
            batch_size=hyper_params["batch_size"],
            loss_dict=hyper_params['loss_w'],
        )
        
        # --- Step 5i: Load checkpoint → extract test C-index ---
        ckpt_path = f'{fn_ckpt}.ckpt'
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            per_fold_ci = ckpt['final_test_ci']
        else:
            logger.error(f"Checkpoint not found: {ckpt_path}")
            per_fold_ci = float('nan')
        
        # Original benchmark result for this seed
        original_ci = original_results.get(seed, None)
        
        logger.info(f"Seed {seed} results:")
        logger.info(f"  Original C-index:  {original_ci:.4f}" if original_ci else "  Original: N/A")
        logger.info(f"  Per-fold C-index:  {per_fold_ci:.4f}")
        if original_ci:
            logger.info(f"  Delta:             {per_fold_ci - original_ci:+.4f}")
        
        results.append({
            'Dataset': 'HCC PRO',
            'Knowledge': 'STRING',
            'Seed': seed,
            'Original_CIndex': original_ci,
            'PerFold_CIndex': per_fold_ci,
            'Delta': (per_fold_ci - original_ci) if original_ci else None,
        })
    
    return pd.DataFrame(results)


# ============================================================
# Section 6: LIHC RNA Reactome Experiment
# ============================================================

def run_rna_reactome_experiment(
    device: str = "cuda:0",
    cox_processes: int = 60,
    num_seeds: int = 10,
    epochs: int = 50,
    smoke_test: bool = False,
    max_cox_genes: int = 0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Run per-fold feature selection experiment for LIHC RNA Reactome.
    
    Flow is similar to PRO but with Reactome knowledge:
      1. Load all Reactome-filtered genes (≈7305) from feature_matrix.csv
      2. Split → Cox on train → pick 400 → filter Reactome H → train HGS
    
    Args:
        device: CUDA device string.
        cox_processes: Parallel workers for Cox regression.
        num_seeds: Number of seeds.
        epochs: Training epochs per seed.
        smoke_test: If True, run only seed=0 with fast settings.
        logger: Logger instance.
    
    Returns:
        DataFrame with per-seed comparison results.
    """
    if logger is None:
        logger = setup_logger("rna_reactome")
    
    # ---------------------------------------------------------------
    # Step 1: Load best hyperparameters and prior knowledge config
    # ---------------------------------------------------------------
    hyper_params = get_best_hparams_rna_reactome()
    if smoke_test:
        hyper_params['epochs'] = 5
    
    pk_config = {
        'method': 'layer_range',
        'type_know': 'Reactome',
        'divisor': 7,
        'layer_Reactome': 8,
    }
    
    # ---------------------------------------------------------------
    # Step 2: Load FULL raw data (ALL Reactome-filtered genes)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("LIHC RNA Reactome — Per-fold Feature Selection Experiment")
    logger.info("=" * 60)
    
    # Load RNA expression matrix (genes × patients, sorted by Cox p-value)
    fn_data = "data/RNA/LIHC/feature_matrix.csv"
    feature_matrix = pd.read_csv(fn_data, index_col=0)
    # feature_matrix shape: (8966 genes, ~370 patients)
    logger.info(f"Loaded feature matrix: {feature_matrix.shape}")
    
    # Load survival data (patients' OS time and status)
    fn_survival = "data/RNA/ClinicalDataFrame_DiscreteTime-Cut15Years.csv"
    survival_df = pd.read_csv(fn_survival, index_col=0)
    # Filter to only patients present in our expression matrix
    survival_df = survival_df.loc[
        survival_df["PatientID"].isin(feature_matrix.columns)
    ]
    survival_df = survival_df.loc[~survival_df.duplicated()]
    logger.info(f"Survival data: {survival_df.shape[0]} patients")
    
    # ---------------------------------------------------------------
    # Step 3: Load Reactome H and intersect genes
    # ---------------------------------------------------------------
    # Load full Reactome H1 matrix (gene × pathway incidence)
    fn_H_path = (
        f"data/PriorKnow/Reactome/reactome_P{pk_config['layer_Reactome']}/H1.csv"
    )
    H_reactome = pd.read_csv(fn_H_path, index_col=0)
    logger.info(f"Full Reactome H1 loaded: {H_reactome.shape}")
    
    # Intersect genes: keep only genes that exist in both Reactome and feature matrix
    # (same as load_opt_data() in utils/data_utils.py lines 370-374)
    common_genes = feature_matrix.index.intersection(H_reactome.index)
    feature_matrix = feature_matrix.loc[common_genes]
    logger.info(f"After Reactome intersection: {feature_matrix.shape[0]} genes")
    
    # ---------------------------------------------------------------
    # Step 4: Parse original benchmark seed results
    # ---------------------------------------------------------------
    fn_original = "Results/Benchmark/RNA/HGS-Reactome/Auto/LIHC/Results-nt20.csv"
    original_results = parse_original_seed_results(fn_original, {
        'n_hid': 200, 'lr': 0.01, 'l2': 0.1, 'glr': 0, 'AGG': 'noAGG',
    })
    logger.info(f"Parsed {len(original_results)} original seed results")
    
    # ---------------------------------------------------------------
    # Step 5: Create output directory
    # ---------------------------------------------------------------
    out_dir = "Results/per_split_fs"
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------------------------------------------------------
    # Step 6: Run per-fold experiment for each seed
    # ---------------------------------------------------------------
    results = []
    seeds_to_run = [0] if smoke_test else range(num_seeds)
    
    for seed in seeds_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Seed {seed}/{max(seeds_to_run)}")
        
        # --- Step 6a: Build patient DataFrame ---
        # Transpose to patients × genes
        data_df = feature_matrix.T.copy()  # patients × n_genes
        
        # Prepare survival subset, aligned with patient IDs
        surv_subset = survival_df.set_index("PatientID")
        surv_subset = surv_subset.loc[surv_subset.index.isin(data_df.index)]
        data_df = data_df.loc[data_df.index.isin(surv_subset.index)]
        
        # Add time and event columns (same column names as HGS expects)
        data_df["time"] = surv_subset.loc[data_df.index, hyper_params["end_point"]].values
        data_df["event"] = surv_subset.loc[data_df.index, "OS Status"].values
        
        logger.info(f"Patient DataFrame: {data_df.shape}")
        
        # --- Step 6b: Split into train/val/test ---
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
        
        # --- Step 6c: Per-fold Cox on TRAINING SET ONLY ---
        seed_label = f"LIHC_RNA_Reactome_seed{seed}"
        selected_genes = load_selected_genes(seed_label)
        
        if selected_genes is not None:
            logger.info(f"Loaded {len(selected_genes)} cached genes (skipping Cox)")
            top_400 = selected_genes
        else:
            feat_train = data_train.iloc[:, :-2]  # all Reactome-filtered genes
            te_train = data_train.iloc[:, -2:]     # time + event
            
            if max_cox_genes > 0 and feat_train.shape[1] > max_cox_genes:
                rng = np.random.default_rng(42)
                sampled = rng.choice(feat_train.columns, max_cox_genes, replace=False)
                feat_train = feat_train[sampled]
                logger.info(f"Limited to {max_cox_genes} random genes for Cox regression")
            
            logger.info(f"Cox regression on training set: "
                        f"{feat_train.shape[1]} genes × {feat_train.shape[0]} patients")
            start_time = time.time()
            _, p_values = cox_feature_selection(
                time_col="time",
                event_col="event",
                feature_matrix=feat_train,
                label_matrix=te_train,
                process_num=cox_processes,
            )
            elapsed = time.time() - start_time
            logger.info(f"Cox completed in {elapsed:.1f}s")
            
            # Sort by p-value, select top 400
            gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
            top_400 = gene_ranking.index[:400].tolist()
            logger.info(f"Selected top 400 genes from training-set Cox ranking")
            
            # Cache the selected genes (skip for sampled subsets since they're not representative)
            if max_cox_genes == 0:
                save_selected_genes(seed_label, top_400)
        
        # --- Step 6d: Filter data splits to top 400 ---
        data_train_filt = data_train[top_400 + ["time", "event"]]
        data_valid_filt = data_valid[top_400 + ["time", "event"]]
        data_test_filt  = data_test[top_400 + ["time", "event"]]
        
        # --- Step 6e: Build Reactome H from selected genes ---
        H = build_Reactome_H(top_400, H_reactome)
        logger.info(f"Built Reactome H: {H.shape}")
        
        # --- Step 6f: Build graph G and compute t_obs ---
        G = generate_G_from_H(H.T) if hyper_params["edge_pooling"] else generate_G_from_H(H)
        t_obs = data_train_filt["time"].max() + 2
        
        # --- Step 6g: Compute pooling_hiddens ---
        hyper_params['pooling_hiddens'] = build_hiddens(
            H.shape[1], pk_config['divisor']
        )
        logger.info(f"pooling_hiddens: {hyper_params['pooling_hiddens']}")
        
        # --- Step 6h: Train HGS model ---
        fn_ckpt = f"{out_dir}/LIHC_RNA_Reactome_seed{seed}"
        logger.info(f"Training HGS model (epochs={hyper_params['epochs']})...")
        
        model = HGS(
            hyper_params,
            data_train=data_train_filt.values,
            data_eval=data_valid_filt.values,
            data_test=data_test_filt.values,
            H=H,
            fn_ckpt=fn_ckpt,
            t_obs=t_obs,
            G=G,
            seed=seed,
        )
        model = model.cuda()
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['l2'],
        )
        
        model.fit(
            optimizer=optimizer,
            logger=logger,
            num_epochs=hyper_params["epochs"],
            batch_size=hyper_params["batch_size"],
            loss_dict=hyper_params['loss_w'],
        )
        
        # --- Step 6i: Extract test C-index from checkpoint ---
        ckpt_path = f'{fn_ckpt}.ckpt'
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            per_fold_ci = ckpt['final_test_ci']
        else:
            logger.error(f"Checkpoint not found: {ckpt_path}")
            per_fold_ci = float('nan')
        
        original_ci = original_results.get(seed, None)
        
        logger.info(f"Seed {seed} results:")
        logger.info(f"  Original C-index:  {original_ci:.4f}" if original_ci else "  Original: N/A")
        logger.info(f"  Per-fold C-index:  {per_fold_ci:.4f}")
        if original_ci:
            logger.info(f"  Delta:             {per_fold_ci - original_ci:+.4f}")
        
        results.append({
            'Dataset': 'LIHC RNA',
            'Knowledge': 'Reactome',
            'Seed': seed,
            'Original_CIndex': original_ci,
            'PerFold_CIndex': per_fold_ci,
            'Delta': (per_fold_ci - original_ci) if original_ci else None,
        })
    
    return pd.DataFrame(results)


# ============================================================
# Section 7: Summary Statistics
# ============================================================

def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics (mean ± std) per dataset.
    
    Args:
        results_df: Per-seed results DataFrame.
    
    Returns:
        Summary DataFrame with mean, std, and delta per dataset.
    """
    summary = []
    
    for (dataset, knowledge), group in results_df.groupby(['Dataset', 'Knowledge']):
        orig = group['Original_CIndex'].dropna()
        per_fold = group['PerFold_CIndex'].dropna()
        
        # Paired differences
        paired = group.dropna(subset=['Original_CIndex', 'PerFold_CIndex'])
        deltas = paired['PerFold_CIndex'].values - paired['Original_CIndex'].values
        
        summary.append({
            'Dataset': dataset,
            'Knowledge': knowledge,
            'N_Seeds': len(group),
            'Original_Mean': orig.mean(),
            'Original_Std': orig.std(),
            'PerFold_Mean': per_fold.mean(),
            'PerFold_Std': per_fold.std(),
            'Delta_Mean': deltas.mean() if len(deltas) > 0 else None,
            'Delta_Std': deltas.std() if len(deltas) > 0 else None,
            'Delta_Min': deltas.min() if len(deltas) > 0 else None,
            'Delta_Max': deltas.max() if len(deltas) > 0 else None,
        })
    
    return pd.DataFrame(summary)


# ============================================================
# Section 8: Command-Line Interface
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Per-fold feature selection validation experiment (R2 #12)"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=10,
        help="Number of random seeds to run (default: 10)",
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
        "--skip_pro", action="store_true",
        help="Skip HCC PRO STRING experiment",
    )
    parser.add_argument(
        "--skip_rna", action="store_true",
        help="Skip LIHC RNA Reactome experiment",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run only seed=0 with 5 epochs for quick verification",
    )
    parser.add_argument(
        "--max_cox_genes", type=int, default=0,
        help="Limit Cox regression to N random genes (0=all, default: 0)",
    )
    parser.add_argument(
        "--only_parse", action="store_true",
        help="Only parse original results (skip training, for testing)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger = setup_logger("per_split_fs")
    logger.info(f"Arguments: {args}")
    
    all_results = []
    
    # Quick test: just parse original results
    if args.only_parse:
        logger.info("\nParsing original HCC PRO STRING results...")
        orig_pro = parse_original_seed_results(
            "Results/Benchmark/PRO/HGS-STRING/Auto/HCC/Results-nt20.csv",
            {'n_hid': 100, 'lr': 0.01, 'l2': 0.005, 'glr': 0.1, 'AGG': 'noAGG'},
        )
        logger.info(f"HCC PRO: {len(orig_pro)} seeds: {orig_pro}")
        
        logger.info("\nParsing original LIHC RNA Reactome results...")
        orig_rna = parse_original_seed_results(
            "Results/Benchmark/RNA/HGS-Reactome/Auto/LIHC/Results-nt20.csv",
            {'n_hid': 200, 'lr': 0.01, 'l2': 0.1, 'glr': 0, 'AGG': 'noAGG'},
        )
        logger.info(f"LIHC RNA: {len(orig_rna)} seeds: {orig_rna}")
        return
    
    # ---- HCC PRO STRING ----
    if not args.skip_pro:
        logger.info("\n\n>>> Starting HCC PRO STRING experiment...")
        df_pro = run_pro_string_experiment(
            device=args.device,
            cox_processes=args.cox_processes,
            num_seeds=args.num_seeds,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
            logger=logger,
        )
        all_results.append(df_pro)
    
    # ---- LIHC RNA Reactome ----
    if not args.skip_rna:
        logger.info("\n\n>>> Starting LIHC RNA Reactome experiment...")
        df_rna = run_rna_reactome_experiment(
            device=args.device,
            cox_processes=args.cox_processes,
            num_seeds=args.num_seeds,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
            max_cox_genes=args.max_cox_genes,
            logger=logger,
        )
        all_results.append(df_rna)
    
    # ---- Save and summarize ----
    if all_results:
        # Combine per-seed results
        results_df = pd.concat(all_results, ignore_index=True)
        out_dir = "Results/per_split_fs"
        os.makedirs(out_dir, exist_ok=True)
        
        # Save per-seed table
        fn_results = os.path.join(out_dir, "comparison.csv")
        results_df.to_csv(fn_results, index=False)
        logger.info(f"\nPer-seed results saved to {fn_results}")
        logger.info(f"\n{results_df.to_string(index=False)}")
        
        # Compute and save summary
        summary_df = compute_summary(results_df)
        fn_summary = os.path.join(out_dir, "summary.csv")
        summary_df.to_csv(fn_summary, index=False)
        
        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        logger.info(f"\nSummary saved to {fn_summary}")
    
    logger.info("\nExperiment completed.")


if __name__ == "__main__":
    main()
