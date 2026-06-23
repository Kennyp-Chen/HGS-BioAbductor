# Plan: Per-Fold Feature Selection Validation Experiment

> **Goal**: Demonstrate that full-data Cox feature selection does not meaningfully inflate benchmark performance, by comparing with per-fold (training-set-only) feature selection on HCC PRO and LIHC RNA.

> **Status**: Ready for review. Say `$start-work` to execute.

---

## Background & Key Insight

**R2 #12**: Cox univariate p-value ranking + top 400 is done on full data before train/test split.

**Why we can NOT directly reuse `HGS_GS.py`**:
- `_load_data()` loads only the top 400 pre-filtered genes
- The Cox selection step must happen **after** `data_split()`, which `HGS_GS.py` doesn't support
- We need a loop that: load ALL genes → split → Cox on train only → select 400 → build H → train

**Solution**: Create a standalone experiment script (`DataPreprocess/per_fold_feature_selection_experiment.py`) that:
- Imports `models.Models.HGS` for model training (not modifying HGS_GS.py)
- Reuses `utils.data_utils.data_split()` for splitting + H construction
- Reuses `DataPreprocess.Preprocess.DATA_preprocess.cox_feature_selection()` for Cox ranking
- Uses the **best known hyperparameters** from existing results (no grid search overhead)

---

## Data Flow Comparison

### Current (Full-Data Cox Selection)
```
data/                           FS_COX.py/PDC_Preprocess.py
  feature_matrix.csv ─────────→ cox_feature_selection(ALL patients)
  (all genes)                     │
                                 ↓
                              genes_list.csv (sorted by Cox p-value)
                              dataset.csv (columns sorted)
                                 │
                    HGS_GS.py:  _load_data() ─→ take first 400
                                 │
                                 ↓
                              data_split(data) ─→ train/val/test
                                 │
                                 ↓
                              Train HGS
```

### Per-Fold (What We Need)
```
data/
  feature_matrix.csv ─────────→ load ALL genes
  (all genes)
                                 │
                              data_split(data) ─→ train/val/test
                                 │
                  ┌──────────────┤
                  │              │
                  ↓              │
        Cox on TRAIN only        │
        sort by p-value          │
        select top 400 ──────────┤
                  │              │
                  ↓              ↓
           train(400 genes)   test(400 genes)
                  │
               Build H (from training-set genes)
                  │
               Train HGS
```

---

## Implementation

### File to Create
`DataPreprocess/per_fold_feature_selection_experiment.py`

### Script Structure

```python
import sys, os, argparse, yaml, logging, warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split

# Model
from models.Models import HGS
# Data utils
from utils.data_utils import data_split, build_hiddens, load_opt_data
from utils.hg_ops import generate_G_from_H
# Cox feature selection
from Preprocess.DATA_preprocess import cox_feature_selection
# Config
from utils.config_loader import ConfigLoader

warnings.filterwarnings("ignore")

def run_per_fold_experiment(
    omics_data: str,      # "PRO" or "RNA"
    cohort: str,          # "HCC" or "LIHC"
    knowledge: str,       # "STRING" or "Reactome"
    hyper_params: dict,   # Best hyperparameters from existing results
    pk_config: dict,      # Best prior knowledge config
    num_seeds: int = 10,
    device: str = "cuda:0",
    results_dir: str = "Results/per_fold_fs",
):
    """
    Run per-fold feature selection experiment.
    
    Flow per seed:
      1. Load ALL genes from the feature matrix
      2. Split into train/val/test (same as HGS_GS.py)
      3. On training set: compute Cox p-values for all genes → rank → select top 400
      4. Filter train/val/test to those 400 genes
      5. Build hypergraph H from the selected genes
      6. Train HGS model with the best hyperparameters
      7. Record test C-index
    """
    ...
```

### Per-Omics Data Loading

#### HCC PRO (2259 genes, STRING knowledge)

```python
# Load FULL data (all 2259 gene columns, not just top 400)
fn_data = f"data/PRO/HCC/dataset.csv"
data_df = pd.read_csv(fn_data, index_col=0)
# data_df: 412 patients × (2259 genes + "time" + "event")

# Split by seed (same split logic as data_split())
for seed in range(num_seeds):
    label = data_df.iloc[:, -1]
    data_train_val, data_test, y_train_val, y_test = train_test_split(
        data_df, label, test_size=0.2, random_state=seed, stratify=label
    )
    data_train, data_valid, _, _ = train_test_split(
        data_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
    )
    
    # Cox on TRAINING SET ONLY
    feat_train = data_train.iloc[:, :-2]       # patients × 2259 genes
    te_train = data_train.iloc[:, -2:]          # patients × 2 (time, event)
    _, p_values = cox_feature_selection(
        time_col="time", event_col="event",
        feature_matrix=feat_train,
        label_matrix=te_train,
        process_num=60                          # parallel workers
    )
    # Rank by p-value, select top 400
    gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
    top_400 = gene_ranking.index[:400].tolist()
    
    # Filter to top 400 genes
    data_train_400 = data_train[top_400 + ["time", "event"]]
    data_valid_400 = data_valid[top_400 + ["time", "event"]]
    data_test_400 = data_test[top_400 + ["time", "event"]]
    
    # Build STRING H from training-set genes
    # (reuse data_split's STRING logic)
    _, _, _, t_obs, H = data_split(
        seed=seed, dataset=cohort,
        data=data_train_400,           # only 400 genes, correct for STRING H
        HD=pk_config, construct_H=True
    )
    # Note: data_split internally splits again, but we already split.
    # Instead, extract the STRING H construction directly:
    H = build_STRING_H(data_train_400, cohort, pk_config)
    
    # Train HGS
    model = HGS(hyper_params, data_train_400.values, data_valid_400.values,
                data_test_400.values, H, fn_ckpt, t_obs, G, seed)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'],
                           weight_decay=hyper_params['l2'])
    model.fit(optimizer=optimizer, logger=logger,
              num_epochs=hyper_params["epochs"],
              batch_size=hyper_params["batch_size"],
              loss_dict=hyper_params['loss_w'])
    
    # Record
    ckpt = torch.load(f'{fn_ckpt}.ckpt', map_location=device)
    test_ci = ckpt['final_test_ci']
```

#### LIHC RNA (8966 genes, Reactome knowledge)

```python
# Load ALL data (not just top 400)
fn_data = "data/RNA/LIHC"
feature_matrix = pd.read_csv(f"{fn_data}/feature_matrix.csv", index_col=0)
# feature_matrix: 8966 genes × patients

# Load survival
survival_df = pd.read_csv("data/RNA/ClinicalDataFrame_DiscreteTime-Cut15Years.csv", index_col=0)
survival_df = survival_df.loc[survival_df["PatientID"].isin(feature_matrix.columns)]
survival_df = survival_df.loc[~survival_df.duplicated()]

# Load Reactome H for intersection
H_path = f"data/PriorKnow/Reactome/reactome_P{pk_config['layer_Reactome']}"
H_reactome = pd.read_csv(f"{H_path}/H1.csv", index_col=0)

# Intersect with Reactome
common_genes = feature_matrix.index.intersection(H_reactome.index)
feature_matrix = feature_matrix.loc[common_genes]  # Reactome-filtered, ALL genes

for seed in range(num_seeds):
    # Build patient DataFrame
    data_df = feature_matrix.T  # patients × genes
    data_df["time"] = survival_df.loc[
        survival_df["PatientID"].isin(data_df.index), "OS_60"
    ].values
    data_df["event"] = survival_df.loc[
        survival_df["PatientID"].isin(data_df.index), "OS Status"
    ].values
    
    # Split
    label = data_df.iloc[:, -1]
    data_train_val, data_test, _, _ = train_test_split(
        data_df, label, test_size=0.2, random_state=seed, stratify=label
    )
    data_train, data_valid, _, _ = train_test_split(
        data_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
    )
    
    # Cox on TRAINING SET ONLY
    feat_train = data_train.iloc[:, :-2]  # all Reactome genes (≈7305)
    te_train = data_train.iloc[:, -2:]
    _, p_values = cox_feature_selection(
        time_col="time", event_col="event",
        feature_matrix=feat_train, label_matrix=te_train, process_num=60
    )
    gene_ranking = pd.Series(p_values, index=feat_train.columns).sort_values()
    top_400 = gene_ranking.index[:400].tolist()
    
    # Filter and build Reactome H
    H = H_reactome.loc[top_400, :]
    H = H.loc[:, H.sum(axis=0) != 0]
    H = H.values
    
    # Train HGS (same as PRO flow)
```

### STRING H Construction Helper

For PRO-STRING, `data_split()` builds H from `data_train.columns[:-2]`. We extract this logic:

```python
def build_STRING_H(data_train, cohort, pk_config):
    """Build STRING H matrix from training-set gene list."""
    fn_H = f"data/PriorKnow/STRING/sorted/{pk_config['method']}/{cohort}-Level{pk_config['layer_STRING']}-H.csv"
    gene_set = data_train.columns[:-2]
    genes_STRING = pd.read_csv("data/PriorKnow/STRING/clusters.protein.ensg.csv")['protein_id'].to_list()
    genes_STRING = sorted(list(set(genes_STRING)))
    gene_set = gene_set[gene_set.isin(genes_STRING)]
    
    if os.path.isfile(fn_H):
        H = pd.read_csv(fn_H, index_col=0)
    else:
        from utils.hg_ops import construct_H_STRING
        H = construct_H_STRING(gene_set, layer_STRING=pk_config['layer_STRING'])
        os.makedirs(os.path.dirname(fn_H), exist_ok=True)
        H.to_csv(fn_H)
    
    H = H.loc[gene_set, :]
    return H.values
```

---

## Hyperparameters to Use

### HCC PRO (STRING)

From `Results/Benchmark/PRO/HGS-STRING/Auto/Final_Results-nt20.csv` → HCC entry:
```
opt_l2:      0.005
opt_lr:      0.01
opt_glr:     0.1
opt_agg:     'noAGG'
opt_PredHids: [100, 50]
opt_depth:   2
best_val:    0.7009±0.0391
best_test:   0.7130±0.0498
pk_config:   {'type_know': 'STRING', 'layer_STRING': 42, 'divisor': 8,
              'method': 'layer_range', 'div_know_minmax': (5, 10, 24, 104)}
event_rate:  0.3422
```

Base hyperparams from config (`hgs_architecture_PRO.yaml`):
```yaml
base_hyperparams:
  n_hid: 100, batch_size: 16, epochs: 50, dropout: 0.5, edge_pooling: true
  pooling_method: 'linear', num_min: 25, HG_BN: true, type_atten: 'additive'
  loss_w: {'a': 1, 'b': 1, 'c1': 1, 'c2': 0, 'd': 1}
  cox_num: 400, repeat_time: 10
```

### LIHC RNA (Reactome)

From `Results/Benchmark/RNA/HGS-Reactome/Auto/Final_Results-nt20.csv` → LIHC entry:
```
opt_l2:      0.1
opt_lr:      0.01
opt_glr:     0
opt_agg:     'noAGG'
opt_PredHids: [200, 100]
opt_depth:   2
best_val:    0.7449±0.0524
best_test:   0.7590±0.0526
pk_config:   {'type_know': 'Reactome', 'layer_Reactome': 8, 'divisor': 7,
              'method': 'layer_range', 'div_know_minmax': (5, 10, 1, 12)}
H_shape:     (400, 708)
event_rate:  0.3458
```

---

## Output

### `Results/per_fold_fs/comparison.csv`

| Dataset | Knowledge | Seed | FullData_CIndex | PerFold_CIndex | Delta |
|---------|-----------|------|-----------------|----------------|-------|
| HCC PRO | STRING | 0 | 0.6620 | 0.XXXX | +0.XXX |
| HCC PRO | STRING | 1 | 0.7123 | 0.XXXX | -0.XXX |
| ... | ... | ... | ... | ... | ... |
| HCC PRO | STRING | mean±std | 0.7130±0.0498 | 0.XXXX±0.XXXX | ±0.XXX |

FullData_CIndex values come from `Results/Benchmark/PRO/HGS-STRING/Auto/HCC/Results-nt20.csv` (the best hyperparam set rows).

### `Results/per_fold_fs/summary.txt`

```
=== Per-Fold Feature Selection Validation ===
Dataset: HCC PRO STRING
  Full-data:  0.7130 ± 0.0498
  Per-fold:   0.XXXX ± 0.XXXX
  Delta:      ±0.XXXX (paired t-test p=0.XXXX)

Dataset: LIHC RNA Reactome
  Full-data:  0.7590 ± 0.0526
  Per-fold:   0.XXXX ± 0.XXXX
  Delta:      ±0.XXXX (paired t-test p=0.XXXX)

Conclusion: Per-fold feature selection yields comparable results.
```

---

## Files to Create

| File | Action | Content |
|------|--------|---------|
| `DataPreprocess/per_fold_feature_selection_experiment.py` | **CREATE** | Main experiment script (~300 lines) |
| `Results/per_fold_fs/comparison.csv` | **GENERATE** | Per-seed C-index comparison |
| `Results/per_fold_fs/summary.txt` | **GENERATE** | Summary statistics |

**No modifications to existing code** — this is a purely additive experiment.

---

## Dependencies

All existing dependencies (already installed):
- `lifelines` (CoxPHFitter) — for Cox regression
- `scikit-learn` (train_test_split) — for data splitting
- `torch`, `numpy`, `pandas` — model training
- `models.Models.HGS` — model definition
- `utils.data_utils` — data utilities
- `utils.hg_ops` — hypergraph operations

---

## Expected Runtime

| Phase | Genes × Patients | Cox on train | Model training | Total (10 seeds) |
|-------|-----------------|-------------|----------------|-----------------|
| HCC PRO STRING | 2259 × 412 | ~1 min | ~2 min | ~30 min |
| LIHC RNA Reactome | ~7305 × 370 | ~3 min | ~2 min | ~50 min |
| **Total** | | | | **~1.5 hours** |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Different per-fold gene sets → different H shape | H changes per seed, but this is intentional | Record H shape per seed; compare consistency |
| Cox regression on training set (60-240 samples) | Less statistical power | `penalizer=0.0001` handles convergence issues |
| Reactome H matrix requires exact gene match | H path pruning by `H.sum(axis=0) != 0` handles this | Same logic as `load_opt_data()` |
| STRING H file not found for custom gene sets | Falls back to `construct_H_STRING()` | Same fallback as `data_split()` |
| GPU OOM for larger H (different per fold) | Unlikely with batch_size=16 | Can adjust batch_size if needed |

---

## Why NOT Modify HGS_GS.py

The experiment script approach is preferred because:
1. **Zero risk** to production code
2. **Clean isolation** — experiment logic doesn't pollute the main pipeline
3. **Reproducible** — the exact experiment steps are documented in code
4. **Faster** — only runs the best hyperparameter set (no grid search)
5. **Comparable** — directly compares with existing benchmark results

To truly fix R2 #12 in the pipeline, one would need to add a `--per_fold_fs` flag to HGS_GS.py. But that's a separate change for the paper revision. This experiment validates whether the fix is even necessary.
