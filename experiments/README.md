# Supplementary experiments

These scripts provide the supplementary validation and analysis workflows for HGS-BioAbductor. Run them from the repository root with the benchmark environment active. Use `--help` for options. Dataset files, benchmark results, and trained checkpoints must be supplied separately.

| Script | Purpose |
| --- | --- |
| `per_split_feature_selection_experiment.py` | Training-only feature selection validation for HCC PRO and LIHC RNA |
| `per_split_fs_all_cohorts.py` | Extend that validation across 18 cohorts and multiple models |
| `per_split_fs_all_models.py` | Compare HGS variants and neural baselines for HCC |
| `per_split_fs_analysis.py` | Compile feature-selection results and figures |
| `statistical_test_18_datasets.py` | Paired comparisons from benchmark seed results |
| `hcc_batch_effect_pca.py` | PCA before/after normalization for Gao, Jiang, and Xing cohorts |
| `preprocessing_sensitivity.py` | Cox threshold sensitivity and normalization analysis |
| `extract_disturb_H_checkpoints.py` | Extract perturbation metrics from existing checkpoints |
| `perturbation_comparison_plot.py` | Compare perturbation curves |

## Feature selection protocol

The legacy training/preprocessing workflow uses cohort-wide Cox-ranked inputs. The per-split experiments instead split patients first, fit feature selection on the training subset, and reuse that selection for validation and test patients. They reuse benchmark hyperparameters to isolate the feature-selection change; they are not a nested hyperparameter-selection evaluation. Use these scripts when assessing the effect of training-only feature selection, and distinguish their outputs from the legacy benchmark.

```bash
python experiments/per_split_feature_selection_experiment.py --smoke_test
python experiments/per_split_fs_all_cohorts.py --omics PRO --cohort HCC --models HGS-STRING DeepSurv
python experiments/per_split_fs_all_cohorts.py --skip_existing
python experiments/per_split_fs_analysis.py
python experiments/statistical_test_18_datasets.py --help
```

`--smoke_test` in the feature-selection scripts still requires input data and trains a reduced experiment. It is not an installation-only check. Feature-selection scripts cache selected genes and reuse benchmark result files; use fresh output/cache directories when changing data or preprocessing. Check the individual script's help before rerunning an experiment.

## Custom paths

```bash
python experiments/hcc_batch_effect_pca.py --data-root data/HCC_MultiCohorts/ProteinCohorts_0.8nafilter_CoxSort/ALL --smoke_test
python experiments/preprocessing_sensitivity.py --dataset data/PRO/HCC/dataset.csv --clinic-data data/HCC_MultiCohorts/ProteomicsCohorts_clinic.csv
python experiments/extract_disturb_H_checkpoints.py --checkpoint-dir Results/Ablation/PRO/HCC/STRING/10seedsModelsWeights --smoke_test
python DataPreprocess/FS_COX.py --help
python DataPreprocess/PDC_Preprocess.py --help
```

The preprocessing entry points publish the original cohort-wide Cox-ranking workflow with configurable paths. They are for legacy reproduction; do not treat their rankings as training-only feature selection. The reusable Cox helper is in `DataPreprocess/Preprocess/DATA_preprocess.py`.

## Legacy ablation template

`legacy/HGS_ablation_study.py` preserves the original ablation training template. It references an `HGS_ablation` model implementation that is absent from the available source and also contains experiment-specific settings. It supports `--help` but refuses to start training without that implementation. Checkpoint extraction and perturbation plotting can still analyze existing outputs. No replacement architecture is inferred here.

The `models/HGS.py`, `models/_Models_interpret.py`, and `models/layers_interpret.py` modules are retained for potential external checkpoint/import compatibility. Current training and XAI entry points use `models/Models.py`. New experiments should use that active implementation.
