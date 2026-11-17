# HGS-BioAbductor

[![Paper DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41698--026--01764--9-blue)](https://doi.org/10.1038/s41698-026-01764-9)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository is the **official implementation** of:

> **HGS-BioAbductor: hypergraph-based explainable survival modelling and LLM-based biological hypothesis abduction**
> Yupeng Chen, Qingyuan Zhang, Pei Jiang, and Linhai Xie
> *npj Precision Oncology* (2026). Published online 14 November 2026.
> **Paper:** [https://doi.org/10.1038/s41698-026-01764-9](https://doi.org/10.1038/s41698-026-01764-9)

HGS-BioAbductor connects survival prediction from clinical omics data with model explanation and biological hypothesis generation. The pipeline includes:

1. **Survival modelling:** train the HGS hypergraph neural network and compare it with DeepSurv, DeepHit, DRSA, P-NET, Cox-PH, RSF, and SHINE.
2. **Model explanation:** combine feature attributions with Borda ranking to identify predictive molecules and pathways.
3. **Hypothesis abduction:** generate and refine biological hypotheses through an LLM-based tournament.
4. **Literature evaluation:** assess hypothesis plausibility and novelty using literature retrieval, PubTator3, and PubMed.

## Installation

Clone the public repository over HTTPS:

```bash
git clone https://github.com/Kennyp-Chen/HGS-BioAbductor.git
cd HGS-BioAbductor
```

The existing [installation guide](INSTALLATION.md) describes separate environments for benchmarking/XAI and hypothesis abduction/evaluation. Install the corresponding dependencies:

```bash
# Benchmarking and XAI
conda create -n hgs-benchmark python=3.10 -y
conda activate hgs-benchmark
# Install a PyTorch build compatible with your GPU and CUDA runtime first.
pip install -r requirements_benchmark.txt

# Hypothesis abduction and evaluation
conda create -n hgs-abduction python=3.10 -y
conda activate hgs-abduction
pip install -r requirements_abduction.txt
```

The neural training and XAI paths use CUDA operations and require a compatible NVIDIA GPU. Dependency compatibility bounds and a [tested Python 3.10 environment snapshot](constraints/python310.txt) are provided. CPU import, CLI, and synthetic-data checks have been run; full GPU training and API-based reproduction require the external resources below.

## Data and configuration

The training data, prior-knowledge files, embedding database, and trained checkpoints are **not bundled** with this repository. Prepare these resources before running the pipeline. Refer to the [paper and its supplementary information](https://doi.org/10.1038/s41698-026-01764-9) for cohort descriptions and data availability.

The study covers eight proteomic cohorts (CCRCC, HaNSCC, LA, LSCC, UCEC, HCC, GBM, PDA) and ten TCGA RNA cohorts (LIHC, STAD, BLCA, OV, LUSC, LGG, LUAD, KIRC, HNSC, BRCA). Biological priors include STRING and Reactome; hierarchical clustering provides a data-driven alternative.

The main paths expected by the scripts include:

```text
data/
├── PRO/<cohort>/dataset.csv
├── RNA/<cohort>/feature_matrix.csv
├── RNA/ClinicalDataFrame_DiscreteTime-Cut15Years.csv
├── PriorKnow/
│   ├── STRING/
│   ├── Reactome/
│   ├── PNET/
│   └── SHINE/
├── XAI/
├── NoveltyScoring/
└── text-embedding-3-small/
```

This is a path overview, not a complete downloadable data package. Some model/knowledge combinations require additional cohort-specific files, including `dataset_Reactome12_1000genes.csv` and Reactome incidence matrices. Check the relevant loader and configuration before running a full experiment.

| Configuration | Purpose |
| --- | --- |
| [configs/hgs_architecture_PRO.yaml](configs/hgs_architecture_PRO.yaml) | Proteomic model parameters, search space, and repetitions |
| [configs/hgs_architecture_RNA.yaml](configs/hgs_architecture_RNA.yaml) | RNA model parameters, search space, and repetitions |
| [configs/xai_hcc.yaml](configs/xai_hcc.yaml) | XAI cohort, GPU, model, and prior-knowledge parameters |
| [configs/config.yaml](configs/config.yaml) | Shared paths, XAI settings, LLM/MCP settings, and evaluation inputs |

The root `config.yaml` is a symbolic link to `configs/config.yaml`. Run scripts from the repository root. Update dataset paths, model settings, and evaluation inputs for your experiment. Use an ignored `configs/config.local.yaml` copy and set `HGS_CONFIG` to its path for local settings and evaluation credentials; see [INSTALLATION.md](INSTALLATION.md). Keep API credentials out of version control.

## Usage

These commands show the current script interfaces. They require the data and configuration described above.

### 1. Train HGS and baseline models

```bash
# HGS: RNA with Reactome prior knowledge
python HGS_GS.py --omics_data RNA --prior_knowledge Reactome --device 0 --n_trials 20

# HGS: proteomics with STRING prior knowledge
python HGS_GS.py --omics_data PRO --prior_knowledge STRING --device 0 --n_trials 20

# DeepSurv, DeepHit, DRSA, and P-NET
python baseline_DL_GS.py --omics_data RNA --device 0 --repeat_time 10

# Cox-PH and random survival forest
python baseline_SA_GS.py --omics_data RNA --repeat_time 10

# SHINE
python baseline_SHINE_GS.py --omics_data RNA --device 0 --repeat_time 10
```

`--device` takes a GPU index such as `0`. For HGS, change `base_hyperparams.repeat_time` in the corresponding architecture YAML to control repeated splits; `HGS_GS.py` does not accept `--repeat_time`. Its `--bayesian_optimization` options are `Random`, `TPE`, `GP`, `CmaEs`, and `Auto`.

### 2. Explain the model

Set the cohort, knowledge type, and model parameters in `configs/xai_hcc.yaml`, and check the shared data paths:

```bash
python HGS_XAI.py --experiment configs/xai_hcc.yaml
```

The bundled XAI configuration describes HCC proteomics with hierarchical clustering. For another experiment, supply a matching YAML with the selected cohort and model parameters.

### 3. Generate biological hypotheses

The abduction script reads `OPENAI_API_KEY` and `BASE_URL` from the environment, or prompts for them if absent. Configure these locally for your API service, then supply a triplet-statistics JSON file as the positional argument:

```bash
python HGS_abduction.py examples/10triplets_statistics.json
```

The current entry point uses `o3-mini` for hypothesis generation, `text-embedding-3-small` for embeddings, and `data/text-embedding-3-small` as the local knowledge database. Override these settings with `--chat-model`, `--embedding-model`, and `--knowledge-path`; use `--output-dir` to change the generation destination. API usage may incur charges.

### 4. Evaluate hypotheses

Configure the local `generic` API settings, PubTator3 MCP server, literature resources, and input/output paths in your ignored local configuration:

```bash
python Hypotheses_evaluation.py
```

Generation and evaluation both default to `Results/AI_Hypothesis/`. Match the evaluation triplet-statistics file to the JSON used for generation. To evaluate the bundled examples, explicitly point the input/output paths in your local configuration to them.

### 5. Inspect example results

The [examples/](examples/) directory contains generated hypotheses, tournament traces, retrieved evidence, and hypothesis scores. These traces include LLM reviewer outputs; they are part of the hypothesis-generation method.

```bash
jupyter notebook notebook/1_experiment_plots.ipynb
jupyter notebook notebook/2_hypothesis_scoring.ipynb
```

Install Jupyter separately if needed. Some notebook cells require experiment outputs that are not included in the repository.

## Supplementary experiments

The [supplementary experiment guide](experiments/README.md) covers per-split feature selection, paired statistical tests, normalization/PCA, and perturbation analysis. Training-only feature-selection validation is provided separately from the legacy cohort-wide Cox-ranked benchmark workflow.

```bash
python experiments/per_split_fs_all_cohorts.py --omics PRO --cohort HCC --models HGS-STRING DeepSurv
```

A legacy ablation template is preserved with its missing model dependency explicitly documented. Older model modules remain available for potential checkpoint compatibility.

## Repository structure

```text
HGS-BioAbductor/
├── HGS_GS.py                    # HGS training and hyperparameter search
├── baseline_DL_GS.py            # Deep learning baselines
├── baseline_SA_GS.py            # Cox-PH and RSF baselines
├── baseline_SHINE_GS.py         # SHINE baseline
├── HGS_XAI.py                   # Model explanation and triplet extraction
├── HGS_abduction.py             # LLM-based hypothesis tournament
├── Hypotheses_evaluation.py     # Literature-based hypothesis scoring
├── models/                     # HGS, baselines, and neural layers
├── utils/                      # Data, hypergraph, survival, and XAI utilities
├── configs/                    # Architecture and shared configurations
├── experiments/                # Supplementary validation and analysis
├── DataPreprocess/             # Legacy preprocessing and reusable Cox helper
├── notebook/                   # Analysis notebooks
├── examples/                   # Example hypotheses, evidence, and scores
├── INSTALLATION.md             # Detailed setup guide
└── LICENSE                     # MIT software license
```

Training and explanation create outputs under `Results/` and `logs/`; output paths vary by script. Large datasets and checkpoints are excluded from version control.

## Citation

If you use HGS-BioAbductor in your research, please cite:

```bibtex
@article{chen2026hgsbioabductor,
  title   = {{HGS-BioAbductor}: hypergraph-based explainable survival modelling and {LLM}-based biological hypothesis abduction},
  author  = {Chen, Yupeng and Zhang, Qingyuan and Jiang, Pei and Xie, Linhai},
  journal = {npj Precision Oncology},
  year    = {2026},
  doi     = {10.1038/s41698-026-01764-9},
  url     = {https://doi.org/10.1038/s41698-026-01764-9}
}
```

## License and contact

The source code is released under the [MIT License](LICENSE). Consult the original sources for the terms governing external data and resources.

For questions, reproducibility issues, and bug reports, please [open a GitHub issue](https://github.com/Kennyp-Chen/HGS-BioAbductor/issues).

We acknowledge the data and tools provided by TCGA, CPTAC, STRING, Reactome, PubMed, PubTator3, and the open-source projects used in this implementation.
