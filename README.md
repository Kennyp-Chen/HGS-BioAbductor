# HGS-BioAbductor

A comprehensive bioinformatics framework for Hypergraph-Based Explainable Survival Modelling and LLM-Based Biological Hypothesis Abduction.

## Project Description

This repository accompanies the paper "HGS-BioAbductor: Hypergraph-Based Explainable Survival Modelling and LLM-Based Biological Hypothesis Abduction". The framework provides an end-to-end pipeline for:

1. **Model Training & Benchmarking**: Train HGS and baseline models with grid search optimization
2. **XAI Explanation**: Explainable AI for model interpretation using multiple methods
3. **Biological Hypothesis Abduction**: LLM-based hypothesis generation from XAI explanations
4. **Hypothesis Evaluation**: Validation and scoring of generated hypotheses using PubTator3 and PubMed APIs

## Key Features

- **Hypergraph Neural Networks**: Advanced hypergraph-based architecture for survival analysis
- **Comprehensive Benchmarking**: Compare HGS with baseline models (DeepSurv, DeepHit, DRSA, P-NET, Cox-PH, RSF, SHINE)
- **Automated Hyperparameter Optimization**: Grid search and Bayesian optimization for model tuning
- **Explainable AI**: Model interpretation using Borda score integrated with multiple XAI methods
- **Prior Knowledge Integration**: Support for STRING, Reactome, and hierarchical clustering knowledge bases
- **LLM-Based Hypothesis Generation**: Automated biological hypothesis generation using large language models
- **Hypothesis Validation**: Comprehensive validation using PubTator3 and PubMed APIs

## Requirements

This project requires **two separate Python environments**:

1. **Benchmark & XAI Environment**: Python 3.9+, PyTorch, CUDA (for GPU)
2. **Hypothesis Abduction Environment**: Python 3.10+, OpenAI API

See [INSTALLATION.md](INSTALLATION.md) for detailed requirements.

## Installation

### Quick Start

```bash
# Clone the repository
git clone git@github.com:Kennyp-Chen/HGS-BioAbductor.git
cd HGS-BioAbductor

# Environment 1: Benchmark & XAI (Python 3.9+)
conda create -n hgs-benchmark python=3.9 -y
conda activate hgs-benchmark
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_benchmark.txt

# Environment 2: Hypothesis Abduction (Python 3.10+)
conda create -n hgs-abduction python=3.10 -y
conda activate hgs-abduction
pip install -r requirements_abduction.txt
```

**📖 For detailed installation instructions, troubleshooting, and system requirements, see [INSTALLATION.md](INSTALLATION.md)**

## Project Structure

```
HGS-BioAbductor/
├── models/                          # Model implementations
│   ├── Models.py                   # HGS and baseline models (DeepSurv, DeepHit, DRSA, P-NET, SHINE)
│   ├── attention.py                # Attention mechanisms for hypergraph
│   ├── layers.py                   # Neural network layers
│   └── Models_interpret.py         # Interpretable model wrappers
├── utils/                           # Utility functions
│   ├── data_utils.py               # Data loading and preprocessing
│   ├── hg_ops.py                   # Hypergraph operations
│   ├── Interpret.py                # XAI interpretation methods
│   ├── survfunc_utils.py           # Survival analysis utilities
│   ├── optuna_utils.py             # Bayesian optimization utilities
│   ├── ReactomeNet.py              # Reactome pathway processing
│   └── Borda_Score_validator.py    # Borda score validation
├── configs/                         # Configuration files
│   ├── hgs_architecture_PRO.yaml   # HGS config for proteomic data
│   └── hgs_architecture_RNA.yaml   # HGS config for RNA data
├── notebook/                        # Jupyter notebooks
│   └── 1_experiment_plots.ipynb    # Experiment visualization
├── examples/                        # Example results
│   ├── AI_hypotheses/              # Generated hypotheses
│   ├── Plausibility/               # Plausibility scores
│   └── repeat*/                    # Cross-validation results
├── data/                           # Data directory (not included)
│   ├── PRO/                        # Proteomic datasets
│   ├── RNA/                        # RNA-seq datasets
│   ├── PriorKnow/                  # Prior knowledge databases
│   └── XAI/                        # XAI explanation data
├── Results/                        # Output results
│   ├── GridSearch/                 # Grid search results
│   ├── Benchmark/                  # Benchmark comparisons
│   └── Interpret/                  # XAI explanations
├── logs/                           # Training logs
├── HGS_GS.py                       # HGS grid search training
├── baseline_DL_GS.py               # Deep learning baselines grid search
├── baseline_SA_GS.py               # Statistical baselines grid search
├── baseline_SHINE_GS.py            # SHINE baseline grid search
├── HGS_XAI.py                      # XAI explanation generation
├── HGS_abduction.py                # Hypothesis abduction
├── Hypotheses_evaluation.py        # Hypothesis evaluation
└── requirements.txt                # Python dependencies
```

## Usage

### 1. Model Training

#### Train HGS Model
```bash
# Grid search for HGS model on RNA data
python HGS_GS.py --omics_data RNA --device cuda:0 --repeat_time 10

# Grid search for HGS model on PRO data
python HGS_GS.py --omics_data PRO --device cuda:0 --repeat_time 10
```

#### Train Baseline Models
```bash
# Deep learning baselines (DeepSurv, DeepHit, DRSA, P-NET)
python baseline_DL_GS.py --omics_data RNA --device 0 --repeat_time 10

# Statistical baselines (Cox-PH, RSF)
python baseline_SA_GS.py --omics_data RNA --repeat_time 10

# SHINE baseline
python baseline_SHINE_GS.py --omics_data RNA --device 0 --repeat_time 10
```

### 2. Model Explanation

```bash
# Generate XAI explanations for trained models
python HGS_XAI.py
```

### 3. Hypothesis Abduction

```bash
# Generate biological hypotheses from XAI explanations
python HGS_abduction.py
```

### 4. Hypothesis Evaluation

```bash
# Evaluate and score generated hypotheses
python Hypotheses_evaluation.py
```

### 5. Interactive Analysis

```bash
# Launch Jupyter notebooks for visualization
jupyter notebook notebook/1_experiment_plots.ipynb
```

## Configuration

The project uses YAML configuration files in the `configs/` directory:

### HGS Model Configuration
- `configs/hgs_architecture_PRO.yaml`: Configuration for proteomic data
- `configs/hgs_architecture_RNA.yaml`: Configuration for RNA-seq data

Each configuration file includes:
- **Base Hyperparameters**: Model architecture settings (n_hid, lr, batch_size, etc.)
- **Grid Search Space**: Hyperparameter ranges for optimization
- **Prior Knowledge**: Settings for Reactome, STRING, and hierarchical clustering

### Command Line Arguments
Most scripts support the following arguments:
- `--omics_data`: Data type (RNA or PRO)
- `--device`: CUDA device (e.g., cuda:0, cuda:1)
- `--repeat_time`: Number of cross-validation repeats
- `--prior_knowledge`: Knowledge base type (Reactome, STRING, hcluster)

## Data Requirements

### Supported Data Types
- **Proteomic Data (PRO)**: Protein expression profiles from 8 cancer cohorts
  - CCRCC, HaNSCC, LA, LSCC, UCEC, HCC, GBM, PDA
- **RNA-seq Data (RNA)**: Gene expression profiles from 10 TCGA cancer types
  - LIHC, STAD, BLCA, OV, LUSC, LGG, LUAD, KIRC, HNSC, BRCA
- **Clinical Data**: Survival outcomes and clinical covariates

### Prior Knowledge Bases
- **Reactome**: Biological pathway database with hierarchical structure
- **STRING**: Protein-protein interaction network
- **Hierarchical Clustering**: Data-driven feature grouping
- **P-NET Pathways**: Pathway-based neural network structure

### Data Structure
```
data/
├── PRO/                    # Proteomic datasets
│   └── [cohort]/
│       └── dataset.csv
├── RNA/                    # RNA-seq datasets
│   └── [cohort]/
│       └── feature_matrix.csv
└── PriorKnow/             # Prior knowledge databases
    ├── PNET/              # P-NET pathways
    ├── SHINE/             # SHINE knowledge
    └── Reactome/          # Reactome pathways
```

**Note**: Due to data privacy and size constraints, the `data/` directory is not included in this repository. Please contact the authors for data access.

## Output and Results

### Generated Outputs
- **Model Checkpoints**: Trained model weights (`.ckpt` files)
- **Grid Search Results**: Hyperparameter optimization results (`Results.csv`)
- **Training Logs**: Detailed training logs in `logs/` directory
- **XAI Explanations**: Feature importance and attention weights
- **Biological Hypotheses**: LLM-generated hypotheses with evidence
- **Validation Scores**: PubTator3 and PubMed validation metrics
- **Benchmark Comparisons**: Performance comparison across all models

### Results Directory Structure
```
Results/
├── Benchmark/            # Model comparison results
│   ├── PRO/              # Proteomic data results
│   └── RNA/              # RNA-seq data results
└── Interpret/            # XAI explanation results
```

## Contributing

We welcome contributions to improve this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support:
- **GitHub Issues**: [Project Issues Page](git@github.com:Kennyp-Chen/issues)
- **Paper**: "HGS-BioAbductor: Hypergraph-Based Explainable Survival Modelling and LLM-Based Biological Hypothesis Abduction"

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hgs-bioabductor,
  title={HGS-BioAbductor: Hypergraph-Based Explainable Survival Modelling and LLM-Based Biological Hypothesis Abduction},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025}
}
```

## Acknowledgments

- STRING database for protein-protein interaction data
- Reactome for pathway information
- OpenAI for LLM capabilities
- PubTator3 and PubMed for hypothesis validation 
- TCGA database for cancers RNA-seq data
- CPTAC for cancers Proteomic data
- HCC proteomic clinical data from Gao, Jiang, Xing.