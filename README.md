# HGS-BioAbductor

A comprehensive bioinformatics framework for Hypergraph-Based Explainable Survival Modelling and LLM-Based Biological Hypothesis Abduction.

## Project Description

This repository accompanies the paper "HGS-BioAbductor: Hypergraph-Based Explainable Survival Modelling and LLM-Based Biological Hypothesis Abduction". The framework provides an end-to-end pipeline for:

1. **HGS Model Training**: Hypergraph-based survival prediction model
2. **XAI Explanation**: Explainable AI for model interpretation
3. **Biological Hypothesis Abduction**: LLM-based hypothesis generation from XAI explanations
4. **Hypothesis Scoring**: Validation and scoring of generated hypotheses using PubTator3 and PubMed APIs

## Key Features

- **Hypergraph Neural Networks**: Hypergraph neural networks for survival analysis
- **Explainable AI**: Model interpretation using borda score integrated with multiple XAI methods 
- **Biological Knowledge Integration**: Integration with STRING, Reactome, and other biological databases
- **LLM-Based Hypothesis Generation**: Automated biological hypothesis generation using large language models
- **API Validation**: Hypothesis validation using PubTator3 and PubMed.

## Requirements

- Python 3.10+
- PyTorch (CUDA compatible for GPU acceleration)
- CUDA (for GPU acceleration)

## Installation

```bash
# Clone the repository
git clone https://github.com/QIngyuanfl/HGS-BioAbductor.git
cd HGS-BioAbductor

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
HGS-BioAbductor/
├── models/                          # Model implementations
│   ├── HGTS.py                     # Hypergraph Transformer
│   ├── FC_Surv.py                  # Survival prediction models
│   ├── attention.py                 # Attention mechanisms
│   ├── layers.py                    # Neural network layers
│   └── Models_interpret.py         # Interpretable models
├── utils/                           # Utility functions
│   ├── data_utils.py               # Data processing utilities
│   ├── hg_ops.py                   # Hypergraph operations
│   ├── Interpret.py                # XAI interpretation
│   ├── survfunc_utils.py           # Survival analysis utilities
│   ├── attention_validator.py      # Attention validation
│   └── Borda_Score_validator.py    # Borda score validation
├── notebook/                        # Jupyter notebooks
│   ├── 1_explain_hgs.ipynb        # HGS model explanation
│   ├── 2_bio_abduction.ipynb      # Biological hypothesis abduction
│   └── 3_hypothesis_scoring.ipynb # Hypothesis scoring analysis
├── examples/                        # Example data and results
│   ├── AI_hypotheses/              # Generated hypotheses
│   ├── Plausibility/               # Plausibility evaluation results
│   └── repeat*/                    # Repeated experiment results
├── data/                           # Data directory
├── Results/                        # Output results
├── HGS_train.py                    # Main training script
├── HGS_XAI.py                      # XAI explanation script
├── HGS_abduction.py                # Hypothesis abduction script
├── Hypotheses_evaluation.py        # Hypothesis evaluation
├── config.yaml                     # Configuration file
└── requirements.txt                # Dependencies
```

## Usage

### 1. Model Training

```bash
# Train HGS model with default configuration
python HGS_train.py

```

### 2. Model Explanation

```bash
# Generate XAI explanations, you can define your own hyperparameters in this script
python HGS_XAI.py
```

### 3. Hypothesis Abduction

```bash
# Generate biological hypotheses
python HGS_abduction.py
```

### 4. Hypothesis Evaluation

```bash
# Evaluate generated hypotheses
python Hypotheses_evaluation.py
```

### 5. Interactive Analysis

```bash
# Launch Jupyter notebooks for interactive analysis
jupyter notebook notebook/
```

## Configuration

The project uses `config.yaml` for configuration management:

- **Hardware Settings**: CUDA device configuration
- **Model Configuration**: HGS model parameters
- **Data Configuration**: Dataset and knowledge base settings
- **Hyperparameters**: Training hyperparameters
- **Paths**: Data and output directory paths

## Data Requirements

The framework supports multiple data types:
- **Proteomic Data**: Protein expression data
- **RNA Data**: Gene expression data
- **Clinical Data**: Survival and clinical information

Knowledge bases supported:
- **STRING**: Protein-protein interaction network
- **Reactome**: Pathway database
- **Custom**: User-defined knowledge graphs

## Output and Results

The framework generates:
- **Trained Models**: Saved model checkpoints
- **XAI Explanations**: Model interpretation results
- **Biological Hypotheses**: Generated hypotheses with evidence
- **Validation Results**: PubTator3 and PubMed validation scores
- **Statistical Analysis**: Comprehensive statistical testing results

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
- **GitHub Issues**: [Project Issues Page](https://github.com/QIngyuanfl/HGS-BioAbductor/issues)
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