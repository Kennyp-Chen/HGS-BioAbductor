# Installation Guide

This project requires **two separate Python environments**:
1. **Benchmark & XAI Environment**: for model training, benchmarking, and XAI interpretation
2. **Abduction Environment**: for LLM-based hypothesis generation and validation

---

## Environment 1: Benchmark & XAI Environment

### Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n hgs-benchmark python=3.9
conda activate hgs-benchmark

# Or using venv
python3.9 -m venv hgs-benchmark
source hgs-benchmark/bin/activate  # Linux/Mac
# hgs-benchmark\Scripts\activate  # Windows
```

### Install Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision

# Install other dependencies
pip install -r requirements_benchmark.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lifelines; print(f'Lifelines: {lifelines.__version__}')"
```

### Run with This Environment

```bash
# Model training
python HGS_GS.py --omics_data RNA --device cuda:0
python baseline_DL_GS.py --omics_data RNA --device 0
python baseline_SA_GS.py --omics_data RNA
python baseline_SHINE_GS.py --omics_data RNA --device 0

# XAI explanation
python HGS_XAI.py
```

---

## Environment 2: Abduction Environment

### Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n hgs-abduction python=3.10
conda activate hgs-abduction

# Or using venv
python3.10 -m venv hgs-abduction
source hgs-abduction/bin/activate  # Linux/Mac
# hgs-abduction\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -r requirements_abduction.txt
```

### Configure OpenAI API

```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or set in Python script
# import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Verify Installation

```bash
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import langchain_openai; print('LangChain OpenAI installed')"
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')"
```

### Run with This Environment

```bash
# Hypothesis generation
python HGS_abduction.py

# Hypothesis evaluation
python Hypotheses_evaluation.py
```

---

## Quick Install (One-Click)

### Method 1: Using conda

```bash
# Create and install Benchmark environment
conda create -n hgs-benchmark python=3.9 -y
conda activate hgs-benchmark
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_benchmark.txt

# Create and install Abduction environment
conda create -n hgs-abduction python=3.10 -y
conda activate hgs-abduction
pip install -r requirements_abduction.txt
```

### Method 2: Using installation script

Create `install.sh`:

```bash
#!/bin/bash

echo "Installing HGS-BioAbductor environments..."

# Benchmark environment
echo "Creating benchmark environment..."
conda create -n hgs-benchmark python=3.9 -y
conda activate hgs-benchmark
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_benchmark.txt
echo "Benchmark environment installed!"

# Abduction environment
echo "Creating abduction environment..."
conda create -n hgs-abduction python=3.10 -y
conda activate hgs-abduction
pip install -r requirements_abduction.txt
echo "Abduction environment installed!"

echo "Installation complete!"
```

Run:
```bash
chmod +x install.sh
./install.sh
```

---

## Environment Switching

### Using Benchmark Environment

```bash
conda activate hgs-benchmark
# or
source hgs-benchmark/bin/activate
```

### Using Abduction Environment

```bash
conda activate hgs-abduction
# or
source hgs-abduction/bin/activate
```

---

## Dependency Overview

### Benchmark & XAI Environment Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=1.10.0 | Deep learning framework |
| lifelines | 0.27.7 | Survival analysis |
| scikit-survival | >=0.17.0 | Survival analysis |
| captum | >=0.5.0 | XAI interpretation |
| optuna | >=3.0.0 | Hyperparameter optimization |
| gprofiler-official | >=1.0.0 | Pathway enrichment analysis |

### Abduction Environment Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| openai | >=1.0.0 | OpenAI API |
| langchain-openai | >=0.0.5 | LangChain OpenAI integration |
| chromadb | >=0.4.0 | Vector database |
| mcp-server-pubtator3 | >=0.1.0 | PubTator3 API |
| biopython | >=1.79 | Bioinformatics tools |

---

## Troubleshooting

### Q1: PyTorch installation fails

```bash
# Check CUDA version
nvidia-smi

# Select appropriate PyTorch based on CUDA version
# Visit https://pytorch.org/get-started/locally/ for correct installation command
```

### Q2: Some packages fail to install

```bash
# Try upgrading pip
pip install --upgrade pip

# Use domestic mirror (for users in China)
pip install -r requirements_benchmark.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: CUDA not available

```bash
# Check if CUDA is installed correctly
nvidia-smi

# Check if PyTorch recognizes CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch matching your CUDA version
```

### Q4: OpenAI API configuration

```bash
# Method 1: Environment variable
export OPENAI_API_KEY="sk-..."

# Method 2: Set in code
# Edit HGS_abduction.py, add:
# import os
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Method 3: Using .env file
# Create .env file with:
# OPENAI_API_KEY=sk-...
```

---

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.9+ (Benchmark), 3.10+ (Abduction)
- **Memory**: 16GB RAM
- **Storage**: 50GB available space

### Recommended Configuration
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.9 (Benchmark), 3.10 (Abduction)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8 or 12.1)
- **Memory**: 32GB+ RAM
- **Storage**: 100GB+ SSD

---

## Installation Verification

Run the following commands to verify both environments are correctly installed:

```bash
# Test Benchmark environment
conda activate hgs-benchmark
python -c "
import torch
import lifelines
import captum
import optuna
print('✓ Benchmark environment OK')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
"

# Test Abduction environment
conda activate hgs-abduction
python -c "
import openai
import langchain_openai
import chromadb
print('✓ Abduction environment OK')
print(f'  OpenAI: {openai.__version__}')
"
```

If both environments display "OK", the installation is successful!
