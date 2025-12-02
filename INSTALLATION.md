# 安装指南 / Installation Guide

本项目需要两个独立的 Python 环境：
1. **Benchmark & XAI 环境**：用于模型训练、基准测试和 XAI 解释
2. **Abduction 环境**：用于 LLM 假设生成和验证

---

## 环境 1: Benchmark & XAI 环境

### 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n hgs-benchmark python=3.9
conda activate hgs-benchmark

# 或使用 venv
python3.9 -m venv hgs-benchmark
source hgs-benchmark/bin/activate  # Linux/Mac
# hgs-benchmark\Scripts\activate  # Windows
```

### 安装依赖

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision

# 安装其他依赖
pip install -r requirements_benchmark.txt
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lifelines; print(f'Lifelines: {lifelines.__version__}')"
```

### 使用此环境运行

```bash
# 模型训练
python HGS_GS.py --omics_data RNA --device cuda:0
python baseline_DL_GS.py --omics_data RNA --device 0
python baseline_SA_GS.py --omics_data RNA
python baseline_SHINE_GS.py --omics_data RNA --device 0

# XAI 解释
python HGS_XAI.py
```

---

## 环境 2: Abduction 环境

### 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n hgs-abduction python=3.10
conda activate hgs-abduction

# 或使用 venv
python3.10 -m venv hgs-abduction
source hgs-abduction/bin/activate  # Linux/Mac
# hgs-abduction\Scripts\activate  # Windows
```

### 安装依赖

```bash
pip install -r requirements_abduction.txt
```

### 配置 OpenAI API

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"

# 或在 Python 脚本中设置
# import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 验证安装

```bash
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import langchain_openai; print('LangChain OpenAI installed')"
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')"
```

### 使用此环境运行

```bash
# 假设生成
python HGS_abduction.py

# 假设评估
python Hypotheses_evaluation.py
```

---

## 快速安装（一键安装）

### 方式 1: 使用 conda

```bash
# 创建并安装 Benchmark 环境
conda create -n hgs-benchmark python=3.9 -y
conda activate hgs-benchmark
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_benchmark.txt

# 创建并安装 Abduction 环境
conda create -n hgs-abduction python=3.10 -y
conda activate hgs-abduction
pip install -r requirements_abduction.txt
```

### 方式 2: 使用安装脚本

创建 `install.sh`:

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

运行：
```bash
chmod +x install.sh
./install.sh
```

---

## 环境切换

### 使用 Benchmark 环境

```bash
conda activate hgs-benchmark
# 或
source hgs-benchmark/bin/activate
```

### 使用 Abduction 环境

```bash
conda activate hgs-abduction
# 或
source hgs-abduction/bin/activate
```

---

## 依赖说明

### Benchmark & XAI 环境主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >=1.10.0 | 深度学习框架 |
| lifelines | 0.27.7 | 生存分析 |
| scikit-survival | >=0.17.0 | 生存分析 |
| captum | >=0.5.0 | XAI 解释 |
| optuna | >=3.0.0 | 超参数优化 |
| gprofiler-official | >=1.0.0 | 通路富集分析 |

### Abduction 环境主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| openai | >=1.0.0 | OpenAI API |
| langchain-openai | >=0.0.5 | LangChain OpenAI 集成 |
| chromadb | >=0.4.0 | 向量数据库 |
| mcp-server-pubtator3 | >=0.1.0 | PubTator3 API |
| biopython | >=1.79 | 生物信息学工具 |

---

## 常见问题

### Q1: PyTorch 安装失败

```bash
# 检查 CUDA 版本
nvidia-smi

# 根据 CUDA 版本选择对应的 PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确的安装命令
```

### Q2: 某些包安装失败

```bash
# 尝试升级 pip
pip install --upgrade pip

# 使用国内镜像（中国用户）
pip install -r requirements_benchmark.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: CUDA 不可用

```bash
# 检查 CUDA 是否正确安装
nvidia-smi

# 检查 PyTorch 是否识别 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，重新安装对应 CUDA 版本的 PyTorch
```

### Q4: OpenAI API 配置

```bash
# 方式 1: 环境变量
export OPENAI_API_KEY="sk-..."

# 方式 2: 在代码中设置
# 编辑 HGS_abduction.py，添加：
# import os
# os.environ["OPENAI_API_KEY"] = "sk-..."

# 方式 3: 使用 .env 文件
# 创建 .env 文件，添加：
# OPENAI_API_KEY=sk-...
```

---

## 系统要求

### 最低要求
- **操作系统**: Linux, macOS, Windows 10+
- **Python**: 3.9+ (Benchmark), 3.10+ (Abduction)
- **内存**: 16GB RAM
- **存储**: 50GB 可用空间

### 推荐配置
- **操作系统**: Linux (Ubuntu 20.04+)
- **Python**: 3.9 (Benchmark), 3.10 (Abduction)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8 or 12.1)
- **内存**: 32GB+ RAM
- **存储**: 100GB+ SSD

---

## 验证安装

运行以下命令验证两个环境都正确安装：

```bash
# 测试 Benchmark 环境
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

# 测试 Abduction 环境
conda activate hgs-abduction
python -c "
import openai
import langchain_openai
import chromadb
print('✓ Abduction environment OK')
print(f'  OpenAI: {openai.__version__}')
"
```

如果两个环境都显示 "OK"，说明安装成功！
