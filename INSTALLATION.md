# Installation and verification

Use Python 3.10 or newer. The validated smoke-test environment uses Python 3.10 and PyTorch 2.5.1 on Linux. Neural training and XAI require an NVIDIA GPU; CPU checks verify imports and data handling only.

## Choose an environment

Run commands from the repository root. Create one environment for the whole pipeline, or separate environments for the two components.

```bash
conda create -n hgs python=3.10 -y
conda activate hgs
python -m pip install --upgrade pip
```

Install a PyTorch build compatible with your GPU and CUDA runtime using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then choose one dependency entry point:

```bash
# Whole pipeline
python -m pip install -r requirements.txt

# Or, benchmarking and XAI only
python -m pip install -r requirements_benchmark.txt

# Or, hypothesis generation and evaluation only
python -m pip install -r requirements_abduction.txt
```

`requirements.txt` includes both component files. `requirements_benchmark.txt` constrains NumPy, SciPy, and setuptools to versions compatible with the retained `lifelines==0.27.7` implementation. Do not remove these bounds without retesting imports and survival metrics. Jupyter is optional and can be installed separately.

For the exact package versions used by the public-release checks, add `-c constraints/python310.txt` to the install command. This is a tested environment snapshot, not a reconstruction of the original paper's training environment. The snapshot's PyTorch version can be installed from the appropriate CPU or CUDA wheel index before installing the other dependencies.

## Configure data and credentials

The root `config.yaml` links to `configs/config.yaml`. Training architecture/search settings are in `configs/hgs_architecture_PRO.yaml` and `configs/hgs_architecture_RNA.yaml`; XAI settings are in `configs/xai_hcc.yaml`.

For local paths and evaluation credentials, make an ignored copy:

```bash
cp configs/config.yaml configs/config.local.yaml
export HGS_CONFIG=configs/config.local.yaml
```

Edit the copy with your data locations. For hypothesis evaluation, fill in `generic.api_key` and `generic.base_url` for your API service and check `default_model`. The evaluation agents and embedding client use this local configuration. Do not commit the copy. For generation, set `OPENAI_API_KEY` and `BASE_URL` in your shell; the script prompts if they are missing.

By default generation and evaluation use `Results/AI_Hypothesis/`. Set `input_triplet.triplets_info_file` to the same JSON used for generation. To inspect the bundled examples instead, point the `plausibility` and `input_triplet` paths in your local copy to `examples/AI_hypotheses/`; evaluation can write to its configured inputs/output directory.

The repository does not include clinical datasets, prior-knowledge tables, literature corpora, an embedding database, or trained checkpoints. See the paper's data availability statement, then prepare the paths described in the main [README](README.md). HCC multi-cohort PCA and legacy preprocessing additionally require cohort-level inputs; their CLI options accept custom locations.

## Verify the installation

These checks do not train a model or call an LLM API:

```bash
python -c "import torch, lifelines, sksurv, captum, optuna; print('Benchmark imports OK'); print('CUDA:', torch.cuda.is_available())"
python HGS_GS.py --help
python HGS_XAI.py --help
python HGS_abduction.py --help
python Hypotheses_evaluation.py --help
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

The regression checks use synthetic cohorts to verify configuration overrides, XAI loading, and alignment of feature columns with prior-knowledge nodes. They do not establish paper-level numerical reproduction.

## Run the pipeline

```bash
python HGS_GS.py --omics_data PRO --prior_knowledge STRING --device 0
python HGS_XAI.py --experiment configs/xai_hcc.yaml
python HGS_abduction.py examples/10triplets_statistics.json
python Hypotheses_evaluation.py
```

Adjust the XAI experiment and generation input to match your training cohort. Generation model/database/output settings are exposed through `--chat-model`, `--embedding-model`, `--knowledge-path`, and `--output-dir`.

See [experiments/README.md](experiments/README.md) for supplementary validation scripts and the limits of the legacy ablation template. Full training and literature evaluation require the external data, GPU resources, and API configuration above.
