# HGS-BioAbductor 工作规范

## 修改原则

### 审稿驱动
- 所有修改必须直接对应审稿意见（R1 Q1-Q4 / R2 #1-#22）
- 尽量不改动其他地方，除非修改本身需要连带调整
- 如果有不确定的修改，先问 Cyp，确认后再执行

### 审稿回复信路径
- 回复信文件：`docs/HGS_latex/review/npj_precision_oncology_review.md`

### 原子化本地提交
- 每个逻辑完整的改动作为一个独立 commit
- 不要混装不相关的改动到同一个 commit
- commit message 格式示例：
  - `feat: Add paired t-test for benchmark comparison (R1 Q3)`
  - `fix: Correct feature selection to per-split training (R2 #12)`
  - `docs: Add response letter draft`

### 推送需确认
- `git push` 必须先列出待推送的 commit 内容
- 获得 Cyp 明确同意后方可推送
- 严禁 `git push --force`

### LaTeX 语法规范（仅适用于 docs/HGS_latex/）
- Em dash 用 `---`，en dash 用 `--`
- 引号用反引号+单引号
- 数学符号用 `$...$` 或 `\(...\)`
- 禁止使用 Unicode 字符

## 工作流

1. 明确当前要处理的 TODO
2. 检查是否已和 Cyp 确认细节
3. 执行修改/实验
4. `git add -A && git commit -m "描述"`
5. 如果要推送：列出 commits → 等确认 → 推送

---

## 实验进度

### R2 #12 — Per-split 特征选择验证实验

**状态**: 单数据集实验（HCC PRO STRING + LIHC RNA Reactome）已完成，全量结果已生成

**脚本** (均在 `review_experiments/` 下):
- `per_split_feature_selection_experiment.py` — 单数据集版
- `per_split_fs_all_cohorts.py` — 全 18 cohorts 版（支持 `--models` 参数，覆盖 HGS×3知识 + DeepSurv/DeepHit/DRSA/Pnet）
- `per_split_fs_analysis.py` — 结果汇总 + 图表生成
- `per_split_fs_all_models.py` — HCC 全模型对比版（HGS×3知识 + DeepSurv + DeepHit + DRSA + Pnet）

**输出目录**: `Results/per_split_fs/`（单数据集结果）/ `Results/per_split_fs/all_cohorts/`（全量结果）

**已知问题**:
- LIHC RNA 8559 基因下 Cox 回归极慢（每 seed ~48min），全量运行注意耗时
- STRING H 只有 ~308/400 PRO 基因有对应，数据分片需同步过滤

**全量实验结果**:
| Dataset | Knowledge | Original C-index | Per-split C-index | Δ | p-value |
|---------|-----------|-----------------:|------------------:|-------:|--------:|
| HCC PRO | STRING | 0.7130±0.0525 | 0.7190±0.0586 | +0.0060±0.0171 | 0.3235 |
| LIHC RNA | Reactome | 0.7590±0.0554 | 0.7081±0.0566 | **-0.0509±0.0279** | **0.0004** |

- **HCC PRO STRING**: Δ = +0.0060 (ns, p=0.32) — per-split 特征选择不显著影响结果
- **LIHC RNA Reactome**: Δ = -0.0509 (p<0.001) — 数据泄漏修正后性能下降约5%，可能原因是 RNA 上 Reactome 通路的冗余结构被破坏

**设计特点**:
- 自动从 benchmark `Results-nt20.csv` 解析每 cohort 最优超参（复用 AutoML 结果）
- 超参不变是为了隔离变量：控制"特征选择方式"为唯一变化量，让 Δ 可归因
- 缓存：每 seed 的 Cox 选出的基因保存到 `selected_genes/`，中断后跳过已完成 seed
- checkpoint：每 seed 训练结束自动保存 `.ckpt`，可用作恢复点

**全量运行**:
```bash
# 所有 cohorts
python review_experiments/per_fold_fs_all_cohorts.py

# 仅 PRO 或 RNA
python review_experiments/per_fold_fs_all_cohorts.py --omics PRO
python review_experiments/per_fold_fs_all_cohorts.py --omics RNA

# 指定单个 cohort
python review_experiments/per_fold_fs_all_cohorts.py --omics PRO --cohort HCC

# 指定模型
python review_experiments/per_split_fs_all_cohorts.py --models HGS-STRING HGS-Reactome DeepSurv

# 冒烟测试
python review_experiments/per_split_fs_all_cohorts.py --smoke_test

# 跳过已完成的（断点续跑）
python review_experiments/per_split_fs_all_cohorts.py --skip_existing

# 分析已生成的结果
python review_experiments/per_split_fs_analysis.py

# HCC 全模型对比（HGS×3知识 + DeepSurv + DeepHit + DRSA + Pnet）
python review_experiments/per_split_fs_all_models.py
```

**下一步**:
1. 运行 HCC 全模型实验（`per_split_fs_all_models.py`）检查各模型的 per-split 趋势
2. 运行全 cohorts 多模型实验（`per_split_fs_all_cohorts.py --models ...`）
3. 全部完成后运行 `per_split_fs_analysis.py` 生成汇总表和图表
