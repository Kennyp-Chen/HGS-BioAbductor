# HGS-BioAbductor 工作规范

## 修改原则

### 审稿驱动
- 所有修改必须直接对应审稿意见（R1 Q1-Q4 / R2 #1-#22）
- 尽量不改动其他地方，除非修改本身需要连带调整
- 如果有不确定的修改，先问 Cyp，确认后再执行

### 原子化本地提交
- 每个逻辑完整的改动作为一个独立 commit
- 不要混装不相关的改动到同一个 commit
- commit message 格式示例：
  - `feat: Add paired t-test for benchmark comparison (R1 Q3)`
  - `fix: Correct feature selection to per-fold training (R2 #12)`
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

### R2 #12 — Per-fold 特征选择验证实验

**状态**: 脚本已完成，冒烟测试通过

**脚本**:
- `DataPreprocess/per_fold_feature_selection_experiment.py` — 单数据集版
- `DataPreprocess/per_fold_fs_all_cohorts.py` — 全 18 cohorts 版（自动解析最优超参 + 缓存 + 断点续跑）
- `DataPreprocess/per_fold_fs_analysis.py` — 结果汇总 + 图表生成

**输出目录**: `Results/per_fold_fs/`（单数据集结果）/ `Results/per_fold_fs/all_cohorts/`（全量结果）

**已知问题**:
- LIHC RNA 8559 基因下 Cox 回归极慢（疑似个别基因挂起），冒烟测试用 `--max_cox_genes 200`
- STRING H 只有 ~308/400 PRO 基因有对应，数据分片需同步过滤

**冒烟测试结果**:
| Dataset | Knowledge | Seed | Epochs | Original C-index | Per-fold C-index | Δ |
|---------|-----------|------|--------|----------------:|----------------:|--------:|
| HCC PRO | STRING | 0 | 5 | 0.6715 | 0.7020 | **+0.0304** |
| LIHC RNA | Reactome | 0 | 5 | 0.7330 | 0.5379 | -0.1951* |

\* RNA 冒烟仅用 200 基因，结果不可靠，需全量运行

**设计特点**:
- 自动从 benchmark `Results-nt20.csv` 解析每 cohort 最优超参（复用 AutoML 结果）
- 超参不变是为了隔离变量：控制"特征选择方式"为唯一变化量，让 Δ 可归因
- 缓存：每 seed 的 Cox 选出的基因保存到 `all_cohorts/selected_genes/`，中断后跳过已完成 seed
- checkpoint：每 seed 训练结束自动保存 `.ckpt`，可用作恢复点

**全量运行**:
```bash
# 所有 cohorts
python DataPreprocess/per_fold_fs_all_cohorts.py

# 仅 PRO 或 RNA
python DataPreprocess/per_fold_fs_all_cohorts.py --omics PRO
python DataPreprocess/per_fold_fs_all_cohorts.py --omics RNA

# 指定单个 cohort
python DataPreprocess/per_fold_fs_all_cohorts.py --omics PRO --cohort HCC

# 冒烟测试
python DataPreprocess/per_fold_fs_all_cohorts.py --smoke_test

# 跳过已完成的（断点续跑）
python DataPreprocess/per_fold_fs_all_cohorts.py --skip_existing

# 分析已生成的结果
python DataPreprocess/per_fold_fs_analysis.py
```

**下一步**:
1. 等待当前后台实验（HCC PRO STRING, PID 960636）完成
2. 手动或通过全量脚本启动其他 cohorts 的实验
3. 全部完成后运行 `per_fold_fs_analysis.py` 生成汇总表和图表
