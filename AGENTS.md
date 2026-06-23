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

**脚本**: `DataPreprocess/per_fold_feature_selection_experiment.py`

**已知问题**:
- LIHC RNA 8559 基因下 Cox 回归极慢（疑似个别基因挂起），冒烟测试用 `--max_cox_genes 200`
- STRING H 只有 ~308/400 PRO 基因有对应，数据分片需同步过滤

**冒烟测试结果**:
| Dataset | Knowledge | Seed | Epochs | Original C-index | Per-fold C-index | Δ |
|---------|-----------|------|--------|----------------:|----------------:|--------:|
| HCC PRO | STRING | 0 | 5 | 0.6715 | 0.7020 | **+0.0304** |
| LIHC RNA | Reactome | 0 | 5 | 0.7330 | 0.5379 | -0.1951* |

\* RNA 冒烟仅用 200 基因，结果不可靠，需全量运行

**下一步**: Cyp 确认后执行全量 10 seeds × 50 epochs 实验
