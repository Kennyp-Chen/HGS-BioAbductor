# review-experiments - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->

**What you'll get:** 4 个新的实验脚本（统计检验、扰动对比、PCA可视化、预处理敏感性分析）+ 2 个已有脚本的冒烟验证 + 最终统一回复信。所有脚本都是独立的、不修改原有代码、带 `--smoke_test` 参数。

**Why this approach:** 新建脚本确保 git merge 融合安全；smoke test 确保每个实验在正式跑之前能够快速验证正确性；先实验后回复确保回复内容有数据支撑。

**What it will NOT do:** 不修改 LaTeX 正文（main.tex）、不修改原有 Python 模块、不做 git push。

**Effort:** Medium-Large（4个新脚本 + 2个验证 + 回复）
**Risk:** Medium - 部分实验需要 GPU（特征筛选训练），CUDA 资源可用性不确定
**Decisions to sanity-check:** 统计检验用 HGS 的哪个知识库版本比基线（默认：PRO=STRING, RNA=Reactome）

Your next move: 审阅计划，提供我需要确认的信息（见 Open Questions），然后 approve 开始执行。

---

> TL;DR (machine): Effort=Medium-Large, Risk=Medium. 6 experiments (4 new scripts + 2 existing smoke tests) + final response letter. All new scripts in `other_experiments/`, no modifications to existing code.

## Scope
### Must have
- 创建 4 个新的实验脚本（统计检验、扰动对比、PCA可视化、预处理敏感性分析）
- 对 2 个已有实验脚本做冒烟测试（特征筛选、转录组消融）
- 每个脚本都有 `--smoke_test` 模式（1 seed, ≤5 epochs）
- 所有实验的输出保存到独立目录
- 最终统一编写回复信 draft_response_letter.md

### Must NOT have (guardrails, anti-slop, scope boundaries)
- ❌ 不修改 `models/`, `utils/`, `HGS_*.py`, `baseline_*.py` 等原有代码
- ❌ 不修改 LaTeX 正文（搁置）
- ❌ 不做 git push（需用户确认）
- ❌ 不批量删除文件
- ❌ 每个脚本不超过 500 行（保持专注）

## Verification strategy
- Test decision: tests-after (每个脚本自带 `--smoke_test` 参数作为验收测试)
- Evidence: `.omo/evidence/task-<N>-review-experiments.md`

## Execution strategy
### Parallel execution waves
> Wave 1: 独立任务（已有脚本冒烟测试 + 三元组确认）
> Wave 2: 新脚本开发（4个可并行）+ Wave 1 结果确认
> Wave 3: 回复信编写（依赖所有实验结果）

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| T1 三元组确认 | - | - | T2 |
| T2 已有脚本冒烟测试 | - | - | T1 |
| T3 统计检验脚本 | - | - | T4, T5, T6 |
| T4 扰动对比脚本 | - | - | T3, T5, T6 |
| T5 PCA可视化脚本 | - | - | T3, T4, T6 |
| T6 预处理敏感性脚本 | - | - | T3, T4, T5 |
| T7 回复信编写 | T3, T4, T5, T6 | - | - |

## Todos
<!-- APPEND TASK BATCHES BELOW THIS LINE - never rewrite the headers above. -->

### Wave 1: 快速验证与确认

- [ ] 1. **确认三元组数据范围并更新回复 (R2 #19)**
  What to do / Must NOT do: 代码已确认三元组统计关联使用全数据（`utils/Interpret.py:1246: self.expr = data.iloc[:,:-2]`）。将此发现及回复内容更新到 docs/HGS_latex/review/draft_response_letter.md 中。不要修改其他内容。
  Parallelization: Wave 1 | Blocked by: - | Blocks: -
  References:
    - `utils/Interpret.py:1234-1629` — TripletsAnalysis 类
    - `utils/Interpret.py:1246` — `self.expr = data.iloc[:,:-2]`（全数据）
    - `utils/Interpret.py:1546` — `get_HRs(self.data, ...)`（全数据 HR）
    - `docs/HGS_latex/review/draft_response_letter.md:211-215` — 待填写的 R2 #19 回复
  Acceptance criteria (agent-executable): 
    1. `grep "R2 #19" docs/HGS_latex/review/draft_response_letter.md` 显示回复已填写
    2. 内容正确说明"统计指标在全数据集上计算"的事实
  QA scenarios: 
    - Happy: 确认回复内容与代码分析一致（全数据非仅训练集）
    - Evidence: `.omo/evidence/task-1-review-experiments.md`
  Commit: Y | `docs: Update R2 #19 triplet data scope response (R2 #19)`

- [ ] 2. **已有实验脚本冒烟测试 (R2 #12, R2 #4)**
  What to do / Must NOT do: 对已存在的实验脚本做冒烟测试，验证能否正常运行。不改动这些脚本。分别在 CPU-only debug 模式下检查脚本是否能 import 所有依赖、解析参数、并开始运行第一个 seed。
  Parallelization: Wave 1 | Blocked by: - | Blocks: -
  References:
    - `DataPreprocess/per_fold_feature_selection_experiment.py` — R2 #12 特征筛选（1058 行）
    - `other_experiments/HGS_ablation_study.py` — R2 #4 转录组消融（402 行）
    - 查看脚本是否有 `--smoke_test` 参数
  Acceptance criteria (agent-executable):
    1. `python DataPreprocess/per_fold_feature_selection_experiment.py --only_parse` 成功执行（仅解析已有结果）
    2. `python other_experiments/HGS_ablation_study.py --help` 显示帮助信息
  QA scenarios:
    - Happy: 两个脚本都能正确导入并运行 help/parse 模式
    - Failure: 缺少依赖 → 记录缺失包
    - Evidence: `.omo/evidence/task-2-review-experiments.md`
  Commit: N （不修改文件，仅验证）

### Wave 2: 新脚本开发

- [ ] 3. **创建统计检验脚本 - paired t-test on 18 datasets (R1 Q3/R2 #2)**
  What to do / Must NOT do: 
    创建 `other_experiments/statistical_test_18_datasets.py`。读取 18 个 PRO/RNA 数据集的 benchmark 结果，针对每个模型对（HGS-STRING vs DeepHit/DeepSurv/DRSA/Pnet 等）计算 paired t-test。脚本必须：
    - 使用 `--smoke_test` 参数（只跑 2 个数据集验证逻辑正确性）
    - 从各模型每个数据集的 per-seed Results.csv / Results-nt20.csv 中提取每折 test_ci
    - 对每对模型（HGS vs baseline）做 paired t-test（10 paired seeds）
    - 输出统计结果表格到 `Results/statistical_tests/`
    - 不修改任何原有文件
  Parallelization: Wave 2 | Blocked by: - | Blocks: T7
  References:
    - `Results/Benchmark/PRO/{DeepHit,DeepSurv,DRSA,Pnet}/{dataset}/Results.csv` — 基线模型 per-seed 结果
    - `Results/Benchmark/PRO/HGS-STRING/Auto/{dataset}/Results-nt20.csv` — HGS per-seed 结果
    - `Results/Benchmark/RNA/{DeepHit,DeepSurv,DRSA,Pnet}/{dataset}/Results.csv` — RNA 基线
    - `Results/Benchmark/RNA/HGS-Reactome/Auto/{dataset}/Results-nt20.csv` — RNA HGS
    - `Results/Benchmark/PRO/DeepHit/Final_Results.csv:1` — 示例 Final_Results 格式（每行一个数据集）
    - `Results/Benchmark/PRO/HGS-STRING/Auto/HCC/Results-nt20.csv:9-15` — per-seed test_ci 格式
    - `Results/Benchmark/PRO/DeepHit/HCC/Results.csv:3-15` — 基线 per-seed test_ci 格式
    - 18 个数据集: PRO=[CCRCC,GBM,HaNSCC,HCC,LA,LSCC,PDA,UCEC], RNA=[BLCA,BRCA,HNSC,KIRC,LGG,LIHC,LUAD,LUSC,OV,STAD]
  Acceptance criteria (agent-executable):
    1. `python other_experiments/statistical_test_18_datasets.py --smoke_test` 跑通（2 个数据集）
    2. 输出到 `Results/statistical_tests/smoke_test.csv`
    3. 输出的格式为 `model_pair,dataset,mean_hgs,mean_baseline,p_value,significant`
  QA scenarios:
    - Happy: smoke_test 成功输出 2 条配对检验结果
    - Failure: 缺少结果文件 → 输出清晰报错
    - Evidence: `.omo/evidence/task-3-review-experiments.md`
  Commit: Y | `feat: Add paired t-test script for 18-dataset benchmark (R1 Q3, R2 #2)`

- [ ] 4. **创建随机 vs 重要性扰动对比脚本 (R2 #3)**
  What to do / Must NOT do:
    创建 `other_experiments/random_vs_importance_perturbation.py`。对 HCC PRO STRING 数据，在同一超图上对比随机扰动和按 Borda 得分重要性移除对 C-index 的影响。脚本必须：
    - 使用 `--smoke_test` 参数（1 seed, 5 epochs）
    - 复用 `utils/hg_ops.py` 的 `disturb_H()` 做随机扰动
    - 复用 `utils/Borda_Score_validator.py` 的 Borda 验证方法做重要性消融
    - 在同一图（matplotlib）上绘制两条 C-index 衰减曲线
    - 输出图片到 `Results/perturbation_comparison/`
    - 不修改任何原有文件
  Parallelization: Wave 2 | Blocked by: - | Blocks: T7
  References:
    - `utils/hg_ops.py:494-524` — `disturb_H()` 随机扰动函数
    - `utils/Borda_Score_validator.py:72-238` — Borda 得分验证（top/bottom 移除）
    - `utils/Borda_Score_validator.py:128-179` — `validate_hyperedge_borda_scores()`
    - `HGS_XAI.py:69-91` — `borda_score_validation()` 协调器
    - `other_experiments/HGS_ablation_study.py:150-151` — `disturb_H` 调用示例
    - `models/_Models_interpret.py:1030-1269` — `HGS_ablation` 类
    - TODO.md: R2 #3 说"数据在代码仓库里"（论文中已有双向掩码和随机扰动的数据）
  Acceptance criteria (agent-executable):
    1. `python other_experiments/random_vs_importance_perturbation.py --smoke_test` 跑通
    2. 输出 `Results/perturbation_comparison/random_vs_importance.png`
    3. 图中包含两条曲线（随机扰动/重要性移除）
  QA scenarios:
    - Happy: smoke_test 生成 png 图片
    - Failure: GPU 不可用 → fallback 到 CPU
    - Evidence: `.omo/evidence/task-4-review-experiments.md`
  Commit: Y | `feat: Add random vs importance perturbation comparison script (R2 #3)`

- [ ] 5. **创建 HCC 批次效应 PCA 可视化脚本 (R2 #6)**
  What to do / Must NOT do:
    创建 `other_experiments/hcc_batch_effect_pca.py`。对 HCC 多队列蛋白质组数据做 PCA 可视化，证明 Z-score 标准化后无显著批次效应。脚本必须：
    - 使用 `--smoke_test` 参数（只运行 1 个队列验证）
    - 从 `data/PRO/HCC/` 读取 HCC 数据
    - 绘制 PCA 散点图（按队列着色），Z-score 前后对比
    - 输出图片到 `Results/batch_effect_pca/`
    - 不修改任何原有文件
  Parallelization: Wave 2 | Blocked by: - | Blocks: T7
  References:
    - `data/PRO/HCC/dataset.csv` — HCC 蛋白质组数据（CSV，index_col=0）
    - `data/PRO/HCC/info.txt` — HCC 队列信息（如果有）
    - `DataPreprocess/FS_COX.py:80-91` — HCC 多队列跨队列处理示例
    - TODO.md #11: "做 PCA 可视化证明整合后无显著批次效应"
    - `docs/HGS_latex/review/draft_response_letter.md:122-128` — R2 #6 回复草稿
  Acceptance criteria (agent-executable):
    1. `python other_experiments/hcc_batch_effect_pca.py --smoke_test` 跑通
    2. 输出至少 2 张 PCA 图（Z-score 前/后）到 `Results/batch_effect_pca/`
    3. 图中点按队列用不同颜色标记
  QA scenarios:
    - Happy: smoke_test 生成图片
    - Failure: HCC 缺少队列标注 → 从 info.txt 或文件名推断
    - Evidence: `.omo/evidence/task-5-review-experiments.md`
  Commit: Y | `feat: Add HCC cohort batch effect PCA visualization script (R2 #6)`

- [ ] 6. **创建预处理敏感性分析脚本 (R1 Q2)**
  What to do / Must NOT do:
    创建 `other_experiments/preprocessing_sensitivity.py`。分析 Cox p-value 阈值选择对特征筛选和模型性能的影响。脚本必须：
    - 使用 `--smoke_test` 参数（1 个数据集, 1 seed）
    - 对 HCC PRO 数据尝试不同 Cox p-value 阈值（0.01, 0.05, 0.1）
    - 对每个阈值筛选特征后训练 HGS 模型，记录 C-index
    - 可选：生成 Z-score 前后对比的 PCA/UMAP 图
    - 输出结果到 `Results/preprocessing_sensitivity/`
    - 不修改任何原有文件
  Parallelization: Wave 2 | Blocked by: - | Blocks: T7
  References:
    - `DataPreprocess/Preprocess/DATA_preprocess.py:172-218` — `cox_feature_selection()` 函数
    - `utils/data_utils.py` — 数据加载、Z-score 标准化
    - `DataPreprocess/FS_COX.py` — Cox 特征筛选预处理流程
    - TODO.md #1/#2: "特征筛选用 Cox p-value"; "需要重新做 Z-score 前后对比的 PCA/UMAP"
  Acceptance criteria (agent-executable):
    1. `python other_experiments/preprocessing_sensitivity.py --smoke_test` 跑通（1 阈值 × 1 seed）
    2. 输出结果表到 `Results/preprocessing_sensitivity/`
  QA scenarios:
    - Happy: smoke_test 输出 CSV 结果
    - Failure: GPU 不可用 → fallback 到 CPU
    - Evidence: `.omo/evidence/task-6-review-experiments.md`
  Commit: Y | `feat: Add preprocessing sensitivity analysis script (R1 Q2)`

### Wave 3: 统一回复

- [ ] 7. **统一更新回复信 (all R2)**
  What to do / Must NOT do:
    在所有实验完成后，更新 `docs/HGS_latex/review/draft_response_letter.md`：
    - 填入实验结果数据（统计检验 p-value、扰动对比图、PCA 图等）
    - 统一所有回复的语调和格式
    - 不要修改回复信以外的文件
    - 删除 R1 Q1 相关（已交给 xlh）
  Parallelization: Wave 3 | Blocked by: T3, T4, T5, T6 | Blocks: -
  References:
    - `docs/HGS_latex/review/draft_response_letter.md` — 完整草稿（254 行）
    - `Results/statistical_tests/` — 统计检验结果
    - `Results/perturbation_comparison/` — 扰动对比图
    - `Results/batch_effect_pca/` — PCA 图
    - `Results/preprocessing_sensitivity/` — 敏感性分析结果
  Acceptance criteria (agent-executable):
    1. 所有实验的回复段落都填充了具体结果
    2. 不再有"待填写"标记
    3. R1 Q1 已移除
  QA scenarios:
    - Happy: 回复信中所有待填写项已填充
    - Failure: 某实验未完成 → 标注为 pending
    - Evidence: `.omo/evidence/task-7-review-experiments.md`
  Commit: Y | `docs: Update response letter with experiment results`

## Final verification wave
- [ ] F1. Plan compliance — 所有 7 个 todo 完成
- [ ] F2. 脚本规范检查 — 每个新脚本都有 `--smoke_test`，不修改原有代码
- [ ] F3. 输出检查 — 每个脚本的输出都保存到独立的 Results 子目录
- [ ] F4. 提交记录检查 — commit message 格式正确

## Commit strategy
每个 todo 对应一个独立 commit（T2 不修改文件则不提交）:
```
feat: Add paired t-test script for 18-dataset benchmark (R1 Q3, R2 #2)
feat: Add random vs importance perturbation comparison script (R2 #3)
feat: Add HCC cohort batch effect PCA visualization script (R2 #6)
feat: Add preprocessing sensitivity analysis script (R1 Q2)
docs: Update R2 #19 triplet data scope response
docs: Update response letter with experiment results
```

## Success criteria
1. ✅ 所有 4 个新脚本 smoke test 通过
2. ✅ 三元组数据范围确认并更新到回复信
3. ✅ 统计检验输出 18 数据集 × 4 基线的 paired t-test 结果
4. ✅ 扰动对比图已生成
5. ✅ PCA 可视化图已生成
6. ✅ 预处理敏感性分析结果已输出
7. ✅ 回复信已更新包含所有实验结果
