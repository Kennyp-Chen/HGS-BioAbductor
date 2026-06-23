---
slug: review-experiments
status: awaiting-approval
intent: clear
pending-action: write .omo/plans/review-experiments.md
approach: Sequential experiment execution - each experiment is an independent script with smoke test, no modifications to existing code.
---

# Draft: review-experiments

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->

| id | outcome | status | evidence |
|----|---------|--------|----------|
| T1-R2#19 | 三元组数据范围确认 | ✅ 已代码确认，非仅训练集 | `utils/Interpret.py:1246: self.expr = data.iloc[:,:-2]` |
| T2-R2#12 | HCC 特征筛选实验 | 🏃 已有脚本 `DataPreprocess/per_fold_feature_selection_experiment.py` | 正在运行中 |
| T3-R2#4 | 转录组消融实验 | 🏃 已有脚本 `other_experiments/HGS_ablation_study.py` | 正在运行中 |
| T4-R1Q2 | 预处理敏感性分析 + PCA/UMAP | 🔲 需新建脚本 | |
| T5-R1Q3 | 统计检验 (paired t-test, 18 datasets) | 🔲 需新建脚本 | |
| T6-R2#3 | 随机vs重要性扰动对比图 | 🔲 需新建脚本 (数据已有) | |
| T7-R2#6 | HCC 批次效应 PCA 可视化 | 🔲 需新建脚本 | |
| T8-回复 | 统一编写回复信 | 🔲 所有实验完成后统一处理 | |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->

| assumption | adopted default | rationale | reversible? |
|-----------|----------------|-----------|-------------|
| 运行环境 | 使用现有 `hgs-benchmark` conda env + CUDA | 项目已有环境 | Yes |
| 脚本位置 | `other_experiments/` 下新建独立脚本 | 不污染原有代码 | Yes |
| 结果输出 | `Results/` 下建独立的子目录 | 融合友好 | Yes |
| 回复编写 | 所有实验做完后统一写 | 实验结果是回复的内容依据 | Yes |

## Findings (cited - path:lines)

1. **三元组统计关联使用全数据 (R2 #19)** ✅
   - `utils/Interpret.py:1246`: `self.expr = data.iloc[:,:-2]` — data 包含所有病人
   - `utils/Interpret.py:1255`: `self.HRs_genes,self.data_es = self.get_HRsgene_GSVA(dn_data)` — HRs 计算于全数据
   - `utils/Interpret.py:1546`: `HRs = get_HRs(self.data, ...)` — 确认使用全数据
   - **结论**: 统计指标（Cox HR、Pearson 相关）均在完整数据集上计算，非仅训练集

2. **特征筛选实验脚本已存在 (R2 #12)**
   - `DataPreprocess/per_fold_feature_selection_experiment.py` — 1058 行完整实现
   - 支持 `--smoke_test` 参数，冒烟测试通过后跑 10 seed × 50 epoch
   - HCC PRO STRING 和 LIHC RNA Reactome 两个实验

3. **消融实验脚本已存在 (R2 #4)**
   - `other_experiments/HGS_ablation_study.py` — 402 行
   - 功能：超图层 vs GCN 层、pooling 消融、损失函数消融

4. **基准测试结果数据可用**
   - HCC PRO: `Results/Benchmark/PRO/HGS-STRING/Auto/HCC/Results-nt20.csv`
   - LIHC RNA: `Results/Benchmark/RNA/HGS-Reactome/Auto/LIHC/Results-nt20.csv`
   - 18 个数据集的结果应该分布在 `Results/Benchmark/` 下

5. **R1 Q1 贡献定位已删除** — 该任务是 xlh 的，已从 TODO 中移除

## Decisions (with rationale)

1. **LaTeX 修改搁置** — 用户明确要求论文修改部分先不处理
2. **回复统一最后写** — 回复内容依赖实验结果
3. **新建脚本不修改原有代码** — 为了融合分支方便
4. **每个脚本必须有 smoke test** — 快速验证脚本正确性

## Scope IN

- 创建 4 个新的实验脚本 (T4-T7)
- 对已有 2 个正在跑的脚本 (T2, T3) 进行冒烟测试验证
- 最终统一编写回复信和更新 draft_response_letter.md

## Scope OUT (Must NOT have)

- ❌ 不修改 LaTeX 正文 (`main.tex` 等)
- ❌ 不修改原有的模型/工具脚本 (`models/`, `utils/`, `HGS_*.py` 等)
- ❌ 不做 git push (需确认)
- ❌ 不批量删除文件

## Open questions

1. 🔲 **18 个数据集的 benchmark 结果路径** — 需要确认完整路径列表以写 paired t-test 脚本
2. 🔲 **随机扰动数据路径** — R2 #3 对比图需要确认数据在代码仓库的什么位置
3. 🔲 **HCC 数据是否有队列标注** — PCA 可视化需要知道每个样本属于哪个队列（Gao/Jiang/Xing 等）。已在 `data/PRO/HCC/` 中找 info.txt，确认 `dataset.csv` 是否有队列列。**需要你回答**
4. 🔲 **R2 #3 扰动对比数据来源** — 你说"论文中已有双向掩码和随机扰动的数据"——这些数据在代码仓库的哪个路径？还是需要重新运行实验生成？**需要你确认**
5. 🔲 **统计检验的对比方案** — 默认用 HGS-STRING（PRO）/ HGS-Reactome（RNA） vs 各基线。是否要同时也对比其他 HGS 知识库变体？**需要你确认**
6. 🔲 **三元组回复是否已写好** — 你说"应该已经做出回复"，需要确认 draft_response_letter.md 是否已包含 R2 #19 内容，还是需要我写进去

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
