# 审稿修改 TODO 总表

> 更新于 2026-06-23，新增实验完成（R2 #3 扰动对比 + R2 #6 PCA 批次效应），统计更新
>
> **回复主文档**: `docs/HGS_latex/review/npj_precision_oncology_review.md` — 所有回复写在此文件

---

## ✅ 已完成

- [x] **R2 #1** (cid #6) XAI 全称定义补充 — cyp
- [x] **R2 #15** (cid #21) 1-PCC 数学符号 + distance→dissimilarity — cyp
- [x] **R2 Typos** (cid #28) 11 处拼写/语法修正 — zqy
- [x] **R2 #22** (cid #27) INSTALLATION.md 英文化 — cyp
- [x] Review docs + AGENTS.md 入 git
- [x] **R2 #3** (cid #8) 随机 vs 按重要性扰动对比 — cyp (图+回复已提交)
- [x] **R2 #6** (cid #11) HCC 批次效应 PCA 可视化 — cyp (per-sample Z-score 对齐 pipeline)

---

## 一、📝 论文补充（LaTeX 正文修改）

| # | 审稿意见 | CID | 内容 | 负责人 | 细节 |
|---|---------|-----|------|--------|------|
| #9 | R2 #4 | 9 | **转录组消融补充** [🏃 进行中] | cyp | 有结果没写进去，cyp 找到后整合到正文 |
| #16 | R2 #11 | 16 | **数据标准化 Z-score 细节补充** | cyp | 在正文 Methods 中补充 proteomics 和 transcriptomics 数据 Z-score 标准化的计算细节 |

---

## 二、🔬 代码实验（HGS 代码仓库执行）

| # | 审稿意见 | CID | 内容 | 负责人 | 细节 | 状态 |
|--|---------|-----|------|--------|------|------|
| #1/#2 | R1 Q2 | 1,2 | **预处理敏感性分析 + HCC PCA/UMAP** | cyp | 脚本 `preprocessing_sensitivity.py` 已创建，Z-score 前后 PCA 对比图已输出到 `Results/preprocessing_sensitivity/` | ✅ 完成 |
| #3/#4/#7 | R1 Q3, R2 #2 | 3,4,7 | **统计检验** | cyp | 脚本 `statistical_test_18_datasets.py` 已创建，paired t-test 结果输出到 `Results/statistical_tests/` | ✅ 完成 |
| #8 | R2 #3 | 8 | **随机 vs 按重要性扰动对比** | cyp | 脚本 `perturbation_comparison_plot.py` 已创建，生成绝对 C-index + ΔC-index 两套图（超边 + 节点各两张），已提交到回复信 | ✅ 完成 |
| #11 | R2 #6 | 11 | **HCC 整合批次效应 PCA 可视化** | cyp | 脚本 `hcc_batch_effect_pca.py` 已修复（cohorts 类型 bug + per-sample Z-score 对齐实际 pipeline），3 队列 PCA 图已生成，回复信已更新 | ✅ 完成 |
| #18 | R2 #12 | 18 | **HCC 每个种子只训练集内特征筛选** [🏃 进行中] | cyp | **实验任务**: 在 HCC 数据上（转录组 + 蛋白组），对每个种子重新只在训练集内做特征筛选，验证结果一致性 | ⏳ 进行中 |
| #25 | R2 #19 | 25 | **三元组增强数据范围确认** | cyp | 代码确认: 统计关联使用全数据计算 `self.expr = data.iloc[:,:-2]`，非仅训练集。回复信已更新 | ✅ 完成 |

---

## 三、✉️ 回复编写（写入 npj_precision_oncology_review.md）

| # | 审稿意见 | CID | 内容 | 负责人 | 细节 | 目标文件 |
|---|---------|-----|------|--------|------|---------|
| #13 | R2 #8 | 13 | **风险组定义补充** | cyp | （撤回中）cyp 确认后决定是否在回复中补充 | → npj_precision_oncology_review.md |

> ✅ R2 #3 扰动对比图 + 回复已在 `npj_precision_oncology_review.md` 中完成
> ✅ R2 #19 三元组回复已在 `npj_precision_oncology_review.md` 中完成

---

## 四、其他负责人任务（供参考，非 cyp）

### xlh（5 项）
| CID | 审稿意见 | 内容 |
|-----|---------|------|
| 10 | R2 #5 | 讨论各类模型可解释性 |
| 12 | R2 #7 | BCLC 临床依据说明 |
| 14 | R2 #9 | 双向遮蔽 rationale 详细说明 |
| 19 | R2 #13 | Cox 400 说明 |
| 23 | R2 #17 | 超图 vs 二部图 + compatibility 公式解释 |

### zqy（5 项）
| CID | 审稿意见 | 内容 |
|-----|---------|------|
| 5 | R1 Q4 | LLM 局限性讨论 |
| 15 | R2 #10 | DEA top 蛋白文献对比 |
| 22 | R2 #16 | 两步优化折叠说明 |
| 24 | R2 #18 | Hcluster 富集分析说明 |
| 26 | R2 #20 | 时间与经济成本测算 |
| 28 | R2 #21 | PFKFB2 措辞修改 |

---

## cyp 任务统计

| 类别 | 总计 | 已完成 | 进行中 | 待处理 | 待确认 |
|------|------|--------|--------|--------|--------|
| 论文补充 (LaTeX) | 2 | 0 | 1 | 1 | 1 |
| 代码实验 (HGS repo) | 6 | 4 | 1 | 1 | 0 |
| 回复编写 | 2 | 2 | 0 | 0 | 0 |
| **合计** | **10** | **6** | **2** | **2** | **1** |

> 已完成的不计入：XAI 定义（R2 #1）、1-PCC（R2 #15）、INSTALLATION 英文化（R2 #22）— 这些已在 ✅ 区体现。
>
> "待你确认" = 需要你提供信息/结果/确认后才能执行的项。

---

## 修正记录

- 2026-06-19: 根据 docx 锚定位置修正 CID → 审稿意见映射。**关键修正：CID #0 从 R2 #3 改为 R1 Q1**
- 2026-06-19: 填入全部 10 项 cyp TODO 的确认细节；拆分为 LaTeX/实验/回复 三板块；添加 cyp 任务统计
- 2026-06-19: #0 R1 Q1 改为"区分概念创新 vs 系统集成"写法；添加"待你确认"统计列
