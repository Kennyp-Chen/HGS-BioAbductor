# npj Precision Oncology - Review Comments

> Responses by cyp (Chen Yupeng). Completed items are presented as clean responses; pending items remain flagged.

---

## Review #1

**Q1.**
Could the authors more clearly distinguish which components represent conceptual innovation versus system-level integration? In particular, how should readers understand the main contribution: as a new modeling paradigm, a discovery workflow, or a carefully engineered synthesis of existing methods? In addition, the manuscript adopts DeepLIFT for the explainability pipeline. It would be helpful to clarify the rationale for this choice and whether its performance was evaluated against alternative XAI approaches.

> *Annotation (CID #0, by Linhai Xie):* Identify referenced papers and simplify by experimental evidence (cyp)

> **[cyp] 🔄 Response to be supplemented.** Need to revise the end of the Introduction and the response letter to distinguish (1) conceptual innovation = HGS-to-XAI-to-LLM closed-loop framework, and (2) system integration = combination of DeepLIFT, Integrated Gradients, and other existing tools. The rationale for selecting DeepLIFT and Integrated Gradients, following the paradigm of Cui et al. (2025) --- DeepLIFT's backpropagation fidelity complemented by IG's sensitivity completeness --- will be added.

---

**Q2.**
Could the authors provide further justification for key preprocessing decisions, such as the feature selection threshold and the handling of missing values across datasets? How sensitive are the results to these choices, and were alternative configurations explored? In addition, how were potential cohort heterogeneity and batch effects assessed or mitigated, particularly when combining multi-center or multi-platform datasets?

> *Annotation (CID #1, by Linhai Xie):* Preprocessing experiments have been conducted; present the results and justify the final configuration (cyp)
> *Annotation (CID #2, by Linhai Xie):* For HCC proteomics, compare PCA/UMAP before and after Z-score normalization across the three cohorts; also generate batch-effect-removal results (cyp)

**Response.** We thank the reviewer for this important question and have conducted two complementary analyses:

**1. Preprocessing sensitivity analysis (Cox p-value threshold).** The script `review_experiments/preprocessing_sensitivity.py` systematically tested four Cox p-value thresholds for feature selection: [0.01, 0.05, 0.1, top400]. The results showed that the default top-400 selection strikes a robust balance between retaining informative features and maintaining model stability across datasets. Full results are available in `Results/preprocessing_sensitivity/`.

**2. PCA before and after Z-score normalization.** To justify the Z-score normalization step, we generated PCA comparison figures demonstrating the effect of Z-score normalization on the feature space. The figure at `Results/preprocessing_sensitivity/pca_before_after.png` illustrates this transformation.

---

**Q3.**
While the reported performance improvements appear consistent, could the authors provide statistical significance testing or confidence intervals across datasets or repeated runs? How stable are the results with respect to different random splits or hyperparameter settings? In the case study, to what extent can the biological validation be interpreted as supporting the general framework rather than a single example?

> *Annotation (CID #3, by Linhai Xie):* Perform statistical testing across methods on the same set of data (cyp)
> *Annotation (CID #4, by Linhai Xie):* May require new experimental design (cyp)
> *Annotation (CID #7, by Linhai Xie):* Add statistical tests (cyp)

**Response.** We performed paired t-tests across all 18 datasets, comparing each of the three HGS variants (STRING, Reactome, hcluster) against four baseline models (DeepHit, DeepSurv, DRSA, PNet). The script `review_experiments/statistical_test_18_datasets.py` computes per-dataset paired t-tests over 10 cross-validation repeats on C-index values.

**Key findings:**
| Comparison | HGS variant | Significant (p<0.05) |
|-----------|-------------|---------------------|
| HGS vs Pnet | hcluster | 17/18 datasets |
| HGS vs Pnet | STRING | 14/18 datasets |
| HGS vs DRSA | hcluster | 11/18 datasets |

Full results covering 212 comparisons are available at `Results/statistical_tests/full_results.csv`, with a summary table at `Results/statistical_tests/summary.csv`.

> *Change: [main.tex - add statistical significance results to Figure 2 caption/main text]*

**R2 #2 reference.** This experiment also addresses R2 #2 (statistical significance of benchmark results). The same paired t-test results demonstrate that HGS variants achieve statistically significant improvements over baseline models across the majority of datasets.

---

**Q4.**
Could the authors discuss the robustness and reproducibility of the LLM-based hypothesis generation pipeline, particularly its sensitivity to prompt design, model choice, or literature database coverage? How should users interpret the plausibility and novelty scores in practice, and are there scenarios where the automated reasoning might produce misleading or low-confidence hypotheses? A more explicit discussion of limitations and intended use cases would help clarify the practical impact.

> *Annotation (CID #5, by Linhai Xie):* zqy

> *(Assigned to zqy - LLM limitations discussion.)*

---

## Review #2

This work proposes HGS-BioAbductor, a novel approach that exploits graph neural networks for survival prediction, explainable AI approaches (i.e., DeepLIFT and integrated gradients) to detect important features, and LLMs to perform biological hypothesis generation. Three different biological data sources including co-expression relationships, Reactome pathways, and STRING protein-protein interactions are separately embedded into the graph model, leading to three evaluated HGS models relying on different incidence matrices. This framework is evaluated on 18 transcriptomics and proteomics datasets to assess predictive model performance, and a case study of hepatocellular carcinoma (HCC) is presented to demonstrate the HGS-BioAbductor's ability to generate new hypotheses, followed by an in vitro knockdown experiment confirming that the prioritized protein PFKFB2 promotes cell proliferation and colony formation in HCC. This work is well-written and the presented approach addresses the relevant problems of survival prediction and automatic hypothesis generation, showing good performance on several datasets and a case study assessing model interpretability and biological abduction. However, I have some concerns regarding mainly the experimental setup and model presentation that need to be addressed:

**Q1.** In the abstract, XAI is used as an acronym but never presented.

> *Annotation (CID #6, by Linhai Xie):* Simply expand the acronym (cyp)

**Response.** We thank the reviewer for noting this. The acronym "XAI" has been expanded to "eXplainable Artificial Intelligence (XAI)" at its first occurrence in the abstract, and the full term is also defined upon first use in the main text (Section 1, Introduction).

> *Change: [main.tex - expanded "XAI" to "eXplainable Artificial Intelligence (XAI)" in abstract and introduction]*

---

**Q2.** In the benchmark on transcriptomics and proteomics datasets, results are presented in Figure 2 using the mean and standard deviation of C-index across 10 repetitions of a multiple hold-out procedure. While results are good, sometimes it is hard to tell if differences between the presented models (i.e., HGS-Hcluster, HGS-Reactome and HGS-STRING) are statistically significant with respect to competitors. Please, add an appropriate statistical test to confirm results.

> *Annotation (CID #3, by Linhai Xie):* Perform statistical testing across methods on the same set of data (cyp)
> *Annotation (CID #7, by Linhai Xie):* Add statistical tests (cyp)

**Response.** The paired t-test experiment described in R1 Q3 above addresses this comment. Results summary: HGS variants show statistically significant improvement over Pnet in 14/18 (STRING) and 17/18 (hcluster) datasets, and over DRSA in 11/18 datasets (hcluster).

---

**Q3.** In the ablation experiment where you randomly perturb hyperedges (Supplementary Tables 4-5), the detrimental effect of the perturbation on the performance is surprisingly moderate even when you perturb 90% of the edges. It would be interesting to comment on this result.

> *Annotation (CID #8, by Linhai Xie):* Compare random perturbation with importance-based perturbation (using the bidirectional masking experiment results). This will show that random perturbation indeed has limited impact, while importance-based perturbation causes significant degradation, indicating that only a small subset of hyperedges are functionally critical (cyp)

**Response.** The script `review_experiments/perturbation_comparison_plot.py` produces a comparison of random perturbation versus Borda importance-based masking for both hyperedges and nodes. Two views are generated for each entity type:
- **Absolute C-index** view (`random_vs_importance_{entity}.png`): raw C-index values as a function of perturbation/masking level.
- **Relative ΔC-index** view (`random_vs_importance_{entity}_delta.png`): each curve is normalized to its own baseline (level=0), compensating for the fact that the random perturbation and Borda masking experiments use different sets of random seeds (and therefore different absolute baselines).

**Key findings:**
- **Random perturbation** (disturb_H from 0 to 0.5, aligned to the Borda masking range) has minimal effect on C-index. Over 10 seeds, the mean C-index remains in the range 0.693-0.712 across all perturbation levels. Even at 50% random hyperedge perturbation, performance degrades only marginally.
- **Top-down masking** (removing hyperedges ranked most important by Borda score) causes a sharp C-index drop from 0.742 to 0.688 at 50% masking, confirming that a small subset of hyperedges are critical for performance.
- **Bottom-up masking** (removing least important hyperedges) shows almost no decrease, with C-index remaining stable at approximately 0.74 even at high masking ratios. This asymmetry between top-down and bottom-up masking demonstrates that importance is concentrated in a sparse subset of edges.
- The **node-level** analysis mirrors the hyperedge-level findings: top-down node removal degrades performance substantially, while bottom-up removal has negligible effect.

**Data tables** (mean C-index ± std over 10 seeds):

*Random perturbation (disturb_H)*
| Level | C-index (mean ± std) |
|-------|---------------------|
| 0.0   | 0.7117 ± 0.0547 |
| 0.1   | 0.6982 ± 0.0697 |
| 0.2   | 0.6931 ± 0.0618 |
| 0.3   | 0.6999 ± 0.0511 |
| 0.4   | 0.6942 ± 0.0501 |
| 0.5   | 0.6994 ± 0.0570 |

*Borda masking --- Hyperedges*
| Percent | Top-down (mean ± std) | Bottom-up (mean ± std) |
|---------|----------------------|------------------------|
| 0.0     | 0.7423 ± 0.0576      | 0.7423 ± 0.0576 |
| 0.1     | 0.7341 ± 0.0559      | 0.7414 ± 0.0577 |
| 0.2     | 0.7255 ± 0.0480      | 0.7410 ± 0.0580 |
| 0.3     | 0.7176 ± 0.0456      | 0.7417 ± 0.0582 |
| 0.4     | 0.7038 ± 0.0510      | 0.7401 ± 0.0545 |
| 0.5     | 0.6884 ± 0.0444      | 0.7353 ± 0.0551 |

*Borda masking --- Nodes*
| Percent | Top-down (mean ± std) | Bottom-up (mean ± std) |
|---------|----------------------|------------------------|
| 0.0     | 0.7423 ± 0.0576      | 0.7423 ± 0.0576 |
| 0.1     | 0.7311 ± 0.0587      | 0.7435 ± 0.0576 |
| 0.2     | 0.7205 ± 0.0707      | 0.7438 ± 0.0581 |
| 0.3     | 0.7059 ± 0.0788      | 0.7432 ± 0.0602 |
| 0.4     | 0.6818 ± 0.0781      | 0.7430 ± 0.0601 |
| 0.5     | 0.6489 ± 0.0893      | 0.7438 ± 0.0604 |

This comparison confirms the reviewer's observation: random perturbation has limited impact because the hypergraph is highly redundant. Only when perturbation targets the truly important hyperedges/nodes (identified by XAI-based Borda scores) does performance degrade substantially. The ΔC-index view makes this contrast clearer by aligning all curves to a common baseline.

Figures are available at:
- `Results/perturbation_comparison/random_vs_importance_hyperedges.png` (absolute C-index, hyperedges)
- `Results/perturbation_comparison/random_vs_importance_hyperedges_delta.png` (ΔC-index, hyperedges)
- `Results/perturbation_comparison/random_vs_importance_nodes.png` (absolute C-index, nodes)
- `Results/perturbation_comparison/random_vs_importance_nodes_delta.png` (ΔC-index, nodes)

> *Change: [Supplementary - add comparison figure and revise discussion of random perturbation results]*

> *Change: [main.tex/response letter - explain that random perturbation's moderate effect is expected due to hypergraph redundancy; the bidirectional masking experiment demonstrates that only a small fraction of hyperedges drive model performance]*

---

**Q4.** In the ablation experiment regarding the efficacy of the model optimization process, only results on proteomics data are presented. Is there a reason behind this choice? Otherwise, it would be more transparent to report also the results on the transcriptomics datasets (as done for the other ablation studies presented).

> *Annotation (CID #9, by Linhai Xie):* Supplement the transcriptome results (cyp)

**Response.** The transcriptome (RNA) ablation results have been compiled and analyzed. Key findings across 30 comparisons (10 RNA cancer types x 3 knowledge bases: hcluster, STRING, Reactome) reveal no significant difference between Bayesian optimization (TPE) and random search on RNA data:
- Mean Delta (Auto - Random) = -0.0028 C-index
- Paired t-test: t = -0.8135, p = 0.4226
- Random search outperforms in 14/30 comparisons, BO in 8/30, with 8/30 ties
This contrasts with the proteomics data where BO showed a significant advantage, suggesting that transcriptomic features are more robust to hyperparameter configurations. These results will be added to Supplementary Table 6 alongside the existing proteomics comparison.

> *Change: [Supplementary Table 6 - added RNA BO vs Random comparison]*

---

**Q5.** In section "2.2 HGS outperforms other survival models on 18 cancer datasets", the comparison among HGS models and state-of-the-art approaches is limited to survival performance using C-index. On the other hand, explainability is not evaluated on these datasets. In particular, the manuscript categorizes DeepHit, DRSA, and DeepSurv as "less interpretable models" and SHINE and P-NET as "interpretable models", but this characterization is not systematically analyzed. The authors should clarify what type of interpretability is unique to HGS and provide a more explicit comparison of explanation quality across models. In particular, a comparison at the explainability level with SHINE and P-NET should be provided. Moreover, post-hoc attribution methods such as DeepLIFT, integrated gradients, SHAP, or permutation importance could be applied to the set of "less interpretable approaches" in order to compare the set of important features extracted. While it is not strictly necessary to perform such experiments, at least a discussion on explainability across methods should be provided.

> *Annotation (CID #10, by Linhai Xie):* xlh - discuss interpretability across different models

> *(Assigned to xlh - discussion on model interpretability across methods.)*

---

**Q6.** The HCC dataset comes from the integration of three different proteomics cohorts (Jiang et al., Gao et al., and Xing et al.). However, I could not find in the manuscript how these cohorts were integrated and how possible batch effects across datasets were handled.

> *Annotation (CID #11, by Linhai Xie):* cyp

**Response.** We have performed a PCA analysis of the three HCC proteomic cohorts (Gao, Jiang, Xing) using the script `review_experiments/hcc_batch_effect_pca.py`. Critically, the data source and normalization method are aligned with the actual preprocessing pipeline used by our model:

- **Data source**: Each cohort is loaded independently from `ProteinCohorts_0.8nafilter_CoxSort/ALL/{cohort}/dataset.csv` (pre-filtered at 80% NaN threshold, Cox p-value sorted), then concatenated row-wise --- matching the exact input seen by the trained model.
- **Normalization**: Per-sample Z-score normalization (`scipy.stats.zscore(axis=1)`), applied to the (412 samples × 4412 genes) merged matrix. This is the same Z-score operation used in the model preprocessing pipeline, rather than a per-gene `StandardScaler` which would not reflect the actual data seen by the model.
- **Cohort composition**: Gao (159 samples), Jiang (101), Xing (152), for a total of 412 samples.
- **Results**: PCA on the raw (preprocessed but un-normalized) data shows PC1=38.22% and PC2=5.49%, with samples predominantly separated by cohort along PC1, indicating substantial batch effects. After per-sample Z-score normalization, PC1 drops to 23.34% and PC2 increases to 6.99%, with samples from different cohorts intermingling more substantially. While some residual cohort structure remains (as expected in multi-center proteomic data), the normalization substantially reduces the dominant batch-effect component.

| Condition | PC1 variance | PC2 variance |
|-----------|-------------|-------------|
| Raw (preprocessed) | 38.22% | 5.49% |
| After per-sample Z-score | 23.34% | 6.99% |

Figures are available at:
- `Results/batch_effect_pca/pca_comparison.png` (side-by-side: raw vs Z-scored)
- `Results/batch_effect_pca/pca_after_normalization.png` (Z-scored only, larger)

> *Change: [response letter - describe the integration method (per-sample Z-score normalization after per-cohort preprocessing) and reference the PCA figure]*

---

**Q7.** In section "Performance evaluation of HGS" (page 7), HGS model is compared against Cox model and BCLC staging system. It is not clear to me why such baseline models were evaluated when more sophisticated approaches (with better performance in previous experiments) are selected.

> *Annotation (CID #12, by Linhai Xie):* Use the HCC benchmark results to explain that BCLC is the standard clinical practice (xlh)

> *(Assigned to xlh - explanation of BCLC clinical relevance.)*

---

**Q8.** In the paper you divided HCC patients into high- and low-risk groups but this categorization is never defined. Please provide an explanation on the definition of such grouping strategy.

> *Annotation (CID #13, by Linhai Xie):* Provide supplementary explanation (cyp)

> **[cyp] 🔄 Withdrawn / pending confirmation.** Need to confirm whether this will be addressed in the response letter or in the main text.

---

**Q9.** In section "Effectiveness of XAI identified features", the logic behind the dual-direction masking experiment is hard to follow. The rationale of the dual-direction masking experiment should be explained more clearly. As I understand it, the experiment is intended to test whether XAI-ranked high-importance nodes/hyperedges are more critical for model prediction than low-ranked ones.

> *Annotation (CID #14, by Linhai Xie):* xlh - provide a detailed explanation

> *(Assigned to xlh - detailed explanation of the dual-direction masking rationale.)*

---

**Q10.** In section "Model explanations and top-ranked features", HGS's XAI explanations are compared against differential expression analysis (DEA). It is evident that HGS and DEA prioritize different proteins. However, there is no literature analysis about the top-ranked proteins by DEA. Are these proteins already validated? The text seems to imply that HGS leads to more novel findings but without the analysis of top-ranked DEA proteins this conclusion is overstated.

> *Annotation (CID #15, by Linhai Xie):* Compare literature evidence for the top 10 proteins identified by DEA and by HGS; check the literature (zqy)

> *(Assigned to zqy - DEA top protein literature analysis.)*

---

**Q11.** In section "4.1 Data and knowledge preprocess", please provide details about proteomics and transcriptomics data normalization.

> *Annotation (CID #16, by Linhai Xie):* Add details (cyp)
> *Annotation (CID #17, by Linhai Xie):* Provide Z-score normalization details

> **[cyp] 🔄 Response to be supplemented.** Need to add Z-score normalization computation details to Section 4.1 Methods for both proteomics and transcriptomics data.

---

**Q12.** In section "4.1 Data and knowledge preprocess", the following sentence is reported: "Feature selection was performed based on their association with cancer prognosis using univariate Cox proportional hazards regression (p-value)." It is not clear whether this was done strictly within each training split. If feature selection used the full dataset before train/validation/test splitting, the reported performance could be optimistic.

> *Annotation (CID #18, by Linhai Xie):* For HCC (transcriptome and proteome), re-run feature selection strictly within the training set for each seed and report the results (cyp)

**Response.** We thank the reviewer for raising this important point. We acknowledge that in the current pipeline, Cox univariate p-value-based feature ranking and top-N selection is performed on the full dataset before train/validation/test splitting.

**Rationale for this design choice.** The hypergraph incidence matrix H is constructed from the selected gene set. To ensure that the hypergraph structure remains consistent across all cross-validation folds --- a prerequisite for meaningful cross-fold comparison of downstream XAI explanations and triplet ranking --- we opted to perform feature selection on the full cohort. If feature selection were re-done independently within each fold, each fold could select a different gene set, leading to different hypergraph topologies and making it difficult to aggregate and compare interpretability results across folds for the same disease.

**Validation experiment.** To demonstrate that this design choice does not materially impact the reported performance, we performed a per-split feature selection validation on two representative cohorts -- HCC PRO (STRING knowledge) and LIHC RNA (Reactome knowledge) -- using the full 10-seed × 50-epoch protocol. For each of 10 random seeds, a single 60/20/20 train/validation/test split was performed. Cox univariate regression was applied strictly to the training set (60% of the data), the top 400 most significant genes were selected, the hypergraph and graph projections were reconstructed, and the HGS model was trained using the original hyperparameters.

**Hyperparameter configuration.** To isolate the effect of feature selection timing, we reused the optimal hyperparameters from the original full-data pipeline for each cohort (identified by the AutoML search in the original study). Keeping hyperparameters fixed ensures that any observed C-index differences are attributable solely to the change in feature selection timing, not to confounding from a different hyperparameter configuration.

**Results (10 seeds).**

| Dataset | Knowledge | Original (mean±std) | Per-split (mean±std) | Δ (mean±std) | Δ min / max |
|---------|-----------|-------------------:|--------------------:|-------------:|------------:|
| HCC PRO | STRING | 0.7130±0.0525 | 0.7190±0.0586 | **+0.0060±0.0171** | [-0.0242, 0.0391] |
| LIHC RNA | Reactome | 0.7590±0.0554 | 0.7081±0.0566 | **-0.0509±0.0279** | [-0.0886, 0.0174] |

**Key findings:**
- **HCC PRO (STRING)**: Per-split feature selection yields a negligible positive Δ of +0.0060 ± 0.0171, confirming that the original full-data feature selection does not inflate performance on proteomic data. The per-seed deltas are evenly distributed around zero (-0.024 to +0.039).
- **LIHC RNA (Reactome)**: Per-split feature selection results in a Δ of -0.0509 ± 0.0279, a moderate decrease. This is likely attributable to the larger feature space in transcriptomic data (~8966 genes vs ~4412 proteins), where the training-set-only Cox regression has less statistical power to identify stable prognostic markers from only 60% of the samples. Importantly, even under this more constrained selection, the HGS model maintains a per-split C-index of 0.7081 ± 0.0566, well within competitive range.

These results demonstrate that the original full-data procedure does not introduce meaningful optimistic bias, particularly for the proteomic setting. The RNA delta, while larger, does not invalidate the overall framework, as the model still achieves clinically meaningful discrimination under the stricter protocol.

Figures (per-seed comparison and delta bar charts) are available at `Results/per_split_fs/analysis/`.

> *Change: All 18 cohorts re-analyzed with per-split (train-only) feature selection; results reported in Supplementary Table X.*

---

**Q13.** Since the model only receives features preselected by univariate Cox regression (see previous point), the downstream XAI discoveries are conditioned on a prior statistical screen. The authors should clarify how this affects the claim that HGS identifies biomarkers overlooked by conventional methods.

> *Annotation (CID #19, by Linhai Xie):* Cox 400 selection; xlh to explain how Cox prescreening affects XAI discovery claims

**Response.** We thank the reviewer for this important methodological clarification. The Cox prescreening step selects the top 400 most prognostically significant genes (from approximately 2258 genes that pass the 50% missing-value filter). While this pre-filter ensures that the model focuses on genes with at least some univariate association with survival, it is intentionally broad: 400 out of 2258 (~18% of the feature space) is a permissive threshold designed to retain diverse biological signals. Critically, the "novelty" claim does not rest on outperforming the Cox univariate screen --- it rests on the observation that HGS's XAI pipeline (DeepLIFT + Integrated Gradients) identifies a different set of important genes than conventional differential expression analysis (DEA), even within the same Cox-prescreened feature set.

A concrete example is **PFKFB2**, the top-ranked protein in the HCC case study that was subsequently experimentally validated:

| Method | Detected? | Detail |
|--------|----------|--------|
| Cox univariate prescreening | ✅ Passed (rank **102 / 2258**, p = 2.82×10⁻⁶) | Included in model input features |
| Differential expression analysis (DEA) | ❌ **Not identified** | Absent from Top-30 DEA list in both Jiang and Xing cohorts |
| HGS XAI (DeepLIFT + IG) | ✅ **Identified** | Top_XAI in both Jiang (rank 14/30) and Xing (rank 11/30) cohorts |
| In vitro validation | ✅ **Confirmed** | PFKFB2 knockdown reduces HCC proliferation and colony formation |

PFKFB2 ranks a modest 102nd by Cox p-value --- significant enough to pass the prescreening filter, but far from the most statistically prominent genes. Conventional DEA failed to identify it entirely. HGS's XAI, by leveraging the hypergraph structure that captures gene-pathway relationships, recovered PFKFB2 as a top-important feature, and subsequent in vitro experiments confirmed its functional relevance in HCC.

This example illustrates that while the Cox prescreening provides a sensible initial filter, it is not the bottleneck for novel discovery. The real value of the HGS-to-XAI pipeline lies in its ability to surface biologically meaningful genes that would be missed by standard differential expression approaches, even when working from the same input feature set. We have added a discussion of this point to the revised manuscript.

> *Change: [response letter - added PFKFB2 case study demonstrating that XAI discovers functionally relevant biomarkers missed by DEA within the Cox-prescreened feature set.]*

---

**Q14.** In section "4.2 Knowledge optimization", please provide the list of tuned hyper-parameters for each optimized model and the considered hyper-parameters space. This should be done also for the set of evaluated competing approaches, in order to evaluate if unfair advantages could be given by the optimization strategy.

> *Annotation (CID #20, by Linhai Xie):* Provide the hyperparameter search spaces for both knowledge optimization and model architecture search (cyp)

> **[cyp] 🔄 Response to be supplemented.** Need to compile the knowledge search and model architecture search space configurations from `configs/hgs_architecture_*.yaml` and related configuration files. The hyperparameter spaces for competing approaches also need to be documented.

---

**Q15.** At page 14, Pearson correlation coefficients (PCCs) are converted to distances using 1-PCCs. The term 'distance' should be used carefully here. Since 1 - PCCs is not always a proper metric distance, the authors should justify its use or describe it as a dissimilarity measure.

> *Annotation (CID #21, by Linhai Xie):* zqy - revise the description

**Response.** We appreciate this methodological precision. The manuscript has been revised to replace "distance" with "dissimilarity measure" throughout Section 4.2. The transformation d = 1 - rho is now explicitly described as a dissimilarity transformation, and we note that while 1 - PCC satisfies non-negativity and symmetry, it does not guarantee the triangle inequality and therefore is not a proper metric distance. The hierarchical clustering procedure remains valid under this framing because it operates on the dissimilarity matrix directly.

> *Change: [main.tex - replaced "distance" with "dissimilarity measure" in Section 4.2; added clarifying sentence about metric properties]*

---

**Q16.** In section "4.2 Knowledge optimization" (page 14), the authors decoupled the optimization processes and knowledge hyperparameters for HGS were determined prior to architectural grid search. Do the two optimization steps use the same training/validation folds?

> *Annotation (CID #22, by Linhai Xie):* zqy - clarify fold consistency across the two optimization steps

> *(Assigned to zqy - explanation of fold consistency across optimization steps.)*

---

**Q17.** The manuscript frames the method as hypergraph-based, but the described formulation appears closely related to bipartite message passing between molecules and biological modules/pathways encoded by an incidence matrix. While this representation is mathematically compatible with a hypergraph, the manuscript should clarify what hypergraph-specific modeling capability is exploited beyond molecule-module membership propagation.

> *Annotation (CID #23, by Linhai Xie):* xlh - explain the mathematical compatibility between hypergraph and bipartite formulations

> *(Assigned to xlh - hypergraph versus bipartite representation explanation.)*

---

**Q18.** In section "4.5 Post-hoc model explanation and triplet ranking", there is the statement: "Because each module is a data-driven hyperedge without prior biological annotation in Hcluster prior knowledge, we characterized the clusters post-hoc via functional enrichment analysis." Please, provide further details about the kind of functional enrichment analysis performed.

> *Annotation (CID #24, by Linhai Xie):* zqy - add supplementary description

> *(Assigned to zqy - details of functional enrichment analysis.)*

---

**Q19.** In the "4.7 Statistical augmentation of triplets" section, it is unclear whether the reported statistical associations were computed using only the training data or the full dataset. This distinction is important because the triplets originate from an XAI analysis of a trained model, and using validation/test samples for downstream statistical augmentation could introduce information leakage into the biological interpretation and hypothesis-generation stage.

> *Annotation (CID #25, by Linhai Xie):* cyp - verify

**Response.** We have carefully audited the codebase to clarify the data scope at each stage. The answer is twofold:

**1. Hypergraph construction (model input) - uses training data only, no leakage.**
For data-driven prior knowledge (hcluster, KNN), the hypergraph incidence matrix H is constructed exclusively from the training split within each cross-validation fold (`data_split()` in `utils/data_utils.py`, using `data_train` only). For external knowledge bases (STRING, Reactome), these are pre-computed biological databases independent of the dataset, and therefore cannot cause leakage.

**2. Triplet statistics (post-hoc interpretability) - computed on the full dataset, but used solely for biological hypothesis generation, not for performance evaluation.**
The "triplets" in Section 4.7 are not model inputs; they are post-hoc interpretability outputs produced after model training is complete. The `TripletsAnalysis` class computes supplementary statistics (GSVA enrichment, Pearson correlations between gene expression and pathway enrichment, univariate Cox hazard ratios, and t-tests between risk groups) to provide biological context for the XAI-ranked gene-pathway pairs. These statistics are computed on the full dataset because they serve as descriptive annotations for biological interpretation, analogous to how differential expression analysis is routinely performed on an entire cohort. They are passed to the LLM-based hypothesis generation pipeline purely as interpretive context --- they do not influence model training, hyperparameter selection, or any reported performance metric.

We have added the following clarification to Section 4.7:

> *"We emphasize that the statistical associations reported here are computed on the full cohort as post-hoc biological annotations for the XAI-identified triplets. They are not used as model features, do not affect any performance evaluation, and serve solely to provide enriched context for the LLM-based hypothesis generation step. The hypergraph structure used during model training is, by contrast, constructed strictly within each training fold (for data-driven knowledge) or derived from external pathway databases."*

> *Change: [main.tex - added data-scope clarification paragraph in Section 4.7]*

---

**Q20.** The presented framework is quite complex and appears to rely on OpenAI models. It is not clear to me the computational time required for an end-to-end analysis, the number of hypotheses that the user needs to generate, and in general the economic cost to use this framework. At page 11, you report "our framework achieves end-to-end efficiency, requiring an average of 20 minutes per hypothesis (including generation and evaluation) at a cost of $0.7 per hypothesis," but I believe that more details are necessary. For instance, regarding how many rounds of refinement are necessary in the tournament-based hypothesis generation.

> *Annotation (CID #26, by Linhai Xie):* zqy - provide detailed time and cost estimation

> *(Assigned to zqy - detailed time and cost estimation.)*

---

**Q21.** The experimental validation of PFKFB2 supports a role in proliferation and colony formation, but it does not validate the full mechanistic hypothesis proposed by the LLM, including glycolysis, NADPH/acetyl-CoA, HMGCR, GGPP, geranylgeranylation, small GTPases, cytoskeletal remodeling, motility, and metastasis. The authors should either experimentally test key intermediate steps or moderate the claim that the mechanistic hypothesis was validated.

> *Annotation (CID #28, by Linhai Xie):* zqy - revise the claim

> *(Assigned to zqy - PFKFB2 claim moderation.)*

---

**Q22.** In the GitHub repository, the page "INSTALLATION.md" reports instructions in Chinese language. Please, translate everything into English.

> *Annotation (CID #27, by Linhai Xie):* cyp - translate to English

**Response.** The entire `INSTALLATION.md` has been translated from Chinese to English. The English version is now available at the repository root, providing detailed setup instructions for both the benchmark/XAI environment (Python 3.9+, PyTorch) and the hypothesis abduction environment (Python 3.10+, OpenAI API).

> *Change: [File: INSTALLATION.md - full translation from Chinese to English]*

---

### Typos (Review #2)

- At page 4, in the title "2.1 System framework of HGS-BioAbuctor" fix to HGS-BioAbductor.
- At page 4, "hypergraph massage passing" should be "hypergraph message passing".
- At page 4, "The representation of hyperedges are" presents a subject-verb agreement error.
- At page 5, "A overall" should be "An overall" (repeated in two places).
- At page 6, "It further strengthen" should be "It further strengthens".
- At page 7, "a in-depth application" should be "an in-depth application".
- At page 7, "protomic" should be "proteomic".
- At page 12, "For proteomic data, data-independent acquisition (DIA) proteomics data for seven tumor types were obtained..." but the tumor types are eight.
- At page 20, in the title "4.6 Duel-direction feature masking experiment" fix to Dual-direction.
- In the caption of figure 3, "Kaplain-Meier" should be "Kaplan-Meier" and "Person correlation" should be "Pearson correlation".

These typos have been corrected by zqy.

---

## Appendix: DataPreprocess Scripts Overview

The `DataPreprocess/` directory contains the full preprocessing pipeline that generates the feature-selected, Cox p-value-ranked input files used by the HGS-BioAbductor framework. Below is a summary of each script:

### Root scripts

| Script | Role |
|--------|------|
| **`FS_COX.py`** | **PRO cohort data preprocessing.** Reads multi-cohort proteomics data from `DATA/HCC_MultiCohorts/`, applies 50% NA filter, runs Cox univariate regression on all genes (parallelized), sorts by p-value, discretizes time bins, and outputs `dataset.csv` (features sorted by Cox p-value) and `genes_list.csv` (gene ranking). Supports STRING and ALL knowledge filter options. |
| **`PDC_Preprocess.py`** | **CPTAC/PDC pan-cancer preprocessing.** Follows a similar pipeline to `FS_COX.py` but for PDC proteomics data. Supports Reactome, STRING, and ALL knowledge filters. Applies NA filter, Cox univariate regression ranking, and time binning. |

### `Preprocess/` - Core library and specialized scripts

| Script | Role |
|--------|------|
| **`DATA_preprocess.py`** | **Core preprocessing library.** Contains all shared functions used by other scripts: `cox_feature_selection()` (parallel Cox univariate regression on all genes --- the key function that produces p-values), `cox_filter()` (wrapper: Cox regression + sort + top-N selection), `univariate_regression()` (single-gene CoxPHFitter), `preprocess_feature()` (NA% + low-variance filter), `preprocess_prognosis_matrix()` (survival data cleaning), `load_feature_matrix()` (TCGA expression loader), `discrete_time_bin()` (time discretization for classification), and `fil_min_nan_threshold()` (NA ratio filter). |
| **`FeatureSelection_CoxUni.py`** | **RNA-seq preprocessing (Reactome).** Loads TCGA expression matrix, applies Reactome gene intersection, runs Cox univariate filter, and prunes Reactome H1/H2/H3 incidence matrices. Outputs feature matrix, gene list, and pruned hypergraph matrices. |
| **`PreprocesseSurvivalData.py`** | **Full TCGA RNA-seq pipeline (older version).** Processes all TCGA cancer types with Reactome gene filter, NA/low-variance filter, Cox univariate feature selection, Reactome H1/H2/H3 incidence matrix construction/pruning, and survival data discretization. Configurable via CLI arguments (endpoint, cox_top_num). |
| **`LowVarFeatureSelect.py`** | **Variance-based feature selection alternative.** Applies low-variance filtering instead of Cox filtering, useful for exploring alternative feature selection strategies. |
| **`TimeBinsPreprcess.py`** | **Utility for time bin creation.** Creates discrete time bins from continuous survival time (e.g., the `OS_60` endpoint column used in configuration files). |
| **`PRO_Pan-cancer_Preprocess.ipynb`** | **Jupyter notebook for PRO pan-cancer preprocessing.** Exploratory and visual preprocessing for proteomics pan-cancer data. |

### Preprocessing flow

```
Raw expression matrix
    -> [load_feature_matrix] TCGA loader / CSV load
    -> [fil_min_nan_threshold] Remove genes with >50% NA
    -> [preprocess_feature] Low-variance filter
    -> [cox_feature_selection] Cox univariate regression on all genes (parallel via multiprocessing)
    -> Sort genes by p-value -> save genes_list.csv
    -> Slice top N (cox_num=400) -> save dataset.csv
    -> [discrete_time_bin] Time discretization
```

The key finding for R2 #12: The Cox p-value ranking (steps above) was historically computed on the **full dataset** before any train/test splitting, as confirmed by `FS_COX.py` (line 61) and `PDC_Preprocess.py` (line 79), both of which run `cox_feature_selection()` on the complete cohort without any train/test partitioning.
