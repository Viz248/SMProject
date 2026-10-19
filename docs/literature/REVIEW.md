# Focused research literature and gap matrix

**Subsequent correction:** DIME (Gadgil, Covert & Lee, ICLR 2024), [official paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9682490bedc064aba8aac1ab3f703507-Abstract-Conference.html), directly learns a value network from incremental prediction-loss improvements. It is the closest methodological base for the proposed Brier adaptation. The original 14-record matrix missed it. The new development pilot did not establish a cross-fitting advantage; see [DIME_PILOT_RESULTS.md](../DIME_PILOT_RESULTS.md).

Expanded search and concrete base-paper/experiment proposal (19 October 2026): [PRIOR_ART_AND_RESEARCH_DIRECTION.md](PRIOR_ART_AND_RESEARCH_DIRECTION.md). The newer document adds close acquisition and stroke-uncertainty work and supersedes any interpretation of this matrix as a completed novelty assessment.

Fourteen verified publication records: eleven from 2023–2026 and three older foundational papers. This is a focused narrative review, not a systematic search or a completed full-text extraction. Unknown dataset/protocol details are marked pending, not guessed. Limitations labeled as our assessment are inferences from the reviewed scope. No absence claim is made from a paper abstract.

## Domain base paper

Ding et al. (2025), DOI 10.3389/fphys.2025.1528910, is the primary DOMAIN comparison because it uses BRFSS questionnaire stroke-history data, conventional models and SHAP. It is not an exact reproduced algorithm baseline. Meyer et al. (2023) and Covert et al. (2023) are methodological anchors for stability and acquisition respectively.

## Publication records

| Key | Year / venue | Title and verified source | Scope / relevance |
|---|---|---|---|
| ding2025 | 2025 / Frontiers in Physiology 16 | [Cross-sectional study on smoking types and stroke risk: development of a predictive model for identifying stroke risk](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1528910/full) | Primary domain comparison: survey stroke history, conventional models and explanations |
| moulaei2024 | 2024 / Scientific Reports 14:31392 | [Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models](https://www.nature.com/articles/s41598-024-82931-5) | Stroke ML/DL and XAI comparator; calibration itself is not our novelty |
| zhi2024 | 2024 / Frontiers in Neurology 15 | [An exploration on the machine-learning-based stroke prediction model](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1372431/full) | Shows why very high hospital-study scores cannot be used as BRFSS targets |
| melnykova2025 | 2025 / Scientific Reports 15:33773 | [Machine learning for stroke prediction using imbalanced data](https://www.nature.com/articles/s41598-025-01855-w) | Class imbalance and failure of accuracy-only optimization |
| meyer2023 | 2023 / UAI, PMLR 216:1434–1444 | [On Minimizing the Impact of Dataset Shifts on Actionable Explanations](https://proceedings.mlr.press/v216/meyer23a.html) | Direct prior art for DSI; stability across retraining is not a new concept |
| chen2024 | 2024 / NeurIPS 37 | [Training for Stable Explanation for Free](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0626822954674a06ccd9c234e3f0d572-Abstract-Conference.html) | Important DSI comparator; new score must beat existing top-k/rank summaries |
| covert2023 | 2023 / ICML, PMLR 202:6424–6447 | [Learning to Maximize Mutual Information for Dynamic Feature Selection](https://proceedings.mlr.press/v202/covert23a.html) | Methodological base for acquisition; our donor approximation is not their algorithm |
| valancius2024 | 2024 / ICML, PMLR 235:48957–48975 | [Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition](https://proceedings.mlr.press/v235/valancius24a.html) | Stronger acquisition baseline and prior art for expected utility |
| gorla2025 | 2025 / ICML, PMLR 267:20187–20225 | [CACTI: Leveraging Copy Masking and Contextual Information to Improve Tabular Data Imputation](https://proceedings.mlr.press/v267/gorla25a.html) | RAI missingness-aware training/imputation comparator |
| hasny2026 | 2026 / ICML, PMLR 306:40772–40790 | [No Data? No Problem: Robust Vision-Tabular Learning with Missing Values](https://proceedings.mlr.press/v306/hasny26a.html) | Resource robustness already has substantial prior art |
| holzmuller2024 | 2024 / NeurIPS 37 | [Better by default: Strong pre-tuned MLPs and boosted trees on tabular data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html) | Supports retaining MLP and tree comparisons rather than assuming TabNet superiority |
| arik2021 | 2021 / AAAI 35(8):6679–6687 | [TabNet: Attentive Interpretable Tabular Learning](https://ojs.aaai.org/index.php/AAAI/article/view/16826) | Foundational architecture reference; using TabNet is not itself novelty |
| lakshminarayanan2017 | 2017 / NeurIPS 30 | [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Foundational CAS prior art; distinguish empirical disagreement from proved epistemic uncertainty |
| lundberg2017 | 2017 / NeurIPS 30 | [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/7062-a-unified-approach-tointerpreting-model-predictions.pdf) | Foundational attribution definition and essential interpretation constraints |

Detailed authors, datasets, methods, evaluation, limitations and extraction status are in [references.json](references.json).

## Research-gap matrix

| Component | Established prior art | What this project has now implemented | Evidence still required |
|---|---|---|---|
| DSI | Meyer 2023, Chen 2024; signed/rank/top-k similarity | Original-variable permutation Shapley; seeds/bootstrap/perturbation; same-model MC noise control | Advantage over constituent similarity metrics; estimator convergence and independent failure detection |
| CAS | Deep ensembles and entropy decomposition | Calibrated member entropy and disagreement with error associations | Added value over ordinary entropy; no uniquely causal attribution claim |
| RAI | CACTI 2025, RoVTL 2026 and missingness robustness literature | Fixed-model masking scenarios and Brier-headroom retention | Subset retraining, realistic deployment frequencies, design-aware intervals and incremental usefulness |
| Acquisition | Covert 2023, Valancius 2024 | Training-only conditional donors versus random/global ranking | Larger complete-case/missingness studies, stronger baselines, realized proper-score benefit and defensible costs |
| Predictive models | TabNet 2021, RealMLP/trees 2024, stroke ML comparisons | Controlled MLP/tree/bottleneck search; bounded TabNet search | Matched tuning budgets and fresh frozen transport evaluation |

**Novelty conclusion:** no verified basis for describing DSI/CAS/RAI names as new scientific concepts. A defensible potential contribution is a reproducible survey-history trust benchmark with explicit estimator-noise controls, calibrated uncertainty, feature-availability analysis and acquisition validation. Whether that is sufficient for a particular conference remains undecided.
