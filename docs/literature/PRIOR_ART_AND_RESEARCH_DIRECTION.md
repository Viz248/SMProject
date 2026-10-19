# Prior art and proposed research direction

**Later evidence, 19 October 2026:** The uploaded full-text audit identified DIME (ICLR 2024) as a direct loss-improvement value-regression precursor. The proposed construction below is an adaptation, not an established novel mechanism. A three-seed development pilot on two real datasets did not establish its cross-fitting hypothesis. See [DIME_PILOT_RESULTS.md](../DIME_PILOT_RESULTS.md). The proposal below is retained as research history. A subsequent [matched DIME follow-up](../DIME_MATCHED_RESULTS.md) also failed to support the cross-fitting improvement; it is now a negative ablation rather than the planned primary method.

Search date: 19 October 2026. Status: expanded focused search and research proposal; no new method has yet demonstrated superiority. This supplements REVIEW.md rather than replacing its historical record.

## Answer to the novelty question

Our existing pipeline is useful experimental groundwork, but using an MLP, TabNet, SHAP, calibrated ensembles, explanation similarity or missing-feature tests is not by itself a new scientific contribution. DSI/CAS/RAI are currently summaries of established ideas. REVIEW.md already acknowledged this; it did not establish an original algorithm or a submission-ready paper.

An IEEE conference paper can extend an existing method or provide a rigorous empirical finding. It needs a specific, supported contribution and appropriate comparisons. Renaming metrics or adding architectures does not establish that contribution.

The outcome in our BRFSS project is self-reported **stroke history**. It cannot establish future stroke incidence, assessment of acute stroke severity, treatment benefit or prospective clinical utility.

## Search scope and reproducibility

Public web discovery followed by publisher/proceedings inspection. Search families included: stroke prediction with BRFSS, explanation stability, calibration, uncertainty, missing features, active feature acquisition, expected Brier improvement and proper-score acquisition. Searches also checked adjacent cardiovascular research and one patent family. This is a focused narrative search, not an exhaustive systematic review, patent-clearance search or proof that no similar method exists. Exact-title citation chasing and full-text extraction remain necessary before any priority claim.

Representative queries:

- `stroke prediction explainability uncertainty calibration feature acquisition BRFSS 2024 2025 2026`
- `stroke risk prediction explanation stability retraining uncertainty missing features`
- `active feature acquisition Brier score conditional mutual information missing not at random robust`
- `stroke prediction BRFSS missing features calibration external validation`
- `"stroke" "BRFSS" "feature acquisition"`
- `"stroke" "BRFSS" "explanation stability"`
- `site.patents.google.com stroke prediction uncertainty feature selection`

No-result or irrelevant-result searches are not evidence of novelty. The original 14-paper matrix remains in REVIEW.md; the table below selects the closest work and newly discovered challenges to our proposed contribution.

## Closest work and what it changes

| Work / primary source | Verified scope | Consequence for our project | Extraction status |
|---|---|---|---|
| [Ding et al., 2025](https://doi.org/10.3389/fphys.2025.1528910), Cross-sectional study on smoking types and stroke risk | Questionnaire stroke history; NHANES 2015–18 and BRFSS 2020–21; LASSO, ten classifiers including MLP, SHAP; random 80/20 division and five-fold tuning described in §§2.7.1–2.7.3. | Strong domain base. Its described protocol does not answer our sequential questionnaire acquisition question. We must not compare its headline scores with our different cohorts as evidence of superiority. | Relevant full-text methods, outcome and limitations inspected. Figures/supplements and exact reproduction remain pending. |
| [Moulaei et al., 2024](https://doi.org/10.1038/s41598-024-82931-5), Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models | Hospital-based ML/DL comparison with explanations; see original matrix. | ML versus DL plus SHAP is already covered. Different clinical sampling and predictors prevent direct score comparisons. | Prior review; not newly reproduced. |
| [IGStrokeNet, 2026](https://doi.org/10.1016/j.bspc.2026.110731), Biomedical Signal Processing and Control 125, 110731 | Publisher abstract describes nonlinear feature encoding, dynamic feature graphs and graph convolution; 5,110 records and five seeds. | Graph feature extraction is also prior art. A graph model alone would not resolve our research question. | Publisher abstract inspected through search; direct full text fetch failed. No unverified claims about missing experiments. |
| [Saleem et al., 2026](https://doi.org/10.1007/s40747-026-02540-9), Uncertainty-aware evidential deep learning for reliable stroke detection and clinical decision support | Accepted early article, published 9 October 2026; abstract describes Dirichlet evidential predictions, calibration and uncertainty comparisons. | Stroke-specific uncertainty estimation is established prior art. Its high scores cannot serve as our target without matching data and endpoint. | Publisher metadata/abstract inspected; PDF/data protocol not yet extracted. Dataset modality and leakage assessment remain unresolved. |
| [Covert et al., ICML 2023](https://proceedings.mlr.press/v202/covert23a.html), Learning to Maximize Mutual Information for Dynamic Feature Selection | Sequential selection from observed values; amortized predictor/policy objective. §4 and Appendix A connect cross-entropy reduction with conditional mutual information under optimal-predictor assumptions. Squared-error/conditional-variance results are also present. | Best methodological base for our acquisition question. Neither entropy-based acquisition nor switching to squared-error acquisition can honestly be claimed as a new principle. | Relevant full-text objective and proofs inspected; official implementation reproduction still required. Our conditional-donor pilot is not their algorithm. |
| [Valancius et al., ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html), Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition | Nongreedy acquisition method. | Required stronger comparator for any claim beyond one-step selection; greedy versus random alone is insufficient. | Proceedings record inspected; PDF fetch failed in this search. |
| [Kobayashi et al., ICML 2026](https://proceedings.mlr.press/v306/kobayashi26a.html), Learning-To-Measure: In-Context Active Feature Acquisition | Uncertainty-guided acquisition with retrospective missingness and limited labels, using autoregressive pretraining. | Missingness-aware uncertainty-guided acquisition is not an empty research area. Compare its relevance and feasibility before claiming an unaddressed gap. | Proceedings abstract inspected; full benchmark/code extraction pending. |
| [Cost-Aware Active Feature Acquisition for Differential Diagnosis under Realistic Clinical Availability Constraints, 2026](https://www.medrxiv.org/content/10.64898/2026.08.30.26361745v1.full) | Preprint: mask-aware MLP, conditional generation and cost-aware information acquisition. | Explicit availability masks, acquisition costs and clinical acquisition are already proposed together. This is not peer-reviewed evidence of effectiveness. | Method excerpt inspected; independent replication pending. |
| [Leakage-Safe Precision-Aware Dual-Branch FT-Transformer for Population-Scale Heart Disease Risk Prediction, 2026](https://www.mdpi.com/1424-8220/26/11/3417) | Adjacent heart-disease study using BRFSS 2024; discusses leakage, calibration and explanation variability. | Even the broad combination of BRFSS, deep learning, calibration and stable explanations has close adjacent prior art. | Relevant dataset/gap sections inspected; not a stroke-specific baseline. |

Discovery maps: [Aronsson et al., IJCAI 2026 survey](https://www.ijcai.org/proceedings/2026/860) organizes active acquisition methods; use its references for further primary-method comparisons. [CN118383715A](https://patents.google.com/patent/CN118383715A/en) concerns carotid-image geometry/flow uncertainty and stroke risk, materially different from questionnaire acquisition. This isolated patent screen establishes no legal novelty or freedom-to-operate conclusion.

## Selected base and a testable improvement

Use **Ding 2025 as the domain base** and **Covert 2023 as the methodological base**. Describe the work as an extension/application unless a genuinely distinct mechanism survives comparison with their algorithms and newer acquisition methods.

Research question: **Can an acquisition policy corrected using observed prediction losses improve calibrated stroke-history probabilities at a fixed questionnaire budget under changing feature availability?**

This is motivated by our own pilot, not an unsupported assertion that all prior papers failed: our entropy policy reduced entropy but did not establish a Brier advantage over random/global selection. Another pilot showed subset refitting could outperform masking a full model. These findings justify testing a predictor trained for partial observations and measuring realized utility rather than confidence alone. They do not establish a limitation of Covert's correctly implemented method.

Candidate extension, still a hypothesis:

1. Fit a predictor on simulated partial questionnaires, retaining observed-value masks and distinguishing naturally missing values from deliberately unasked questions. Fit preprocessing using training records only.
2. Generate acquisition episodes within development data. Obtain out-of-fold predictions so a record's label is never used to train the predictor that creates its realized loss-reduction target.
3. Train an action-value correction model to estimate `(p_before-y)^2 - (p_after-y)^2` for each candidate question. Inputs may contain observed values, mask, action identity, and training-derived uncertainty summaries. They must exclude the unasked value, hidden label and post-acquisition prediction for the current person.
4. Select questions using the estimated benefit, with a development-chosen stopping threshold. Treat a lower-confidence-bound penalty as an ablation, not an established guarantee. Predictive intervals for learned action values require separate validation.
5. Test whether this correction helps beyond ordinary loss-minimizing acquisition, partial-observation training and calibration. Any gain caused only by better predictor training is not evidence that the policy is better.

The candidate mechanism is a practical correction for finite-sample/misspecified acquisition estimates. Loss-based action values, stopping and uncertainty penalties have broad prior art; their combination is **not yet verified novel**. Search the correction mechanism specifically and inspect implementation overlap before making an algorithmic novelty claim.

## Required experiment and decision gates

| Element | Planned comparison / protection |
|---|---|
| Baselines | No acquisition; random; static ranking; current donor-entropy policy; reproduced Covert predictor/policy; loss-based policy without correction; availability-specific refit. Add ACO/L2M where code/data assumptions and compute permit; otherwise narrow the claims explicitly. |
| Fairness of comparison | Match predictors, initial observed variables, masks, budgets, calibration data and tuning budgets. Report both matched-predictor policy comparisons and complete-system comparisons. |
| Budgets | Start with sex and age; acquire 0–5 additional questions. Equal question costs are a declared simulation. Do not invent clinical monetary costs or assume a respondent can answer naturally missing fields. |
| Missingness | MCAR, observed-covariate-dependent masks and natural missingness evaluated separately. Simulated patterns cannot prove robustness to real MNAR mechanisms. Acquisition evaluation must report which rows/features are actually observable. |
| Primary outcome | Paired Brier difference versus reproduced Covert at three added questions, chosen before new evaluation. Positive benefit means baseline Brier minus candidate Brier. Also report the whole budget curve, AP, log loss, calibration and subgroup sensitivity. |
| Ablations | Mask training only; loss objective only; correction only; stopping only; uncertainty penalty; survey-weighted versus unweighted development. Include actual compute/memory cost. |
| Statistical reporting | Multiple training seeds; paired intervals; survey strata/PSUs and weights for population claims. IID intervals alone describe conditional record-level uncertainty, not the BRFSS sampling design. |
| Data discipline | 2024 final test and 2023 transport scores have already been inspected. They are historical/exploratory evidence for this new direction, not fresh confirmation. Continue development only within development partitions. |
| Fresh confirmation | CDC now lists [BRFSS 2025](https://www.cdc.gov/brfss/annual_data/annual_2025.html). Check variable definitions and geographical coverage before committing to it. Freeze algorithm, costs, primary outcome and analysis before examining its predictive results. A later survey year tests transport of stroke-history classification, not incident stroke prediction. |
| Success gate | Require practically useful primary improvement with uncertainty reported, replication across seeds, and no unexplained loss in calibration or relevant subgroups. A positive development result alone is insufficient. Preserve null results and do not switch the primary endpoint after testing. |

## Conference-paper plan

Working title: **Evaluating and Improving Adaptive Questionnaire Acquisition for Calibrated Stroke-History Classification**. This is provisional and must reflect the eventual evidence.

Build the paper around one question and one tested extension. Introduction: questionnaire burden and prediction reliability. Related work: survey stroke history versus incident prediction, then acquisition methods. Methods: endpoint, data provenance, split discipline, reproduced base, correction and assumptions. Results: budget curves, primary paired comparison, missingness/transport tests, ablations and compute. Discussion: why gains occurred or failed, self-report/reverse-causality limits, sampling design and lack of clinical validation. Supply reproducible code and an explicit record of all inspected evaluations.

Current deliverable is the search and experiment specification. The extension is not yet implemented or validated; no improved result or IEEE acceptance is claimed. REVIEW.md provides the earlier literature matrix; this document gives the proposed route from that matrix to a falsifiable contribution.
