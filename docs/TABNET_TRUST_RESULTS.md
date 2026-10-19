# TabNet and trust-framework research milestone

All numbers below are from executed experiments on public CDC BRFSS data. This report separates exploratory DEVELOPMENT analysis from frozen survey-year transport evaluation. The existing 2024 final-test report remains unchanged and is not reused for tuning. No clinical recommendation, new-metric novelty or conference acceptance claim is supported.

## TabNet optimization

Tested on the full 250,919-record fitting partition, 45,622 stopping records and 68,433 selection records from BRFSS 2024. Categorical vocabularies and numeric scaling are fitted only on fitting records; missing/unseen categories use a reserved code. Numeric missing indicators are grouped with their source variable in attention. AdamW, initial learning rate .01, StepLR decay .8 every five epochs, batch 4096, virtual batch 256, categorical embeddings of size 2, seed 42, CPU threads 2. Early stopping uses stopping average precision with patience 8, maximum 30 epochs. Three configurations were declared before the run; this is a bounded search, not exhaustive optimization.

| Configuration | Width n_d=n_a | Steps | Sparsity penalty | Weight decay | Best epoch (zero-based) | Selection AP | Selection Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| tabnet8 | 8 | 3 | 0.001 | 0.0001 | 7 | 0.165868 | 0.040364 |
| tabnet16 | 16 | 3 | 0.001 | 0.001 | 29 | 0.167201 | 0.040192 |
| tabnet16_sparse | 16 | 4 | 0.01 | 0.001 | 24 | 0.167540 | 0.040229 |

Best TabNet selection AP was .167540 versus .193836 for the previously selected MLP/tree blend. The blend is retained. These are shared selection records, not final-test or independent confirmation scores. One configuration was still improving at the budget cap; no claim is made that TabNet cannot do better with a larger search or different training regime. This result does not support replacing the current model with TabNet.

TabNet custom-metric callbacks exposed a checkpoint portability failure, and the initial production artifact could not be loaded for the external comparison. A native network/configuration checkpoint embedded inside the predictor replaces pickling training callbacks. A fresh-process reload test now covers this path. The full-data same-setting refit reproduced selection AP to within 1e-8, and the recovered native checkpoint passed a prediction roundtrip. The external comparison was then completed in a new process. No hyperparameters were changed after the partially completed first external attempt; the failure record and frozen protocols are preserved.

## Trust pilot model and cohort

The trust pilot uses FIVE calibrated direct MLPs (128→64→32, alpha=.1, Adam, at most 20 epochs), NOT the frozen tree/MLP blend or TabNet. Three members use identical fitting records with seeds 42–44; two use respondent-bootstrap fitting samples with seeds 45–46. Fitting/stopping/calibration partitions remain shared. The ensemble averages individually calibrated probabilities. Metrics are measured on a fixed natural-prevalence sample of 2,048 DEVELOPMENT records, reused from earlier model development. This is component validation, not an end-to-end externally validated final framework.

## DSI and estimator-noise control

Probability-scale interventional permutation Shapley explanations use the original 15 survey variables, 2,048 fitting-background records and 128 matched explanation records. Common random paths are used between retrained models. Independent-path repeat estimates for the same model diagnose Monte Carlo noise. Numerical probe differences are not causal or longitudinal effects.

| Same-model explanation budget | Mean repeat DSI | IID respondent 95% interval |
|---|---:|---|
| 16 permutations | 0.736453 | 0.721187–0.750938 |
| 64 permutations | 0.853826 | 0.844496–0.862425 |
| 256 permutations | 0.927764 | 0.923155–0.932162 |

The 16-permutation same-model repeat score (.736) was lower than the matched-path seed-retraining score (.870), showing why a low-budget score alone cannot be interpreted as model instability. Increasing to 256 permutations improved repeat DSI to .928; this does not prove convergence. Do not subtract these scores to obtain a noise-corrected stability measure.

| 256-permutation comparison | DSI | Absolute-rank Spearman | Top-5 Jaccard |
|---|---:|---:|---:|
| Three seed pairs | 0.890181 | 0.825958 | 0.737537 |
| Reference versus two bootstraps | 0.855687 | 0.797405 | 0.695219 |

The score ranges and meanings differ across columns; larger numerical values across different metrics are not a superiority comparison. Confidence intervals average model pairs within records before bootstrapping records and do not represent uncertainty over all possible retrainings. Fidelity and estimator convergence remain open beyond a local-accuracy/telescoping implementation check.

## CAS uncertainty summaries

CAS reports a vector of predictive entropy, mean member entropy and their Jensen-gap disagreement. It does not identify unique causal epistemic/aleatoric components.

| Summary / error detector | Observed value |
|---|---:|
| Mean predictive_entropy | 0.215213 |
| Mean mean_member_entropy | 0.214682 |
| Mean ensemble_disagreement | 0.000531 |
| Mean entropy_confidence | 0.784787 |
| Mean mean_probability | 0.043279 |
| entropy_error_detection_auroc | 0.936487 |
| disagreement_error_detection_auroc | 0.830233 |

Here conventional predictive entropy detected errors better than ensemble disagreement. This does NOT establish incremental benefit from the CAS label. Error detection uses the fixed .14 research threshold; no error-prediction model was fitted on this cohort.

## RAI fixed-model availability

RAI summarizes retained Brier headroom above the fitting-prevalence constant predictor, with equal weights across seven declared scenarios. It is not a probability of trustworthiness. Raw losses accompany clipping/normalization. These are fixed calibrated MLP ensemble masks with training-derived imputation.

Observed RAI: 0.717437. Full-input Brier: 0.030689; null-reference Brier: 0.033135.

| Scenario | Brier | Average precision | Mean entropy change |
|---|---:|---:|---:|
| demographics_only | 0.032250 | 0.079013 | 0.006520 |
| demographics_history | 0.031769 | 0.158505 | 0.077486 |
| without_health_state | 0.031627 | 0.167610 | 0.081953 |
| mcar_0.1 | 0.030858 | 0.194582 | 0.003393 |
| mcar_0.3 | 0.030701 | 0.164728 | 0.012245 |
| mcar_0.5 | 0.031636 | 0.130811 | 0.016938 |
| age_associated_bmi_missing | 0.030819 | 0.185301 | -0.007750 |

Simulation seeds, patterns and scalar normalization are implementation choices, not estimated deployment frequencies or clinical missingness behavior. The three-pattern MCAR curve need not be monotonic on one finite cohort/masking draw.

### Separate subset-retraining ablation

This comparison uses a DIFFERENT predictor family: histogram boosted trees with identical declared settings across feature sets, fitted, stopped and calibrated on the same partitions. Compare fixed masks against subset retraining WITHIN this table; do not treat its scores as a direct effect on the MLP-based RAI above.

| Feature set | Retrained AP | Fixed-full-model masked AP | Retrained Brier | Fixed-mask Brier |
|---|---:|---:|---:|---:|
| full | 0.191828 | 0.191828 | 0.030515 | 0.030515 |
| demographics | 0.082968 | 0.083301 | 0.032300 | 0.032337 |
| demographics_history | 0.183429 | 0.159560 | 0.030886 | 0.030900 |
| without_health_state | 0.206757 | 0.169087 | 0.030391 | 0.031201 |

Excluding current health-state items while retraining produced higher AP in this 2,048-record pilot than the full-input version. This is exploratory, not a proven optimization; no new final model was selected using this small cohort. Removing those variables does not turn a history label into future risk.

## Conditional acquisition prototype

Uses one calibrated MLP reference, an 8,192-record complete FITTING-donor pool, 64 nearest donors, initially known sex and age group, and three additional questionnaire items. Evaluation is on 256 complete DEVELOPMENT records. Global importance uses training-record explanations; hidden held-out feature values are read only AFTER the policy chooses a feature. All strategies use the same donor-completion predictor. No clinical measurement costs are invented.

| Policy | Mean Brier reduction after 3 items | IID paired-record 95% interval | Mean entropy reduction |
|---|---:|---|---:|
| conditional_information | -0.000714 | -0.002683–0.001573 | 0.033701 |
| random | -0.000335 | -0.001752–0.001503 | 0.026311 |
| global_importance | -0.000704 | -0.002808–0.001599 | 0.031164 |

Positive Brier reduction would be improvement. All three point estimates were negative and intervals included zero: reducing entropy did not establish better predictions. The conditional policy did not establish an advantage over random or global acquisition. This is a negative result that must remain visible; the recommendation engine is a research prototype, not validated advice.

## Frozen 2023 survey-year transport comparison

Public CDC BRFSS 2023: 433,323 records, 431,849 known stroke-history labels, 18,350 positives. The 15 original predictors and their response coding were inspected in the 2023 codebook and matched the cleaner. All transformations, model weights, sigmoid calibration and the .14 threshold remain fixed from 2024 development. The previously selected blend was declared the retained model BEFORE computing external model scores.

| Model | External AUROC | External average precision | External Brier | Sensitivity at .14 | Specificity at .14 |
|---|---:|---:|---:|---:|---:|
| original_blend | 0.815018 | 0.177416 | 0.037576 | 0.319728 | 0.944239 |
| tabnet | 0.795426 | 0.153886 | 0.038115 | 0.272807 | 0.950800 |

Paired Brier advantage of TabNet over blend (blend loss minus TabNet loss): -0.000539, IID respondent bootstrap 95% interval -0.000598–-0.000476, 500 replicates. Negative means TabNet has higher loss. This is conditional on the frozen models and is NOT a design-based survey interval.

TabNet also underperformed in this frozen year-transport comparison, supporting retention of the blend under the tested configurations. The external outcomes will not be used for further model/hyperparameter selection. The same comparison was retried after the serialization failure; it is not a second independent cohort.

2023 is EARLIER than training year 2024. This checks transport across separately sampled survey years, not future forecasting, longitudinal patient monitoring, incident stroke, or external clinical validation. Participating-state coverage differs; anonymous respondent overlap cannot be verified from these data. Survey-weighted point metrics are in the JSON, but no survey-design confidence intervals were computed. The trust layer itself has NOT been validated on 2023.

## Validation status and next gates

Twenty-one unit tests pass, including native TabNet checkpoint loading in a fresh process and exclusion of removed features in subset retraining. Synthetic fixtures are only software tests; the tables above use actual CDC survey records. A 14-publication literature matrix and provisional mathematical protocol are provided separately.

No claim of novel metrics or publication readiness is supported yet. Needed next: full-text extraction for remaining literature fields, stronger acquisition/uncertainty baselines, larger matched samples and explanation budgets, subgroup and design-aware intervals, independent trust-layer validation, and a clear contribution beyond conventional similarity/entropy/degradation summaries. The target conference has not been selected.
