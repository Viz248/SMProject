# Optimization results — 2026-10-19

This report supersedes the initial synthetic-only status for the new optimization experiments. The original audit is retained as a historical snapshot.

## Data and execution

Google Drive metadata resolved the selected LLCP2024.XPT file, but its 1,093,874,240 bytes exceeded the connector download limit of 268,435,456 bytes. Experiments instead use the independently downloaded official CDC 2024 XPT release. Its byte size matches, but identity to the selected Drive copy is unverified. Hashes and exact sources are in DATA_PROVENANCE.json.

All 457,670 records were read in chunks. The analysis retains 456,218 known labels (20,661 positive; 435,557 negative), excluding 1,452 unknown labels. The original 15 predictors were retained; no unreviewed extra variables or target proxies were added. Survey design fields remain separate.

## Search protocol

Fit/stopping/selection/calibration/test counts: 250,919 / 45,622 / 68,433 / 45,622 / 45,622. Seed 42. Full fitting partition, no training subsample. The direct MLPs use Adam, ReLU, batch 512, initial learning rate .001 and at most 30 epochs with stopping-set average precision and patience 8. Regularization values and widths are recorded in development.json. The bottleneck remains a comparator, rather than being assumed the best representation.

Each candidate fits transformations only on fitting records. L1 selects encoded input dimensions (not whole clinical variables). Boosting uses explicit stopping data and log-loss early stopping. The selection criterion is average precision, followed by Brier for exact ties. This is a finite exploratory search with unequal candidate counts and training budgets across families; it is not an equal-budget benchmark.

After the 11-candidate search, the strongest direct MLP was refitted with the same settings and combined with the selected boosted model using neural weights .25, .50 and .75. This additional search stage reused selection data and is explicitly exploratory. No test-based tuning occurred.

## Selection results — not final-test scores

| Candidate | Average precision | Brier | Encoded dimensions before bottleneck |
|---|---:|---:|---:|
| logistic_C0.1 | 0.186749 | 0.039878 | 61 |
| logistic_C1 | 0.186568 | 0.039876 | 61 |
| logistic_balanced | 0.186048 | 0.179401 | 61 |
| l1_selected_logistic | 0.186835 | 0.039875 | 48 |
| boosted_15 | 0.192650 | 0.039627 | 61 |
| boosted_31 | 0.193097 | 0.039614 | 61 |
| mlp_shallow | 0.188918 | 0.039767 | 61 |
| mlp_deep | 0.186391 | 0.039780 | 61 |
| mlp_regularized | 0.191504 | 0.039696 | 61 |
| mlp_balanced | 0.190327 | 0.177789 | 61 |
| bottleneck15_logistic | 0.189487 | 0.039725 | 61 |
| Blend, neural weight 0.25 | 0.193836 | 0.039601 | 61 |
| Blend, neural weight 0.5 | 0.193784 | 0.039611 | 61 |
| Blend, neural weight 0.75 | 0.192996 | 0.039642 | 61 |

The selected predictor is 75% histogram gradient boosting plus 25% regularized MLP (128→64→32), followed by sigmoid calibration. L1 retained 48 of 61 encoded dimensions without a material validation gain. Adding neural depth alone was not an improvement. The blend gain over the best tree on selection data was approximately 0.00074 AP; no statistical significance is claimed.

## Frozen one-time internal test

Only the selected calibrated model was evaluated on the locked test. Threshold .14 was selected by F1 on selection data after fitting the sigmoid on the separate calibration partition. The classifier, blend and threshold were not changed after test access.

| Metric | Unweighted test | IID respondent 95% percentile CI |
|---|---:|---|
| auroc | 0.813497 | 0.805072–0.821502 |
| average_precision | 0.179478 | 0.166447–0.194397 |
| brier | 0.039928 | 0.038405–0.041390 |
| sensitivity | 0.332527 | 0.310778–0.351426 |
| specificity | 0.938746 | 0.936337–0.940837 |
| f1 | 0.253459 | 0.237418–0.268315 |
| Precision | 0.204769 | Not computed |
| Equal-width 10-bin ECE | 0.001716 | Not computed |

Confusion matrix: TN 40,888; FP 2,668; FN 1,379; TP 687. Test prevalence 4.5285%. Always-negative accuracy would already be 95.47%, illustrating why accuracy is not the optimization objective.

These 500-replicate intervals condition on the fitted model and assume IID respondents. They are not design-based survey intervals or estimates of retraining variability. The approximately 33.3% sensitivity at this F1-selected threshold means many positive histories are missed. These findings do not establish suitability for clinical screening. Low coarse-bin ECE alone does not establish reliable calibration, especially in sparse high-probability bins.

### Survey-weighted sensitivity analysis

| Metric | Weighted point estimate |
|---|---:|
| prevalence | 0.033316 |
| auroc | 0.838477 |
| average_precision | 0.180301 |
| brier | 0.029591 |
| sensitivity | 0.328674 |
| specificity | 0.952954 |
| precision | 0.194050 |

Weights are used only for these test point estimates; model fitting/calibration remain unweighted. Strata and PSU are preserved but no design-aware variance estimation is implemented. Weighted and unweighted scores target different distributions; do not choose whichever looks better.

## Validation and boundaries

Twelve unit tests pass. The original synthetic smoke tests remain labeled artificial. The new full-data search and locked evaluation executed successfully; the fitted predictor was loaded in a separate process for the final evaluation, verifying serialization. Notebook JSON and code-cell syntax were checked, but Colab/Drive execution itself was not tested.

The unchanged original pipeline also completed (exit 0). Its MLP stopped after 13 epochs, extracting 15 latent features, on its original 364,974/91,244 split. Saved probability arrays were independently checked against the results CSV for AUROC, AP and F1 and matched for all four downstream classifiers.

| Original downstream model | Original test AUROC | Original test AP | Original default-threshold sensitivity |
|---|---:|---:|---:|
| Logistic regression | .814069 | .182583 | .761617 |
| Decision tree | .785161 | .166291 | .760649 |
| Random forest | .812022 | .179671 | .713698 |
| Gradient boosting | .813746 | .183066 | .001936 |

These are different test records and different training/threshold policies from the new optimizer. The new method has NOT established improvement over the original pipeline. In particular, the original gradient boosting default threshold almost never predicts a positive history, while balanced logistic regression trades many false positives for higher sensitivity. Proper matched-cohort, matched-calibration and matched-threshold comparisons remain necessary. Exact aggregates and verification evidence are in experiments/original_classification.csv and experiments/original_reproduction.json.

## Next scientific priorities

Freeze a repeated-training protocol with a shared held-out cohort; compare calibrated tree, MLP and blend under matched budgets; quantify paired differences; run consequence-sensitive variable ablations; add survey-aware intervals and subgroup support checks. Implement DSI/CAS/RAI and conditional feature acquisition with their own baselines and failure-detection outcomes. These components remain unimplemented, so the framework is not yet conference-submission-ready.

Figures and machine-readable results: experiments/selection_comparison.pdf, experiments/calibration.pdf, experiments/development.json and experiments/locked_test.json. No raw records or fitted models are included in these tracked artifacts.
