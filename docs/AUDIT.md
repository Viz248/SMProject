# Technical audit and research plan

Audit date: 2026-10-19. Fork commit inspected: 095ae4d367d1c34000e1c235e9f4a93596c99791.
Branch: research/trust-baseline. All seven project files and the existing .gitignore were read. Original scripts are unchanged.
No original baseline results or raw dataset were present. No original BRFSS experiment has been reproduced.

## Existing implementation

| File | Observed behavior | Assessment |
|---|---|---|
| preprocessing.py | Reads entire XPT; selects 15 predictors and CVDSTRK3; deterministic recodes; drops unknown labels; stratified 80/20 split; train mode/median imputation; train numerical scaling; exports CSV | Correct outer train/test isolation for fitted preprocessing. Does not persist imputation/scaler or row IDs. Survey design discarded. Codebook verification incomplete. |
| neural_feature_learning.py | One-hot categories, numeric passthrough; supervised ReLU MLP 64→32→15→16→binary output; Adam, alpha 1e-4, batch 1024, LR .001, max 80 epochs, internal 10% early stopping, patience 8, seed 42; extracts third hidden activation; standardizes latent features | Supervised representation learning, not original-variable feature selection. Fits encoder on outer training only. Saves encoder, MLP, latent scaler. Manual forward pass matches current ReLU architecture. |
| classification_neural.py | Logistic regression (balanced, max_iter 1000), tree (balanced, depth 10, leaf 20), RF (200 trees, balanced, depth 12, leaf 10), GB (150 stages, LR .05, depth 3); defaults otherwise; evaluates all on test | Reports accuracy, precision, recall, F1, AUROC and average precision; confusion counts. No validation-based selection, tuning, probability calibration, CIs, specificity summary or thresholds study. Saves test predictions, not downstream fitted models. |
| visualizations_neural.py | Loads saved predictions; dynamic confusion, ROC, PR and comparison plots | Useful foundations; show() only, no durable figures, calibration plots or CIs. |
| run_neural_pipeline.py | Sequential subprocess stages, stops on failure | Uses caller working directory; import executes work. Final instruction names nonexistent visualizations.py rather than visualizations_neural.py. |
| requirements.txt | Seven unpinned dependencies | Algorithm changes with installed sklearn: MLP sample weights supported in current environment; fallback changes training distribution. |
| neural_feature_names.csv | nn_feature_01 through nn_feature_15 | Latent labels, not clinical feature metadata. |

## Reproducibility and leakage assessment

- No direct outer-test preprocessing leakage found in static inspection. Labels are removed from predictors before training. This does not establish execution correctness on real data.
- The supervised MLP uses all outer-training labels before fitting downstream models. This is legitimate on that training set, but invalid for downstream cross-validation if the precomputed latent matrix is reused across folds. Refit the entire encoder/MLP/latent-scaler within each fold.
- Imputation/scaling and encoding use records that later become the MLP's internal early-stopping validation subset. This weakens strict validation isolation, though it does not use outer test data.
- Early stopping optimizes classifier score (accuracy by default), poorly matched to rare-label goals. Explicit validation with AUPRC/proper scoring should be evaluated before adopting the bottleneck.
- Training on a label-informed representation is not leakage by itself. Comparing multiple models on test and selecting winners for further iteration would contaminate model selection. Current code prints maxima but does not explicitly select a winner.
- Balanced weights and the old-sklearn fallback (up to four negatives per positive) are different experiments. The fallback is not a 1:1 balanced dataset and changes prevalence. Probability calibration must use natural-prevalence held-out records.
- General health and physical-health days may reflect consequences of a past stroke. They are legitimate contemporaneous correlates but cannot support a future-risk or causal claim. Audit measurement timing and evaluate an ablation excluding consequence-sensitive variables.
- Latent neurons may permute or rescale across retraining. DSI must compare explanations in the same original input-variable space, for the same class, output scale, background/reference policy and matched records. Do not interpret latent-neuron SHAP as clinical-variable importance.
- Single-year cross-sectional internal splits cannot establish longitudinal reliability, prospective stroke risk, external clinical validity, or stability over time.
- Survey weights and class-balance weights serve different purposes. CDC specifies _LLCPWT, _STSTR and _PSU for core-data complex survey analyses. Preserve them separately; never let survey IDs become predictors. Population claims require weighted sensitivity analyses and design-aware uncertainty. The new IID bootstrap is conditional on the fitted model and is not a survey-design CI.

## Retain versus modify

Retain the original deterministic recodes provisionally, outer split principle, one-hot encoding, supervised bottleneck as a comparator, classifier set, and dynamic plotting. Before claiming valid BRFSS results, verify all 2024 response codes and variable availability against the full official codebook. Validate, rather than assume, an advantage of the 15-neuron bottleneck against raw-variable logistic regression, direct MLP and trees.

## First implemented changes

A parallel truststroke baseline leaves legacy files intact. It provides an explicit 60/20/20 stratified split, train-only fitted preprocessing in a serializable pipeline, original-variable logistic/RF/MLP models, validation-only F1 threshold selection, prevalence, AUROC, average precision (not trapezoidal PR area), specificity, Brier score, equal-width ECE and calibration-bin data, and IID respondent bootstrap percentile intervals. A run records seeds, package versions, input hash, split positions and a fitted end-to-end model. Existing run directories cannot be overwritten. Model and record-level artifacts are ignored by git.

The first baseline deliberately uses unweighted fitting, with no implicit resampling, hyperparameter search or fitted calibration. Those are future controlled comparisons. F1 threshold optimization is an experimental policy, not a clinical utility function. The test threshold is frozen from validation.

## Verified execution

Five unittest tests passed. Logistic synthetic smoke: n=1000, split 600/200/200, seed 42, 50 bootstrap replicates. Artificial test AUROC 0.69739; AP 0.22823; Brier 0.11005; ECE 0.04422. These are SOFTWARE CHECKS ONLY and cannot appear as BRFSS results. Raw figures are in results/smoke/metrics.json. RF and direct MLP smoke verification are recorded separately in the research log. No real-data results are available.

## Prioritized implementation sequence and evidence gates

1. Obtain LLCP2024.XPT; verify official 2024 codebook; inventory missingness and design fields; preserve original pipeline results and environment before methodological changes. Record dataset hash and exclusion counts. Current blocker: dataset unavailable.
2. Run baseline on natural-prevalence splits; keep development partition distinct from locked test; compare raw inputs, bottleneck and direct MLP with equal tuning budgets. Fit all preprocessing inside CV folds. Add calibration on separate development folds and compare class weighting. Preserve original comparator.
3. Add design-aware weighted evaluation and population subgroup analysis. Quantify support, CIs and small-cell uncertainty. Do not treat repeated seeds as new cohorts.
4. DSI: pre-register comparison protocol using original-variable attribution vectors; test normalized signed L1 similarity, rank correlation and top-k overlap separately before choosing aggregation. Compare seeds, bootstraps and plausible input changes separately. Decide zero-vector treatment explicitly; identical zeros need an informativeness flag. Compare fidelity and stability together because constant explanations are stable but useless.
5. CAS: start with ensemble probability means, predictive entropy, mean member entropy and their difference. That difference reflects disagreement under the chosen ensemble, not uniquely identified clinical epistemic uncertainty. Validate Brier/calibration and error ranking. Missing-feature sensitivity is an intervention-specific quantity, not an additive causal uncertainty share. Avoid naming arbitrary normalized percentages causal contributions.
6. RAI: report performance and calibration change across pre-specified supported feature scenarios. Separate subset retraining from fixed-model masking; fit imputers only on training. Compare MCAR with declared informative masking. Start with scenario-wise degradation curves, not an arbitrary composite index; include prevalence reference and worse-than-reference cases if normalizing scores.
7. Acquisition: estimate conditional expected scoring/uncertainty change given currently observed variables using training-only conditional models. Keep hidden feature values unavailable until acquisition. Compare random and global-importance policies on identical held-out complete records and masks. Report scoring loss and calibration as well as entropy; reduced entropy can mean overconfidence. No invented clinical costs or clinical recommendations.
8. Run multi-seed/ablation comparisons with matched evaluation records, paired effect sizes and design-appropriate CIs. Only then finalize metric definitions and manuscript claims.

## Research-gap assessment (preliminary, not systematic review)

| Proposal | Existing evidence | Defensible test of added value |
|---|---|---|
| DSI | Meyer et al., UAI 2023, study explanation stability under dataset shift | Incremental error/failure detection over signed similarity, rank and top-k metrics on BRFSS; no novelty claim for stability itself |
| CAS | Lakshminarayanan et al., NeurIPS 2017, deep ensembles for predictive uncertainty | Calibrated predictive usefulness and defensible uncertainty summaries; unique causal decomposition is not established |
| RAI | Feature acquisition and dynamic feature selection already evaluate changing feature access | Explicit separation of retraining and fixed-model masking; compare simple degradation summaries before proposing a new scalar |
| Acquisition | Covert et al., ICML 2023, conditional mutual information; Valancius et al., ICML 2024, nongreedy acquisition | Conditional per-person utility and held-out realized improvements beyond random/global ranking |

A universal statement that no prior stroke model evaluates trustworthiness is unsupported. No stroke-specific base paper selected yet. The complete requested 10–15-paper extraction and novelty review remain open. Current evidence supports an integrated evaluation hypothesis, not four proven novel inventions.

## Verified source starting points

- CDC 2024 data/documentation: https://www.cdc.gov/brfss/annual_data/annual_2024.html
- CDC 2024 complex sampling: https://www.cdc.gov/brfss/annual_data/2024/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf
- CDC stroke indicator interpretation: https://www.cdc.gov/dhds/data-guides/chronic-conditions.html (not a replacement for the complete 2024 codebook).
- Anna P. Meyer, Dan Ley, Suraj Srinivas, Himabindu Lakkaraju (2023). On Minimizing the Impact of Dataset Shifts on Actionable Explanations. UAI, PMLR 216:1434–1444. https://proceedings.mlr.press/v216/meyer23a.html
- Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 30. https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- Ian Connick Covert, Wei Qiu, Mingyu Lu, Na Yoon Kim, Nathan J White, Su-In Lee (2023). Learning to Maximize Mutual Information for Dynamic Feature Selection. ICML, PMLR 202:6424–6447. https://proceedings.mlr.press/v202/covert23a.html
- Michael Valancius, Maxwell Lennon, Junier Oliva (2024). Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition. ICML, PMLR 235:48957–48975. https://proceedings.mlr.press/v235/valancius24a.html

Source publication pages/abstracts verified; full-text methodology and limitations extraction remains outstanding. No invented DOI or quantitative literature results are included.
