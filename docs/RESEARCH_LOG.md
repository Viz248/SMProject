# Research evidence log

2026-10-19: Audited seven project files and .gitignore at fork commit 095ae4d367d1c34000e1c235e9f4a93596c99791. Original files unchanged. Created research/trust-baseline.

| Claim | Supporting code or evidence | Scope |
|---|---|---|
| Data split occurs before fitted transformations | truststroke/baseline.py split_indices/build_model/run; tests/test_baseline.py | New baseline only |
| Target and design variables excluded from inputs | clean and cleaning test | Listed original 15 predictors only |
| Validation determines threshold | run, threshold_selection in saved metrics | No clinical threshold claim |
| Execution works for logistic, RF and MLP | results/smoke/metrics.json; results/smoke-forest/metrics.json; results/smoke-mlp/metrics.json | Artificial n=1000 fixture, seed 42; all commands exit 0 |
| Five unit tests pass | python -m unittest discover -s tests -v | Cleaning, split isolation/repeatability, imputer isolation, metrics, bootstrap repeatability |
| Original neural approach has not been reproduced | No LLCP2024.XPT available | No real BRFSS performance claims |
| Prior work exists on stability and acquisition | docs/AUDIT.md verified publisher links | Initial review, not systematic novelty assessment |

Smoke outcomes (not publishable BRFSS findings):

| Model | AUROC | Average precision | Brier | ECE | Bootstrap count |
|---|---:|---:|---:|---:|---:|
| Logistic | .69739 | .22823 | .11005 | .04422 | 50 |
| Random forest | .66313 | .22254 | .10895 | .01734 | 10 |
| Direct MLP | .65208 | .18430 | .16876 | .23808 | 10 |

Small smoke bootstrap counts exercise software only. Direct MLP's synthetic calibration is poor, illustrating why architecture alone does not establish trustworthy probabilities. No model selection is based on these smoke outcomes.

Pending: dataset and full codebook verification; legacy baseline capture; frozen BRFSS protocol; nested preprocessing for supervised feature learning; calibration and fair tuning; survey-aware evaluation; complete literature review; implementation/validation of trust components; conference choice and manuscript. No publication phase is complete.

## 2026-10-19 — real-data optimization milestone

Connected Drive exact-file metadata resolved, raw fetch failed at the 256 MiB connector cap. Downloaded the independent public CDC 2024 XPT archive and official codebook instead; byte equality to the selected Drive file has not been verified. See DATA_PROVENANCE.json for source hashes.

Completed 11-candidate full-fitting-partition search and three exploratory blend weights. Selected 75% boosted trees plus 25% regularized MLP, calibrated on a separate partition. Final locked test evaluated exactly once: n=45,622, AUROC=.813497, AP=.179478, Brier=.039928, sensitivity=.332527, specificity=.938746 at development-selected threshold .14. 500 IID bootstrap replicates; no design-aware CIs or external validation. Details, exact settings and negative findings are in OPTIMIZATION_RESULTS.md and experiments/*.json.

Twelve unit tests pass, including no test predictions during default development, model serialization, and guards against post-test tuning or result overwrite. Notebook syntax checked, Colab execution not performed. The original baseline reproduction completed in an unchanged detached worktree of the audited commit (exit 0). Original MLP stopped at 13 epochs. All four saved-prediction AUROC/AP/F1 values match the results CSV; see experiments/original_reproduction.json and experiments/original_classification.csv. Different test partitions prevent a direct improvement claim. DSI/CAS/RAI and conditional acquisition remain pending.

## 2026-10-19 — TabNet, trust prototypes and survey-year transport

Full-data bounded TabNet search (three configurations, 30 epochs, original development partitions) completed. Best selection AP .167540, below prior blend .193836. TabNet native checkpoint format and fresh-process test were added after a portability/corrupt-pickle failure; identical-setting refit reproduced selection AP. No hyperparameter changes used external outcomes.

Implemented and executed DSI/CAS/RAI and conditional acquisition pilot on 2024 development records: three initialization seeds plus two bootstrap MLP members; 2,048 evaluation records, 128 explanation records, 256 acquisition records. Attribution MC repeat stability rose from .736 at 16 paths to .928 at 256; seed DSI at 256 was .890 and bootstrap DSI .856. CAS did not establish benefit beyond predictive entropy; acquisition did not establish proper-score benefit over random/global baselines. Fixed-mask and subset-retraining experiments are reported separately. See TRUST_PROTOCOL.md and TABNET_TRUST_RESULTS.md.

Frozen models evaluated on 431,849 labeled public BRFSS 2023 records, after inspecting compatible coding: blend AUROC .815018/AP .177416 versus TabNet AUROC .795426/AP .153886. This is reverse-calendar year transport, not prospective risk or clinical validation. An aborted first comparison after blend scoring is preserved; native checkpoint recovery enabled the completed, unchanged-setting comparison.

21 tests passed, including fresh-process native checkpoint loading and removed-feature isolation. Notebook JSON/Python cell syntax checked; Colab account/Drive execution untested. Fourteen publication records verified (11 recent plus 3 foundational); full-text extraction gaps remain explicit. No metric novelty or submission-readiness claim.


## 2026-10-19 — DIME prior-art correction and target-estimation pilot

Read both uploaded research reports; verified and cloned the official DIME reference at 06ba70f7bcd54dbd8ccd64bed398dfef90d09da5. Its incremental-loss value regression overlaps the proposed mechanism; cross-fitting is an unvalidated adaptation. Executed a three-seed fixed-predictor pilot on 6,000/1,500 complete-case BRFSS development respondents and 398/171 Wisconsin diagnostic breast-cancer records. Eight policies, three-fold targets, same-size teacher and split-sample controls, budget curves, value diagnostics and paired record bootstraps saved. Every primary interval includes zero; static mean Brier is lower than cross-fitting on both datasets. No improvement adopted. No 2025 or new final-test evaluation.

Official DIME classes also completed a separate three-seed execution check with explicit environment compatibility wrappers; not a published-score reproduction or matched superiority comparison. 24 unit tests ran, 22 passed and 2 optional TabNet checks skipped in refreshed runtime. See DIME_PILOT_RESULTS.md and experiments/dime_pilot/.


## 2026-10-19 — Matched official-DIME follow-up and checkpoint diagnostic

Executed official CE and Brier joint-training classes plus three Brier fold teachers for each of three seeds on BRFSS development and Wisconsin diagnostic data. Shared frozen Brier predictor isolates refitted in-sample versus CF value targets; native joint heads, same-size teachers, post-loss, random and static controls are reported separately. Independent stopping partitions, fold-only preprocessing, original-variable acquisition groups, same-state diagnostics and native checkpoints are retained. Matched CF was worse in all six seed runs; two Wisconsin conditional record intervals excluded zero, all BRFSS intervals included zero. Mean BRFSS Brier at budget three: in-sample .036958, CF .037471, static .036339. No improvement adopted, no 2025/final-test evaluation.

Post-hoc checkpoint-only analysis showed Wisconsin CF values fit the average fold-teacher utility better but full deployment utility worse, consistent with teacher/deployment mismatch; it is not a causal or novelty claim. This does not rescue the method as a new algorithm. See DIME_MATCHED_RESULTS.md and experiments/dime_matched/. Twenty-seven tests ran (25 passed, two optional TabNet skips). All 192 saved Brier values verified; execution source hash and disjoint splits checked. Cross-fitting remains a negative ablation, not the established paper contribution.


## 2026-10-19 — Frozen availability study and IEEE-format draft

Committed final availability protocol at 4772823 before 2025 predictive evaluation. Verified 2025 schema/codebook compatibility for all inherited predictors/outcome/design fields and downloaded independent official CDC XPT (SHA-256 d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2). Thirty models were trained on 60,000 2024 respondents with separate 10,000 stopping and 45,622 calibration records; 15,000 development records evaluated. A ColumnTransformer dropped-remainder implementation error stopped the first fitting attempt before external scoring; fixed and restarted with identical scientific settings. Hash-frozen models scored the official 2025 release exactly once without external refitting.

2025 labeled n=354,950, positive=16,353; common-jurisdiction primary n=349,241, positive=16,082, 49 jurisdictions. Weighted seven-variable HGB Brier: global .0332212, scenario calibration .0329937, mask training .0328371, subset refit .0327834. Primary difference -.000156566. Prespecified Taylor rule correctly returned no interval because 93 strata had one PSU. A separately reported post-hoc average-stratum-contribution sensitivity gives approximately [-.0002563,-.0000569], conditional on models and an extra assumption; it is not substituted for primary inference and was not independently checked in R. Models/predictions unchanged.

Produced five-page IEEEtran manuscript, editable source, full metric tables, plots, provenance, download/checksum script, one-task-per-cell Colab reproduction notebook and source package. All 33 tests passed, including optional TabNet checks. No final remote push or conference submission; author/venue/disclosures and scientific/statistical review are listed in SUBMISSION_READINESS.md. 2025 is no longer untouched.

## Reconstructed revision v3 (2026-10-19)

The previous revised archive could not be recovered; the uploaded renamed PDF matched the older five-page manuscript byte for byte. A separate reconstruction protocol was committed before re-execution. The original frozen models/predictions remained intact. Full-design domain inference, actual R survey verification, weighted calibration and 18 additional fixed-protocol classifiers were re-executed. The three fitting-sample ensemble contrasts reproduce -0.0001626924, -0.0000399038 and -0.0001749244. All 35 tests pass. The revised manuscript adds a compact workflow figure, more relevant prior art and AI disclosure. New versioned filenames identify the reconstructed v3 delivery; a source manifest and checksums verify its contents. Follow-up analyses remain post hoc, and the original primary interval is still unavailable.
