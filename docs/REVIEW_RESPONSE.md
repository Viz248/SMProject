# Reconstructed revision v3 — 2026-10-19

This revision reconstructs unavailable post-review artifacts using actual retained source data and frozen predictions. Results were re-executed rather than copied from chat statements. It supersedes the five-page manuscript whose SHA256 is3458e74e09ff1a11e777f430f8f157b137b809589b3d1e83a89ac2f574e19f29. The historical original experiment is unchanged; all follow-up analyses are post hoc on already-inspected 2025 data.

| Concern | Completed action |
|---|---|
| Survey design after outcome restriction | Full common-jurisdiction design retained 350,427 rows; valid outcomes 349,241 form a ratio domain. Both have 2,034 strata and 93 singletons; filtering did not create them. |
| Independent inference | Actual R 4.3.3/survey 4.2.1 execution of full-design svyratio; fail returns no interval, adjust/average sensitivities stored. Separate Python PSU algebra agrees. No FPC, no invented certainty units, no arbitrary strata merging. |
| Calibration explanation | Weighted CITL, joint intercept/slope, weighted ECE and curves for 32 frozen ensembles. Diagnostic fits are not deployed updates. |
| Training-sample uncertainty | Three new stratified 60,000-record fitting draws; three unchanged seeds each;18 classifiers. Stop/calibration cohorts and hyperparameters fixed; draws can overlap. Every result reported. |
| Closest literature | Janssen2009, Fletcher Mercaldo/Blume2020, Hoogland2020, Sisk2023, Zhou2023 and temporal updating by Cooray2023. Older direct methodological predecessors retained because relevant. |
| Framing | Controlled empirical method comparison; no algorithmic invention, prospective stroke risk, clinical benefit or arbitrary-mask claim. Operational contrasts are not a causal decomposition. |
| Figures | Categorical scenario labels; weighted calibration; compact study-workflow diagram. Numerical subgroup results retained, domain figure supplied in package. |
| Disclosure and reproducibility | AI assistance disclosed; versioned PDF and ZIP, checksums, tests and executed audit records supplied. |

## Interpretation

The original planned confidence interval remains unavailable. Both later inference options are assumption-dependent sensitivities, even if they exclude zero. The additional fitting-sample results are robustness observations, not fresh 2025 confirmation. Lower Brier does not imply every calibration metric improves.

## Still required for an actual conference submission

Real author order/affiliations, target venue and page rules, author/supervisor scientific approval, truthful funding/conflict statements and the institution's determination for secondary public-data research. No approval/exemption or acceptance is invented. No external push or submission is performed.

## Reproduction

The original notebook reproduces the frozen experiment. Then run `python -m truststroke.revision_audit diagnostics --data PATH_TO_2025_XPT --run FIT_DIRECTORY --external EVALUATION_DIRECTORY --output AUDIT_DIRECTORY` and `Rscript scripts/verify_survey_domain.R EVALUATION_DIRECTORY/posthoc_survey_domain.csv AUDIT_DIRECTORY/r_survey.csv`. Run `python -m truststroke.revision_audit robustness --data PATH_TO_2024_XPT --data25 PATH_TO_2025_XPT --run FIT_DIRECTORY --external EVALUATION_DIRECTORY --output AUDIT_DIRECTORY --robustness-output NEW_MODEL_DIRECTORY` for the 18 additional models. Output model directories must be new; raw records and participant predictions remain excluded from the source archive.
