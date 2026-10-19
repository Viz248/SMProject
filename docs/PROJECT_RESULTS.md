# Completed analysis and IEEE-format draft

The final research package studies **calibration and training controls for stroke-history classification under questionnaire availability shift**. It is an empirical comparison, not a new deep-learning architecture, a new trust metric, or prospective stroke prediction.

## Executed final study

The protocol was committed at `4772823` before 2025 predictive evaluation. Thirty classifiers were fitted on actual public CDC 2024 data: boosted trees and an MLP, three seeds, full/mixed-mask training and three subset refits. Classifier fitting used 60,000 respondents, stopping used 10,000, calibration used 45,622 and development evaluation used 15,000. Natural predictor nonresponse was retained. Models and source hashes were frozen before scoring 2025.

The official 2025 file contained 356,158 records. Valid stroke-history labels remained for 354,950 respondents (16,353 positives); the common-jurisdiction primary cohort contained 349,241 respondents (16,082 positives), across 49 reporting jurisdictions.

All four procedures use the same original-variable availability scenarios: 15 full inputs; 12 excluding current-health-state items; 7 demographics plus five disease histories; and 2 demographics. Procedures separate global calibration, scenario-specific calibration of the fixed model, mixed-mask training and subset retraining.

## Main result

The prespecified comparison is the survey-weighted paired Brier difference between mask-trained and scenario-recalibrated boosted-tree ensembles under the seven-variable condition. Each prediction is the mean of three calibrated seed probabilities.

| Procedure | Weighted Brier, 2025 common jurisdictions |
|---|---:|
| Fixed full model, global calibration | 0.0332212 |
| Same fixed model, scenario calibration | 0.0329937 |
| Mixed-mask training | 0.0328371 |
| Subset retraining | 0.0327834 |

Primary difference (mask training minus scenario calibration): **-0.000156566**. Lower Brier is better. This is a modest probability-loss change, not a clinical effect.

The frozen Taylor interval rule returned **no primary interval**, because 93 strata contained only one PSU. This outcome is retained and disclosed; no prespecified statistical-superiority claim is made. A separately labeled post-hoc sensitivity replaces singleton variance contributions by the average non-singleton stratum contribution, following the survey package documentation. Its conditional 95% interval is approximately [-0.0002563, -0.0000569]. This sensitivity requires an additional comparability assumption, was not independently run in R survey and does not replace the unavailable frozen interval.

A practical secondary finding is that scenario recalibration can remove much of the apparent failure of naive masking. For the demographics-only MLP, weighted Brier changes from 0.0842884 under global calibration to 0.0343312 under scenario calibration, and 0.0343008 under mask training. This illustrates why recalibration belongs in a robustness comparison. Secondary results are descriptive and not a basis for selecting a new model after external evaluation.

## What earlier experiments established

- The original pipeline was reproduced; controlled MLP/tree optimization and a bounded TabNet search were executed on actual CDC data. TabNet did not beat the earlier development blend under the declared budget.
- DSI/CAS/RAI prototypes and explanation-estimator noise controls were implemented. Their names do not establish scientific novelty or general trustworthiness.
- The proposed acquisition objective overlaps DIME. Two controlled development studies did not support a cross-fitting improvement. These remain negative ablations, not the main paper contribution.
- Frozen 2024 models were previously evaluated on 2023, which is reverse-year transport. The final study adds forward survey-year transport to 2025; neither is future incident-stroke prediction.

## Deliverables

- `paper/manuscript.pdf`: five-page IEEEtran conference-format anonymous research draft.
- `paper/manuscript.tex` and `paper/build_manuscript.py`: editable, regenerable source using actual aggregate results.
- `docs/experiments/availability_paper/`: final protocol, manifest, training histories, provenance, all metric tables, cohort summaries, variance sensitivity and scientific figures.
- `notebooks/Availability_Study_Colab.ipynb`: one-task-per-cell reproduction workflow; syntax checked, account-specific integration untested.
- `scripts/download_brfss.py`: official CDC download with archive/XPT checksum verification.
- `truststroke/availability_study.py` and CLI shim: training/calibration and guarded external evaluation.
- `docs/SUBMISSION_READINESS.md`: remaining author, venue and review decisions.

The full model checkpoints, split positions and participant-level predictions remain in ignored experiment directories and are not included in the source ZIP. They can be regenerated from verified public files and the frozen configuration. The original seven project scripts are preserved.

## Reproduction

Install the CPU torch build first to avoid a large CUDA download, then install `requirements-availability.txt`. Use a fresh output directory for every run.

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-availability.txt
python scripts/download_brfss.py --directory /path/to/brfss_data
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m truststroke.run_availability_study fit --config configs/availability_study_frozen.json --data /path/to/brfss_data/LLCP2024.XPT --output results/reproduction_fit
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m truststroke.run_availability_study evaluate --run results/reproduction_fit --data /path/to/brfss_data/LLCP2025.XPT --output results/reproduction_2025
python -m truststroke.package_availability --run results/reproduction_fit --external results/reproduction_2025 --development-data /path/to/brfss_data/LLCP2024.XPT --output docs/experiments/reproduction
python -m truststroke.survey_sensitivity --run results/reproduction_fit --external results/reproduction_2025 --output docs/experiments/reproduction/singleton_sensitivity.json
```

The manuscript builder reads the supplied aggregate path `docs/experiments/availability_paper/`; change that path deliberately if building a separately reproduced version. Build the PDF twice using `pdflatex` in `paper/`. A rerun of already evaluated 2025 data is reproduction, not another independent confirmation.

All 33 automated tests pass. The paper was rendered and visually inspected; unresolved citations and template markers were checked. The final package's verification record lists the data-integrity checks. These checks establish software traceability, not conference acceptance or clinical validation.

## Current reconstructed v3 revision

See `docs/REVIEW_RESPONSE.md` and `docs/experiments/revision_audit/` for re-executed full-design R/Python inference,32 weighted calibration diagnostics and 18 additional classifiers across three fitting samples. All 35 tests pass. The primary interval remains unavailable; all follow-up analyses are post hoc. Current versioned PDF and ZIP supersede the earlier five-page delivery.
