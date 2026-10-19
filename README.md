> **Current version: reconstructed v3 (2026-10-19).** Includes re-executed R/Python survey audit, weighted calibration,18 extra classifiers, revised literature and a workflow diagram. See `docs/REVIEW_RESPONSE.md`. All follow-up evidence is post hoc; the original no-CI primary result remains unchanged.

## Completed availability-shift study and manuscript draft

The latest result is an empirical study of **stroke-history classification**, with frozen 2024 training and 2025 survey-year evaluation. Read [PROJECT_RESULTS.md](docs/PROJECT_RESULTS.md), the [IEEE-format draft](paper/manuscript.pdf), and [submission-readiness notes](docs/SUBMISSION_READINESS.md). The primary point effect is modest; the planned interval was unavailable due to singleton strata and a post-hoc sensitivity is reported separately. This is not a new-algorithm or clinical-validation claim. BRFSS 2025 has now been evaluated and must not be reused as an untouched test after further tuning.

Reproduce using [Availability_Study_Colab.ipynb](notebooks/Availability_Study_Colab.ipynb) or the commands in PROJECT_RESULTS.md. Raw data/checkpoints/participant predictions are excluded from the source package. Original project scripts remain preserved.

# Trust-aware stroke-history classification research

Research development branch. Original teammate implementation remains at the repository root, unchanged, for attribution and baseline reproduction. This project classifies self-reported stroke history; it does not predict incident stroke or provide clinical guidance.

Read the latest [optimization results](docs/OPTIMIZATION_RESULTS.md), the initial [technical audit](docs/AUDIT.md) and [research log](docs/RESEARCH_LOG.md) for evidence and outstanding work. DSI/CAS/RAI and conditional acquisition now have tested exploratory implementations. Scientific validation and novelty claims remain incomplete; see docs/TABNET_TRUST_RESULTS.md and docs/TRUST_PROTOCOL.md.

## Tested new baseline

From the repository root, in an isolated Python environment:

```bash
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
python -m truststroke.baseline --smoke --output results/my-smoke --bootstrap 50
```

For a local BRFSS file (not included):

```bash
python -m truststroke.baseline --data /path/to/LLCP2024.XPT --output results/brfss-logistic --model logistic
```

Select `--model forest` or `--model mlp` for original-variable comparators. The MLP uses the same hidden widths as the legacy learner but evaluates its direct probability output; it is not the legacy downstream bottleneck experiment. Each output directory must be new. Outputs include metrics.json, split positions (relative to cleaned records) and a fitted end-to-end model. Model artifacts contain no raw records but should remain local pending explicit publishing review. Do not load untrusted joblib files.

No dataset is bundled or uploaded. Obtain LLCP2024.XPT from the CDC 2024 documentation page or use your existing copy. Official codebook verification of inherited predictor recodes remains required before reporting real-data findings. Full XPT loading can use substantial RAM; use Colab when necessary.

The synthetic fixture is solely for execution testing. It does not model BRFSS prevalence or justify clinical performance claims. Bootstrap intervals currently assume IID respondents and condition on the fitted model. This simple baseline omits calibration fitting and tuning; the later optimization/trust workflows add these and exploratory attribution/missingness experiments. Design-aware CIs remain pending.

## Original pipeline reproduction (Legacy baseline)

The original teammate pipeline scripts are preserved under the [`legacy/`](legacy/) directory for provenance, attribution, and baseline comparison:

- `legacy/preprocessing.py`
- `legacy/feature_extraction.py`
- `legacy/feature_selection.py`
- `legacy/feature_scaling.py`
- `legacy/classification.py`
- `legacy/threshold_tuning.py`
- `legacy/visualizations.py`

Legacy scripts execute sequentially and rely on root/relative file paths with `LLCP2024.XPT` present. See [`legacy/README.md`](legacy/README.md) for details. Legacy scripts execute work on import; do not import them as a library.

## Controlled optimization

The new optimizer compares 11 predefined configurations: regularized/class-weighted logistic regression, L1 selection of encoded inputs, histogram boosting, shallow/deep/regularized/weighted MLPs, and a supervised 15-dimensional bottleneck followed by logistic regression.

```bash
python -m truststroke.optimize --data /path/to/LLCP2024.XPT --output results/development --epochs 30 --threads 2
```

Partitions are 55% fitting, 10% stopping, 15% selection, 10% calibration and 10% final test. Every candidate fits its preprocessing and feature selector only on fitting records. Neural stopping uses stopping average precision; boosting uses stopping log loss. Model selection uses selection average precision, with Brier as a tie breaker. Sigmoid calibration is fitted on calibration records; calibrated F1 threshold is selected on selection records. This is a compact engineering search with unequal search sizes across families, not a publication-ready equal-budget benchmark.

The default command never predicts on the final test. After freezing the development protocol:

```bash
python -m truststroke.evaluate_locked --data /path/to/LLCP2024.XPT --run results/development --bootstrap 500
```

This command refuses to overwrite an existing test result. Different seeds must retain the same held-out cohort in a future repeated-training study; the current CLI's seed changes partitions, so do not present separate-seed runs as a matched-cohort experiment.

A [Colab notebook](notebooks/BRFSS_Optimization_Colab.ipynb) mounts Drive and applies the provided patch to your fork without requiring a push. The notebook structure and cell syntax are checked locally; its Drive/Colab integration has not been executed here. Dataset provenance and coding checks are in docs/DATA_PROVENANCE.json and docs/FEATURE_METADATA.md.

An optional exploratory blend stage evaluates neural weights 0.25, 0.50 and 0.75 using the same selection partition, before any final testing:

```bash
python -m truststroke.ensemble_search --data /path/to/LLCP2024.XPT --run results/development
```

The strongest direct MLP is refitted with its selected settings and blended with the originally selected model. This adds search flexibility and must be disclosed; it is not independent evidence. After the one-time final evaluation, survey-weighted point metrics are also saved when _LLCPWT is available; those do not include design-based intervals.

To regenerate charts from saved aggregate metrics without reopening the dataset:

```bash
python -m pip install -r requirements-figures.txt
python -m truststroke.make_report --run results/development
```


## TabNet and trust experiments

TabNet is optional. The bounded 30-epoch CPU search did not beat the existing blend; retaining that negative result is part of the study. Use requirements-tabnet-cpu.txt for the tested CPU build, or preserve a compatible GPU torch build and install requirements-tabnet.txt on Colab.

```bash
python -m truststroke.tabnet_search --data /path/to/LLCP2024.XPT --partitions-from results/development --output results/tabnet --epochs 30
python -m truststroke.trust_experiment --data /path/to/LLCP2024.XPT --partitions-from results/development --output results/trust
python -m truststroke.attribution_convergence --data /path/to/LLCP2024.XPT --run results/trust
python -m truststroke.availability_refit --data /path/to/LLCP2024.XPT --trust-run results/trust --output results/availability
```

A standalone split protocol can be created with `python -m truststroke.prepare_development --data /path/to/LLCP2024.XPT --output results/protocol` without rerunning all baseline models. New TabNet/trust searches do not access the previously inspected final test. All model directories are local and ignored by git. TabNet checkpoints use native network/config serialization inside the portable predictor; cross-process reload is tested.

Run the [new Colab notebook](notebooks/TabNet_Trust_Colab.ipynb) to reproduce the development experiments from the uploaded patch. Notebook syntax is checked locally, not its Colab account/mount integration. Prefer setting BLAS/OpenMP thread counts to 2 on constrained CPUs.

Mathematical definitions, edge cases, no-causal-attribution limitations and evaluation rules are in [TRUST_PROTOCOL.md](docs/TRUST_PROTOCOL.md). [The literature review](docs/literature/REVIEW.md) contains 14 verified references with explicit full-text extraction gaps. Existing work on stability, uncertainty, missingness and acquisition prevents a blanket claim that these concepts are novel.

The retained model and TabNet were also compared, without external-data fitting, on the separate 2023 CDC survey release. This is reverse-calendar year transport of stroke-history classification, not prospective future-stroke prediction. Exact scores, source hashes, the failed first checkpoint attempt and frozen-model evidence are in docs/TABNET_TRUST_RESULTS.md and docs/experiments/trust/.

```bash
python -m truststroke.external_validation --data /path/to/LLCP2023.XPT --original-run results/development --tabnet-run results/tabnet --output results/external2023
```

Do not use the external results to continue tuning. The trust layer itself currently has development-only validation, and the acquisition prototype has not demonstrated a benefit over its baselines.
