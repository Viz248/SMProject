# Current project status — 19 October 2026

| Component | Executed status | Evidence / limitation |
|---|---|---|
| Original audit and reproduction | Completed | AUDIT.md; original scripts preserved |
| MLP/tree/TabNet comparisons | Completed bounded searches | OPTIMIZATION_RESULTS.md; TABNET_TRUST_RESULTS.md; no universal best-model claim |
| Trust prototypes and attribution-noise analysis | Implemented and evaluated exploratorily | TRUST_PROTOCOL.md; no novel metric or comprehensive trust claim |
| DIME acquisition hypothesis | Two development pilots complete; improvement unsupported | DIME_PILOT_RESULTS.md; DIME_MATCHED_RESULTS.md |
| Availability-study protocol | Frozen before 2025 predictions | configs/availability_study_frozen.json; commit 4772823 |
| Final training/calibration | 30 classifiers, two families, three seeds, separate stopping/calibration | docs/experiments/availability_paper/model_manifest.json |
| Forward survey-year evaluation | Frozen evaluation completed on 354,950 labeled 2025 respondents | 349,241 common-jurisdiction primary cohort; no external refitting |
| Primary inference | Point estimate complete; frozen interval unavailable | 93 singleton strata; sensitivity is explicitly post-hoc, not replacement primary inference |
| Figures/tables/provenance | Completed | docs/experiments/availability_paper/ |
| IEEE-format manuscript | Five-page anonymous research draft completed | paper/manuscript.pdf and source; author/venue/statistical review remain |
| Reproduction package | Code, notebook, download/checksum utility, manifests, tests | Notebook syntax checked; Colab account integration untested |
| Tests | 33 passed, no skips | Includes optional TabNet checkpoint and survey variance checks |
| Submission/publication | Not submitted | SUBMISSION_READINESS.md; no acceptance or algorithmic-novelty guarantee |

2025 is now an inspected evaluation year. Any later model or analysis tuning using its results must be described as exploratory. The endpoint remains self-reported stroke history, not incident stroke.
