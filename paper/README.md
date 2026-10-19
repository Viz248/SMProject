# Manuscript package

This directory contains the IEEE conference-format research draft and its editable LaTeX source. It is an empirical study, not a claim to invent missingness-aware learning. It is not submitted or accepted. Anonymous author metadata is used until the real author list and target venue are supplied.

The main experiment is frozen in `configs/availability_study_frozen.json` at commit `4772823`, before 2025 predictive evaluation. Model and source hashes are checked by the external evaluator. Historical development/test results elsewhere in this repository remain exploratory for the final research direction.

## Rebuild

Run `pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex` twice in this directory. The included unmodified IEEEtran class comes from CTAN, version 1.8b; its original license header is preserved. Match the eventual conference's template/page limit and author-identification rules before submission. This generic draft uses US letter paper and IEEEtran conference mode.

Generated scientific figures are based on saved aggregate results; the underlying frozen-model prediction arrays remain in ignored experiment directories. Do not commit raw BRFSS records or participant-level prediction arrays.

## Current reconstruction v3

Reconstructed on 2026-10-19 after loss of the earlier revised archive. The new PDF includes actual re-executed survey/calibration/fitting-sample analyses and a workflow diagram. See `docs/REVIEW_RESPONSE.md`. It supersedes the older five-page PDF, not the frozen statistical history. Run `python paper/build_manuscript.py` after generating revision aggregates, then compile twice inside `paper/`. Versioned deliverables are included in the source ZIP.
