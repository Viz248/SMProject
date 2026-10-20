# Instructions for Writing the TrustStroke IEEE Conference Paper

**Role:** You are an expert scientific author writing a submission-ready, rigorous IEEE conference paper for the TrustStroke project.  
**Deliverable Directory:** Create a complete, self-contained LaTeX project in `paper/ieee_conference_submission/`. Do **not** overwrite `paper/manuscript.tex`.

---

## 1. Ground Truth Source Documents
You must base all mathematical formulas, numbers, experimental protocols, and literature citations strictly on the accompanying handoff dossier:
1. `paper/research_handoff/RESEARCH_DOSSIER.md` (Equations, cohorts, protocols, exact numerical results, negative findings)
2. `paper/research_handoff/FIGURES_AND_TABLES.md` (Figure assets, exact tabular data, recommended visual plan)
3. `paper/research_handoff/REFERENCES_AND_CLAIMS.md` (Verified BibTeX entries, claim-to-artifact mapping)

Additional source assets you can directly copy into your deliverable directory:
* Figures: `paper/figures/availability.pdf`, `paper/figures/calibration.pdf`, `paper/figures/subgroups.pdf`, `docs/experiments/trust/trust_overview.pdf`
* Algorithm Draft: `paper/adaptive_algorithm.tex`
* LaTeX Template Style: `paper/IEEEtran.cls`

---

## 2. Narrative Architecture & Scientific Positioning

### 2.1 The Core Scientific Thesis
When machine learning models are deployed on survey questionnaires (such as CDC BRFSS for stroke-history screening), unavailable questions cause severe performance degradation. This paper presents **TrustStroke**, an auditable, trust-aware evaluation framework that:
1. **Disentangles Information Loss from Input Representation Shift:** Demonstrates that post-hoc scenario-specific probability recalibration (Platt scaling) recovers the vast majority of degradation under naive feature masking without modifying model weights.
2. **Quantifies Trust Diagnostics:** Provides empirical benchmarks for feature-attribution stability (DSI), uncertainty decomposition into member entropy and ensemble disagreement (CAS), and headroom retention across missingness patterns (RAI).
3. **Validates External Forward Temporal Transport (2024 $\to$ 2025):** Evaluates 30 frozen classifiers across 349,241 survey-weighted respondents, demonstrating that mixed-mask training achieves statistically significant probability loss reductions over scenario recalibration alone ($\hat\theta = -0.000157$).
4. **Empirically Assesses Sequential Acquisition & Abstention:** Evaluates dynamic donor-based feature acquisition against static clinical heuristics across 91,244 locked test records, revealing critical deployment lessons:
   - Dynamic acquisition is matched or outperformed by simple clinical priority rules (e.g., querying prior myocardial infarction, `CVDINFR4`) at less than half the computational latency.
   - Symmetric Shannon entropy safety gates fail in low-prevalence screening ($4.5\%$ base rate), inadvertently abstaining on high-risk individuals ($26.9\%$ prevalence among abstained cases).

### 2.2 Incompatibility Guardrail (Crucial Methodological Rule)
* **Never merge or average the 2024–2025 temporal survey-weighted results with the 2024 unweighted adaptive results into a single table.**
* Clearly present them as **two distinct, complementary evaluations**:
  - **Section V-A:** The Large-Scale Temporal Transport Study ($N = 349,241$, survey-weighted, Taylor series design variance).
  - **Section V-B:** Diagnostic Trust Pilots (DSI, CAS, RAI on development cohorts).
  - **Section V-C:** The Adaptive Decision Policy & Acquisition Benchmark ($N = 91,244$ locked test partition, I.I.D. evaluation).

---

## 3. Required Paper Outline (Professor's Structure)

### I. Abstract
* Clinical/survey motivation: Telephone screening for stroke history under variable question availability.
* Core problem: Conflating information loss with input representation mismatch; lack of calibrated probabilities and attribution stability.
* Framework components: Recalibration, mixed-mask training, DSI explanation stability, CAS uncertainty quantification, and sequential acquisition.
* Key quantitative findings:
  - 2025 temporal evaluation ($N=349,241$): Scenario recalibration closes the bulk of degradation; mixed-mask training yields $\hat\theta = -0.000157$ ($p < 0.05$ under post-hoc sensitivity).
  - DSI attribution stability converges to $>0.92$ at 256 paths; seed stability is $0.890$.
  - Acquisition benchmark: Static clinical priority (`CVDINFR4`) matches dynamic donors with $2.3\times$ faster runtime ($8.13\text{ ms}$ vs. $18.39\text{ ms}$).
  - Honest disclosure: Symmetric entropy abstention discards 532 stroke cases at $26.9\%$ prevalence.
* Scope disclaimer: Classifies self-reported stroke history, not incident stroke prediction.

### II. Introduction
* Context: AI screening tools on population health surveys (BRFSS).
* Deployment dilemma: Missing questions at kiosk or telephone interviews. Naive imputation breaks calibrated probabilities.
* Four key questions answered by this paper:
  1. How much degradation is due to miscalibration vs. representation loss?
  2. Are post-hoc explanations stable against Monte Carlo sampling and retraining?
  3. Does ensemble disagreement signal misclassification?
  4. Does active information acquisition justify inquiry latency?
* Summary of contributions (clearly delineated, no overclaimed novelty).

### III. Related Work & Research Gap
* **Stroke Risk Modeling on Questionnaires:** Review Ding et al. (2025) and Moulaei et al. (2024). Explain why survey models classify history rather than predict incident events.
* **Missing Data & Pattern Submodels:** Review Janssen et al. (2009), Fletcher-Mercaldo & Blume (2020), Hoogland et al. (2020), and Sisk et al. (2023). Contrast our deliberate availability sets with ad-hoc imputation.
* **Explanation Stability & Attribution Noise:** Review Meyer et al. (2023) and Chen et al. (2024). Note that explanation variability under dataset shift is an established problem; our DSI diagnoses Monte Carlo path noise vs. seed retraining.
* **Predictive Uncertainty & Active Feature Acquisition:** Review Lakshminarayanan et al. (2017) (deep ensembles / entropy decomposition), Covert et al. (2023), and Valancius et al. (2024).
* **Research Gap:** Existing literature either evaluates models on static complete datasets or proposes complex neural architectures without testing whether simple recalibration or static clinical heuristics suffice under real survey-weight transport.

### IV. Proposed Work & System Architecture
* **System Architecture Overview:** Provide a clear architectural block diagram (Figure 1) connecting data ingestion, preprocessing, availability models, Platt scenario recalibration, CAS uncertainty gate, recommender engine, and decision/abstention.
* **Algorithm Pseudocode:** Include the complete IEEE algorithm environment matching `paper/adaptive_algorithm.tex` (`AdaptiveDecisionPolicy.decide()`).
* **Module Descriptions & Mathematical Equations:**
  - *Module 1: Preprocessing & Missingness Encoding:* Explain categorical encoding, training-median imputation, missingness indicators, and zeroed feature blocks.
  - *Module 2: Model Families & Scenario Recalibration:* Boosted trees, MLPs, and Platt scaling equation ($P_{\text{cal}} = \sigma(A_s \cdot \text{logit}(p) + B_s)$).
  - *Module 3: Trust Diagnostics (DSI, CAS, RAI):* Detail Equations for DSI ($L_1$ signed similarity), CAS ($H_{\text{total}} = H_{\text{within}} + H_{\text{between}}$), and RAI (Brier headroom retention).
  - *Module 4: Recommender Engine:* Contrast `ConditionalDonors` (mutual information over 64-NN complete training donors) with `StaticFeatureRecommender` (clinical priority queue).
  - *Module 5: Safety Abstention:* Symmetric entropy threshold $\tau_{\text{abstain}}$.
* **Dataset Description & Preprocessing:**
  - Official CDC BRFSS 2024 ($N=456,218$) and 2025 ($N=349,241$ common-jurisdiction).
  - The 15 canonical predictors, coding rules, and the four predefined availability scenarios (Full, Without Health State, Demographics+History, Demographics).
  - Table I (Cohort and partition splits).

### V. Results & Discussion
* **V-A. External Temporal Transport Evaluation (2025 Common Jurisdictions, $N=349,241$):**
  - Discuss Table II and Figure 2 (availability curves).
  - Show how scenario recalibration (SC) recovers the vast majority of degradation caused by global calibration (FG).
  - Report the primary contrast: Boosted trees MT vs. SC gives $\hat\theta = -0.000157$.
  - Transparently report the 93 singleton strata, the omission of the prespecified Taylor linearized CI, and the post-hoc sensitivity intervals.
  - Discuss calibration reliability curves (Figure 3) and CITL/slope metrics.
* **V-B. Diagnostic Trust Framework Evaluation (DSI, CAS, RAI):**
  - Discuss DSI Monte Carlo convergence ($0.736 \to 0.928$) and seed/bootstrap stability ($0.890 / 0.856$).
  - Discuss CAS uncertainty: Report the negative finding that standard predictive entropy achieves higher error-detection AUROC ($0.936$) than ensemble disagreement ($0.830$).
  - Discuss RAI: Report 7-scenario retention ($0.717$) and explain why scalar indices obscure scenario-specific trade-offs.
* **V-C. Adaptive Decision Policy on the Locked 2024 Test Partition ($N=91,244$):**
  - Discuss Table III: Full-population probability metrics vs. covered-only decision metrics across all four scenarios.
  - Highlight the scenario dichotomy: Significant gains in `Demographics` (AUROC $+0.0205$, sensitivity $0\% \to 14.59\%$), but statistically significant Brier degradation in `Full` ($+0.000269$) and `Without Health State` ($+0.000438$).
  - Discuss the rare-disease abstention defect: The symmetric entropy gate discarded 532 stroke cases ($26.9\%$ prevalence among abstained respondents).
* **V-D. Acquisition Strategy Benchmark (Dynamic vs. Static Heuristics):**
  - Discuss Table IV: Compare `ConditionalDonors` against Static `CVDINFR4` and Static `GENHLTH`.
  - Report that static `CVDINFR4` achieves lower Brier, higher AUROC ($0.7455$ vs. $0.7281$), and runs $2.3\times$ faster ($8.13\text{ ms}$ vs. $18.39\text{ ms}$). Explain why domain-guided queries beat noisy non-parametric donor distributions.

### VI. Conclusion & Future Work
* Reiterate the main takeaway: Recalibration and static clinical heuristics provide robust, computationally efficient remedies for missing survey data; dynamic non-parametric acquisition and symmetric entropy abstention require substantial adaptation for rare-prevalence endpoints.
* Future Work: Cost-sensitive, asymmetric abstention gates; survey-weighted online active learning; multi-question adaptive interview paths.

### References
* Use standard IEEE citation format (`\bibliographystyle{IEEEtran}`) citing all verified entries from `paper/research_handoff/REFERENCES_AND_CLAIMS.md`.

---

## 4. Deliverable Structure & Compilation Requirements

Save all output files into `paper/ieee_conference_submission/`:
```
paper/ieee_conference_submission/
├── main.tex                  # Primary LaTeX document (IEEEtran conference format)
├── references.bib            # Complete BibTeX bibliography
├── IEEEtran.cls              # Copied from paper/IEEEtran.cls
├── figures/                  # Directory containing all referenced PDF/PNG figure assets
│   ├── availability.pdf
│   ├── calibration.pdf
│   ├── subgroups.pdf
│   └── trust_overview.pdf
```

### Strict Verification Rules:
1. **Compilability:** The LaTeX project must compile cleanly with `pdflatex` and `bibtex` without missing reference errors (`?`) or missing figure warnings.
2. **Numerical Exactness:** Check every number against `RESEARCH_DOSSIER.md`. Do not invent or round numbers arbitrarily.
3. **Traceability:** Every major result in the text must cite the corresponding table or figure.
4. **Author Placeholders:** Use realistic placeholder authors (e.g., `Student Author`, `Faculty Supervisor`, `Department of Computer Science and Engineering`).
