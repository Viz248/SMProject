# TrustStroke System Architecture Notes & Component Mapping

This document provides the architectural specification and implementation trace for the publication-quality TrustStroke system architecture diagram ([`truststroke_architecture.pdf`](truststroke_architecture.pdf), [`truststroke_architecture.svg`](truststroke_architecture.svg), [`truststroke_architecture.drawio`](truststroke_architecture.drawio)).

---

## 1. Architectural Overview & Design Principles

The TrustStroke framework is organized into **four distinct functional sections**, strictly distinguishing:
1. **Section 1 (Population Data Ingestion, Availability Modeling & Scenario Recalibration):** Public CDC BRFSS ingestion, deterministic cleaning, simulated missingness masks, baseline model training, and scenario-specific Platt recalibration.
2. **Section 2 (Runtime Adaptive Decision Policy):** The sequential online per-record inference loop (`AdaptiveDecisionPolicy.decide()`) combining CAS predictive entropy gating, on-demand feature acquisition, and safety abstention.
3. **Section 3 (Offline Trust Diagnostics):** Offline model auditing (DSI explanation stability, CAS uncertainty decomposition, RAI headroom retention), explicitly excluded from runtime decision routing.
4. **Section 4 (Dual Independent Evaluation Regimes):** Two separately evaluated empirical protocols (the forward temporal survey-weighted study vs. the locked test adaptive benchmark).

> **Crucial Architectural Rule (Verified in Code):** As explicitly documented in [`truststroke/adaptive_layer.py:42-44`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/adaptive_layer.py#L42-L44):  
> *"DSI (attribution stability) and RAI (resource adaptability) are offline diagnostic metrics and are deliberately excluded from per-record online routing."*  
> Accordingly, the architecture diagram depicts DSI and RAI as offline diagnostic audits with dashed connections, never as active loops inside the runtime inference path.

---

## 2. Component Mapping & Code Evidence

### Tier 1: Population Data Ingestion & Survey Design
* **`Survey Sources` (`baseline.py:40-75`, `optimize.py:24-35`):** Ingests official CDC BRFSS releases. 2024 source contains 456,218 labeled respondents (20,661 stroke positive); 2025 source contains 354,950 labeled respondents (349,241 across 49 common jurisdictions).
* **`Preprocessing Pipeline` (`baseline.py:40-75`):** 
  - Target: Binary lifetime stroke history `CVDSTRK3` (1=Yes, 0=No). Dropping 1,452 invalid/refused responses. Base prevalence: $4.53\%$.
  - 15 Canonical Predictors: 4 categorical (`SEXVAR`, `EXERANY2`, `_SMOKER3`, `EDUCA`) and 11 numeric/ordinal (`_AGEG5YR`, `GENHLTH`, `DIABETE4`, `CVDINFR4`, `CVDCRHD4`, `CHCKDNY2`, `HAVARTH4`, `INCOME3`, `PHYSHLTH`, `MENTHLTH`, `_BMI5`).
  - Strict isolation: Median imputation and standard scaling are fitted exclusively on outer training splits.
* **`Complex Survey Design` (`survey.py:15-60`, `manuscript.tex:95-108`):** Final survey weights `_LLCPWT`, strata `_STSTR`, and clusters `_PSU` are preserved for design-based Taylor series linearization.

### Tier 2: Availability Scenarios & Core Estimators
* **`Availability Scenarios` (`availability_study.py:17-18`, `evaluate_adaptive.py:36-53`):** 
  - 4 predefined masks: Full (15), Without Health State (12), Demographics+History (7), Demographics Only (2).
  - Deliberately unavailable predictors have their feature columns zeroed and separate availability indicator flags set to 0.
* **`Estimator Families` (`baseline.py:130-180`, `availability_study.py:45-90`):**
  - Evaluates Histogram Gradient Boosting (`HistGradientBoostingClassifier`), Multi-Layer Perceptrons (MLP), and Logistic Regression.
  - Controls: Fixed Global (FG), Scenario Recalibration (SC), Mixed-Mask Training (MT), and Subset Retraining (SR).
* **`Scenario Recalibration` (`adaptive_layer.py:165-198`, `availability_study.py:75-88`):**
  - Applies Platt scaling: $P_{\text{cal}} = \sigma(A_s \cdot \text{logit}(p) + B_s)$ fitted on development data.
  - Resolves input representation mismatch without altering model weights.

### Tier 3: Runtime Adaptive Decision Policy (`AdaptiveDecisionPolicy.decide()`)
* **`Observed Input` (`adaptive_layer.py:252-255`):** Takes initial scenario dictionary $x_{\text{obs}}$. Unobserved features are strictly absent or concealed as `np.nan`.
* **`CAS Uncertainty Gate` (`adaptive_layer.py:260-265`):** Computes initial calibrated $p_{\text{mean}}$, $H_{\text{total}}$, and $H_{\text{between}}$. If $H_{\text{total}} < \tau_{\text{acq}}$ ($0.30\text{ bits}$) and $H_{\text{between}} < \tau_{\text{disagree}}$, skips acquisition directly to the safety gate.
* **`Sequential Recommender` (`adaptive_layer.py:271-292`, `acquisition.py:58-61`, `benchmark_acquisition.py:36-47`):**
  - Evaluates unobserved candidates via expected information gain $\Delta H$.
  - Supports dynamic donor mutual information (`ConditionalDonors`, 64-NN) or static clinical priority queues (`StaticFeatureRecommender`, querying `CVDINFR4`).
  - Stops if $\Delta H < \gamma_{\text{min}}$.
* **`On-Demand Query Callback` (`adaptive_layer.py:295-305`):**
  - Calls `ground_truth_provider(c*)`. True value is revealed strictly on-demand.
  - Updates $x_{\text{obs}} \leftarrow x_{\text{obs}} \cup \{c^*\}$ and recalculates $p_{\text{mean}}$ and uncertainty.
* **`Safety Abstention Gate` (`adaptive_layer.py:308-325`):**
  - Checks if final $H_{\text{total}} \ge \tau_{\text{abstain}}$ ($0.85\text{ bits}$) or $|p_{\text{mean}} - \theta| \le \delta_{\text{margin}}$.
  - If breached: emits `DecisionRecord(action='abstain', prediction=None)`.
  - If safe: emits `DecisionRecord(action='classify', prediction=\mathbb{I}(p \ge 0.14))`.

### Tier 4: Offline Trust Diagnostics & Dual Evaluation Framework
* **`Decision Stability (DSI)` (`trust.py:33-53`, `attribution_convergence.json`):** Evaluates Monte Carlo permutation Shapley attribution stability across 16, 64, and 256 paths ($0.736 \to 0.928$) and seed retraining ($0.890$).
* **`Confidence Attribution (CAS)` (`trust.py:17-30`, `trust.json`):** Decomposes entropy into within-ensemble and between-ensemble disagreement ($H_{\text{between}} = 0.000531$). Disagreement achieves misclassification AUROC of $0.830$ vs. $0.936$ for standard predictive entropy.
* **`Resource Adaptability (RAI)` (`trust.py:85-104`, `availability.json`):** Computes normalized Brier headroom retention $r_s$ across 7 scenarios (aggregate $\text{RAI} = 0.717$).
* **`Dual Evaluation Regimes`:**
  - **Regime A (Temporal Survey Transport, `manuscript.tex`):** 2024 $\to$ 2025 forward transport ($N=349,241$) with survey-weighted Taylor linearization (primary contrast $\hat\theta = -0.000157$; 93 singleton strata disclosed).
  - **Regime B (Adaptive Policy Benchmark, `evaluation_results.json`):** 2024 locked test partition ($N=91,244$) with unweighted paired bootstrap differences ($B=200$).

---

## 3. Verified Connections vs. Unresolved Discrepancies

### Fully Verified Connections (Solid Lines in Diagram)
1. **Raw Survey Data $\to$ Preprocessing $\to$ Availability Masks:** Confirmed in `baseline.py:clean()` and `availability_study.py:score_bundle()`.
2. **Estimator Output $\to$ Scenario Recalibration:** Confirmed in `availability_study.py` and `adaptive_layer.py:_predict_prob()`.
3. **Observed Feature $\to$ CAS Gate $\to$ Recommender $\to$ Query Callback $\to$ Safety Gate:** Confirmed in `adaptive_layer.py:decide()`.
4. **Safety Gate $\to$ Actions (`Classify` vs. `Abstain`):** Confirmed in `adaptive_layer.py:311-340`.

### Architectural Limitations & Nuances (Explicitly Disclosed in Notes)
1. **Post-Acquisition Calibrator Routing Fallback:** In `adaptive_layer.py:175-193`, when acquisition adds a feature (e.g., in `demographics`, adding `GENHLTH` gives 3 features), the policy falls back to the largest registered subset in `calibrator_map`, which routes back to the 2-feature `demographics` Platt calibrator. The diagram notes show that calibrators are scenario-routed, while disclosing this fallback mismatch in documentation.
2. **Offline Diagnostic Boundary:** DSI and RAI do not feed into runtime decision making; their connection is strictly diagnostic/auditing.
3. **Evaluation Regime Incompatibility:** The 2024 $\to$ 2025 temporal survey-weighted study (Regime A) and the 2024 unweighted adaptive evaluation (Regime B) are separate experimental designs and are depicted as complementary evaluation regimes rather than a single unified test set.
