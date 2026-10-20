# Figures and Tables Inventory & Publication Plan

---

## 1. Existing Figure Assets & Scripts

| File Path | Generation Script | Underlying Data Source | Visual Description | Publication Suitability |
|---|---|---|---|:---:|
| [`paper/figures/availability.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/figures/availability.pdf)<br>([`docs/experiments/availability_paper/availability.png`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/availability_paper/availability.png)) | [`truststroke/package_availability.py:63-71`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/package_availability.py#L63-L71) | `docs/experiments/availability_paper/transport2025.json`<br>([`all_metrics.csv`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/availability_paper/all_metrics.csv)) | **Two-panel line plot** showing survey-weighted Brier loss across 4 availability scenarios (15, 12, 7, 2 variables) for Boosted Trees (left) and MLP (right) across FG, SC, MT, and SR. | **High (Core Figure).** Clear, vector PDF, standard IEEE formatting. |
| [`paper/figures/calibration.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/figures/calibration.pdf)<br>([`docs/experiments/availability_paper/calibration.png`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/availability_paper/calibration.png)) | [`truststroke/package_availability.py:72-80`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/package_availability.py#L72-L80) | `docs/experiments/availability_paper/transport2025.json` (`calibration` bins) | **Reliability diagram / calibration curves** for Boosted Trees under Demographics+History (7 variables) comparing Global vs. Scenario calibration vs. Mask Training vs. Subset Refit. | **High (Core Figure).** Demonstrates probability calibration recovery. |
| [`paper/figures/subgroups.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/figures/subgroups.pdf)<br>([`docs/experiments/availability_paper/subgroups.png`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/availability_paper/subgroups.png)) | [`truststroke/package_availability.py:81-89`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/package_availability.py#L81-L89) | `docs/experiments/availability_paper/transport2025.json` (`subgroups`) | **Forest / error-bar plot** of domain point contrasts ($d_i = \text{MT} - \text{SC}$) across Sex, Age (18–44, 45–64, 65+), and Income (<$35k, $\ge$35k). | **Medium.** Good for demographic domain analysis; singleton strata suppress formal CIs. |
| [`docs/experiments/trust/trust_overview.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/trust/trust_overview.pdf)<br>([`docs/experiments/trust/trust_overview.png`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/trust/trust_overview.png)) | [`truststroke/plot_trust.py:10-50`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/plot_trust.py#L10-L50) | `docs/experiments/trust/attribution_convergence.json`<br>`docs/experiments/trust/trust.json`<br>`docs/experiments/trust/tabnet_development.json` | **Four-panel diagnostic plot:**<br>1. Top-left: DSI Monte Carlo path repeat stability (16, 64, 256 paths).<br>2. Top-right: RAI horizontal bar chart across 7 missingness scenarios.<br>3. Bottom-left: Acquisition pilot Brier loss curves.<br>4. Bottom-right: TabNet bounded search AP comparison. | **High (Diagnostic Milestone Figure).** Excellent for showing DSI convergence and RAI retention. |

---

## 2. Existing LaTeX Algorithm & Table Drafts

1. **Algorithm Pseudocode:** [`paper/adaptive_algorithm.tex`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/adaptive_algorithm.tex)
   * Formalizes `AdaptiveDecisionPolicy.decide()`: uncertainty evaluation via CAS, gated recommendation, on-demand feature query callback, and safety abstention. Ready for direct inclusion in the Proposed Work section.
2. **Comparative Results Table Template:** [`paper/results_table.tex`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/results_table.tex)
   * Two-column IEEE layout cleanly partitioning Operational Rates (Coverage, Abstention, Acquisition), Full-Population Probability Metrics (Brier, AUROC, AP, ECE), and Covered-Only Decision Metrics (Sensitivity, Specificity, Precision, F1).

---

## 3. Recommended Publication Set for the New IEEE Paper

To produce a self-contained, publication-ready conference paper, Claude should include the following **4 Figures** and **4 Tables**:

### Recommended Figures
* **Figure 1 (System Architecture & Pipeline):** Block diagram illustrating the end-to-end framework: (1) Preprocessing & missingness encoding, (2) Model training & Platt scenario recalibrators, (3) CAS uncertainty gate & DSI explanation stability, (4) Feature recommender (static priority vs. conditional donors), and (5) Decision/Abstention safety gate. *(Can be rendered via TikZ or imported from `paper/figures/`)*.
* **Figure 2 (Availability Shift & Recalibration Recovery):** Use [`paper/figures/availability.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/figures/availability.pdf). Demonstrates how scenario recalibration closes the majority of the degradation gap caused by naive masking across both Tree and MLP models.
* **Figure 3 (Reliability Diagrams / Calibration Curves):** Use [`paper/figures/calibration.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/figures/calibration.pdf). Shows calibration curves under the 7-variable scenario.
* **Figure 4 (DSI Convergence & Trust Diagnostics):** Use Panel (0,0) and (0,1) from [`docs/experiments/trust/trust_overview.pdf`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/trust/trust_overview.pdf) or a dedicated plot showing Monte Carlo Shapley stability rising to $>0.92$ at 256 paths.

### Recommended Tables
* **Table I (Cohort & Partition Inventory):** 2024 source ($N=456,218$), fitting ($60,000$), stopping ($10,000$), calibration ($45,622$), dev-eval ($15,000$), 2025 common-jurisdiction ($349,241$ across 49 states).
* **Table II (Core Temporal Transport Benchmark - 2025 Common Jurisdictions):** Primary survey-weighted results across Trees and MLP for Demographics+History (7 features), matching Table II of the core manuscript.
* **Table III (Adaptive Policy Evaluation on 2024 Locked Test Partition - $N=91,244$):** Complete populated results from `results/brfss_adaptive_full/evaluation_results.json` showing both full-population and covered metrics across all 4 scenarios.
* **Table IV (Feature Acquisition Benchmark - Dynamic vs. Static Heuristics):** Complete results from `results/acquisition_benchmark/benchmark_results.json` comparing Latency (ms/rec), Brier, AUROC, and AP in the Demographics scenario.
* **Table V (Trust Metrics Summary: DSI, CAS, RAI):** Exact metrics from `docs/experiments/trust/trust.json` and `attribution_convergence.json`.

---

## 4. Exact Numerical Source Data for Proposed Tables & Plots

### Data for Table II: 2025 Common-Jurisdiction Transport ($N=349,241$)
*Source: `docs/experiments/availability_paper/all_metrics.csv` (`cohort=transport2025_common`)*

```csv
family,scenario,condition,brier,average_precision,auroc,weighted_brier,weighted_average_precision,weighted_auroc
hist_gradient_boosting,full,fixed_global_calibration,0.038753,0.170668,0.806655,0.031541,0.175114,0.814238
hist_gradient_boosting,full,fixed_scenario_calibration,0.038753,0.170668,0.806655,0.031541,0.175114,0.814238
hist_gradient_boosting,without_health_state,fixed_global_calibration,0.039600,0.158229,0.785434,0.032230,0.160170,0.793739
hist_gradient_boosting,without_health_state,fixed_scenario_calibration,0.039324,0.158229,0.785434,0.031980,0.160170,0.793739
hist_gradient_boosting,without_health_state,mask_trained,0.039019,0.165213,0.795856,0.031737,0.168128,0.803876
hist_gradient_boosting,without_health_state,subset_retrained,0.038967,0.167823,0.795586,0.031707,0.170940,0.803527
hist_gradient_boosting,demographics_history,fixed_global_calibration,0.040608,0.144415,0.770020,0.033221,0.149023,0.804412
hist_gradient_boosting,demographics_history,fixed_scenario_calibration,0.040336,0.144415,0.770020,0.032994,0.149023,0.804412
hist_gradient_boosting,demographics_history,mask_trained,0.040180,0.148766,0.777085,0.032837,0.154378,0.811364
hist_gradient_boosting,demographics_history,subset_retrained,0.040103,0.151740,0.776606,0.032783,0.157712,0.810729
hist_gradient_boosting,demographics,fixed_global_calibration,0.045059,0.082729,0.702989,0.036660,0.091173,0.751682
hist_gradient_boosting,demographics,fixed_scenario_calibration,0.041697,0.082729,0.702989,0.034177,0.091173,0.751682
hist_gradient_boosting,demographics,mask_trained,0.041607,0.084534,0.707759,0.034091,0.093412,0.755088
hist_gradient_boosting,demographics,subset_retrained,0.041595,0.085002,0.708846,0.034079,0.093922,0.756184
mlp,demographics_history,fixed_global_calibration,0.045339,0.140880,0.770967,0.037215,0.144189,0.804791
mlp,demographics_history,fixed_scenario_calibration,0.040417,0.140880,0.770967,0.033022,0.144189,0.804791
mlp,demographics_history,mask_trained,0.040134,0.149258,0.778152,0.032789,0.154812,0.812105
mlp,demographics_history,subset_retrained,0.040087,0.151954,0.777855,0.032761,0.157820,0.811685
```

### Data for Table III: Populated Adaptive Policy Results ($N_{\text{test}} = 91,244$)
*Source: `results/brfss_adaptive_full/evaluation_results.json`*

```
Scenario: Full (15)
- No Intervention:     Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04024, AUROC=0.8067, AP=0.1731, ECE=0.0030 | n=91244, Sens=0.2904, Spec=0.9490, Prec=0.2126, F1=0.2455
- Fixed Recalibration: Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04026, AUROC=0.8067, AP=0.1731, ECE=0.0033 | n=91244, Sens=0.2945, Spec=0.9479, Prec=0.2115, F1=0.2462
- Adaptive Policy:     Cov=0.9783, Abst=0.0217, Acq=0.000 | Brier=0.04053, AUROC=0.8075, AP=0.1723, ECE=0.0048 | n=89266, Sens=0.2250, Spec=0.9547, Prec=0.1728, F1=0.1955
  Paired Diff (Adaptive - Fixed): +0.000269 (95% CI: [+0.000158, +0.000370])

Scenario: Without Health State (12)
- No Intervention:     Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04074, AUROC=0.7844, AP=0.1562, ECE=0.0092 | n=91244, Sens=0.2585, Spec=0.9515, Prec=0.2019, F1=0.2267
- Fixed Recalibration: Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04071, AUROC=0.7844, AP=0.1562, ECE=0.0025 | n=91244, Sens=0.2335, Spec=0.9578, Prec=0.2080, F1=0.2200
- Adaptive Policy:     Cov=0.9719, Abst=0.0281, Acq=0.3143 | Brier=0.04115, AUROC=0.7914, AP=0.1662, ECE=0.0094 | n=88677, Sens=0.2287, Spec=0.9476, Prec=0.1506, F1=0.1816
  Paired Diff (Adaptive - Fixed): +0.000438 (95% CI: [+0.000233, +0.000602])

Scenario: Demographics + History (7)
- No Intervention:     Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04114, AUROC=0.7665, AP=0.1405, ECE=0.0045 | n=91244, Sens=0.1576, Spec=0.9739, Prec=0.2226, F1=0.1845
- Fixed Recalibration: Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04107, AUROC=0.7665, AP=0.1405, ECE=0.0021 | n=91244, Sens=0.2064, Spec=0.9620, Prec=0.2049, F1=0.2057
- Adaptive Policy:     Cov=0.9809, Abst=0.0191, Acq=0.2728 | Brier=0.04089, AUROC=0.7752, AP=0.1576, ECE=0.0032 | n=89502, Sens=0.1938, Spec=0.9623, Prec=0.1805, F1=0.1869
  Paired Diff (Adaptive - Fixed): -0.000183 (95% CI: [-0.000310, -0.000061])

Scenario: Demographics (2)
- No Intervention:     Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04272, AUROC=0.6991, AP=0.0831, ECE=0.0151 | n=91244, Sens=0.0000, Spec=1.0000, Prec=0.0000, F1=0.0000
- Fixed Recalibration: Cov=1.000, Abst=0.000, Acq=0.000 | Brier=0.04233, AUROC=0.6991, AP=0.0831, ECE=0.0014 | n=91244, Sens=0.0000, Spec=1.0000, Prec=0.0000, F1=0.0000
- Adaptive Policy:     Cov=1.0000, Abst=0.0000, Acq=0.3592 | Brier=0.04183, AUROC=0.7196, AP=0.1093, ECE=0.0013 | n=91244, Sens=0.1459, Spec=0.9685, Prec=0.1801, F1=0.1612
  Paired Diff (Adaptive - Fixed): -0.000503 (95% CI: [-0.000614, -0.000394])
```

### Data for Table IV: Acquisition Benchmark in Demographics Scenario ($N_{\text{test}} = 1,000$)
*Source: `results/acquisition_benchmark/benchmark_results.json`*

| Strategy | Target Feature | Latency (ms/rec) | Acq. Freq. | Brier Score $\downarrow$ | AUROC $\uparrow$ | Average Precision $\uparrow$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| No Acquisition | None | 5.42 | 0.000 | 0.05662 | 0.6991 | 0.1604 |
| **ConditionalDonors** | Dynamic (64-NN) | 18.39 | 0.620 | 0.05518 | 0.7281 | 0.1845 |
| **Static CVDINFR4** | Prior Heart Attack | **8.13** | 0.623 | **0.05500** | **0.7455** | **0.1856** |
| **Static GENHLTH** | General Health | 10.65 | 0.626 | 0.05511 | 0.7313 | 0.1854 |

*Paired Bootstrap Contrasts ($B=100$):*
* $\text{ConditionalDonors} - \text{No Acquisition}$: $-0.001445$ (95% CI $[-0.002768, -0.000151]$) $\to$ Significant benefit over no acquisition.
* $\text{Static CVDINFR4} - \text{No Acquisition}$: $-0.001619$ (95% CI $[-0.002802, -0.000309]$) $\to$ Significant benefit over no acquisition.
* $\text{ConditionalDonors} - \text{Static CVDINFR4}$: $\mathbf{+0.000174}$ (95% CI $[-0.001437, +0.001836]$) $\to$ ConditionalDonors does not outperform static heuristic.
