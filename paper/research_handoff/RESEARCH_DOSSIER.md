# TrustStroke Research Dossier
**Project:** Trust-Aware Explainable AI Framework for Stroke Risk Assessment  
**Data Source:** Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS)  
**Endpoint Target:** Self-reported lifetime history of stroke (`CVDSTRK3`), **NOT** incident stroke prediction or acute clinical diagnosis.

---

## 1. Project Motivation & Research Questions

### 1.1 Motivation & Clinical Context
Machine learning models trained on survey questionnaires can evaluate population-level disease patterns, but their deployment reliability degrades when input variables are missing or withheld (e.g., in shortened telephone interviews or triage kiosks). Replacing unavailable variables with zero or imputed values causes a mismatch between training and deployment input representations, conflating **information loss** (the feature is absent) with **representation mismatch** (the model receives out-of-distribution inputs). 

Furthermore, simply computing an accuracy or ROC score fails to capture whether a model's predicted probabilities are calibrated, whether feature attributions are numerically stable across retrainings, whether ensemble disagreement signals actionable uncertainty, or whether acquiring additional questions justifies the inquiry cost.

### 1.2 Core Research Questions
1. **Recalibration vs. Retraining:** Under predefined questionnaire availability scenarios (15, 12, 7, and 2 variables), how much predictive degradation is corrected purely by post-hoc scenario recalibration versus mask-conditioned retraining or subset refitting under forward temporal transport (2024 $\to$ 2025) with complex survey weights?
2. **Attribution Stability (DSI):** How stable are Monte Carlo permutation Shapley explanations across random initialization seeds, training bootstrap samples, and explanation-path sampling budgets?
3. **Uncertainty Decomposition (CAS):** Does decomposing predictive entropy into expected member entropy (aleatoric proxy) and ensemble disagreement (epistemic proxy) provide incremental diagnostic value for error detection beyond standard predictive entropy?
4. **Resource Robustness (RAI):** How can performance retention across diverse missingness patterns be quantified against a constant-prevalence baseline?
5. **Adaptive Sequential Acquisition & Abstention:** Can a gated decision policy dynamically query informative features via conditional donor distributions to recover discriminative power, and does it outperform static domain heuristics?

---

## 2. Mathematical Formulations & Symbol Definitions

### 2.1 Predictive Entropy & Confidence Attribution Scheme (CAS)
Given an ensemble of $M$ binary classifiers $\{f_m\}_{m=1}^M$ outputting probabilities $p_m(x) \in [0, 1]$:
* **Ensemble Mean Probability:**
  $$\bar{p}(x) = \frac{1}{M} \sum_{m=1}^M p_m(x)$$
* **Binary Shannon Entropy (in bits):**
  $$H(p) = -p \log_2(p) - (1-p) \log_2(1-p), \quad H(0) = H(1) = 0$$
* **Predictive Total Uncertainty ($H_{\text{total}}$):**
  $$H_{\text{total}}(x) = H(\bar{p}(x))$$
* **Expected Member Entropy (Within-Ensemble Uncertainty, $H_{\text{within}}$):**
  $$H_{\text{within}}(x) = \frac{1}{M} \sum_{m=1}^M H(p_m(x))$$
* **Ensemble Disagreement (Between-Ensemble Uncertainty, $H_{\text{between}}$):**
  $$H_{\text{between}}(x) = H_{\text{total}}(x) - H_{\text{within}}(x) = H\left(\frac{1}{M} \sum_{m=1}^M p_m(x)\right) - \frac{1}{M} \sum_{m=1}^M H(p_m(x)) \ge 0$$
  *(By Jensen's inequality applied to concave function $H$.)*
* **Entropy Confidence:**
  $$\text{Conf}(x) = 1 - H_{\text{total}}(x) \in [0, 1]$$

### 2.2 Decision Stability Index (DSI)
Given two feature attribution vectors $a, b \in \mathbb{R}^d$ computed on the original feature space:
$$\text{DSI}(a, b) = 1 - \frac{1}{2} \left\| \frac{a}{\|a\|_1} - \frac{b}{\|b\|_1} \right\|_1 \in [0, 1]$$
* **Boundary Conventions:** If $\max(\|a\|_1, \|b\|_1) < 10^{-12}$, DSI is undefined (`None`, uninformative); if only one vector is all-zero, $\text{DSI} = 0.0$.
* **Permutation Shapley Attributions:** Probability-scale attributions $a_j(x)$ computed via $K$ random permutations $\pi \in S_d$ using background donors $B$:
  $$a_j(x) = \frac{1}{K} \sum_{k=1}^K \left[ f(x_{S_j^\pi \cup \{j\}}) - f(x_{S_j^\pi}) \right]$$
  where $S_j^\pi$ is the set of features preceding $j$ in permutation $\pi$.

### 2.3 Resource Adaptability Index (RAI)
Let $B_{\text{full}}$ be the Brier score of the full-information model, $B_s$ the Brier score under availability scenario $s \in \mathcal{S}$, and $B_{\text{null}} = \bar{y}(1 - \bar{y})$ the Brier score of the constant training-prevalence predictor.
* **Per-Scenario Headroom Retention ($r_s$):**
  $$r_s = 1 - \text{clip}\left(\frac{B_s - B_{\text{full}}}{B_{\text{null}} - B_{\text{full}}}, 0, 1\right) \in [0, 1]$$
* **Aggregate RAI:**
  $$\text{RAI} = \sum_{s \in \mathcal{S}} w_s r_s, \quad \sum_{s \in \mathcal{S}} w_s = 1, \quad w_s \ge 0$$
  *(If $B_{\text{null}} \le B_{\text{full}} + 10^{-12}$, RAI is undefined due to zero headroom.)*

### 2.4 Scenario Recalibration (Platt Scaling)
For scenario $s$ with observed predictor subset $x_s$:
$$\text{logit}(p) = \ln\left(\frac{\text{clip}(p, \epsilon, 1-\epsilon)}{1 - \text{clip}(p, \epsilon, 1-\epsilon)}\right), \quad \epsilon = 10^{-6}$$
$$P_{\text{cal}}(Y=1 \mid x_s) = \sigma(A_s \cdot \text{logit}(f(x_s)) + B_s) = \frac{1}{1 + \exp(-(A_s \cdot \text{logit}(f(x_s)) + B_s))}$$
Fitted using unweighted logistic regression on the held-out development partition ($C=1.0$).

### 2.5 Survey-Weighted Contrast & Taylor Series Linearization
For 2025 evaluation respondent $i$ in stratum $h$ and primary sampling unit (PSU) $g$:
* **Respondent Loss Difference:**
  $$d_i = (y_i - \bar{p}_{i,\text{MT}})^2 - (y_i - \bar{p}_{i,\text{SC}})^2$$
  *(Negative values favor mixed-mask training $\text{MT}$.)*
* **Survey-Weighted Mean Contrast:**
  $$\hat\theta = \frac{\sum_i w_i d_i}{\sum_i w_i}$$
  where $w_i$ is the official CDC final survey weight (`_LLCPWT`).
* **Stratified Multistage Variance Estimator:**
  $$t_{hg} = \frac{\sum_{i \in (h, g)} w_i (d_i - \hat\theta)}{\sum_i w_i}, \quad \bar{t}_h = \frac{1}{m_h} \sum_{g=1}^{m_h} t_{hg}$$
  $$\widehat V(\hat\theta) = \sum_{h=1}^H \frac{m_h}{m_h - 1} \sum_{g=1}^{m_h} (t_{hg} - \bar{t}_h)^2$$
  where $m_h$ is the number of sampled PSUs in stratum $h$. If any stratum contains $m_h = 1$ (singleton stratum), $m_h - 1 = 0$ and the prespecified variance is strictly undefined.

### 2.6 Dynamic Information-Gain Acquisition (`ConditionalDonors`)
Conditioned on observed features $x_{\text{obs}}$, retrieve the $k=64$ nearest complete training donors using mixed distance:
$$\text{dist}(x_{\text{obs}}, d) = \sum_{c \in \text{Cat}} \mathbb{I}(x_c \ne d_c) + \sum_{j \in \text{Num}} \min\left(1, \frac{|x_j - d_j|}{\text{IQR}_j}\right)$$
Given donor completion predictions $\{p_l\}_{l=1}^k$ with mean $\bar{p}$, candidate feature $c^*$ is chosen to maximize expected Jensen-gap information gain:
$$c^* = \arg\max_{c \notin x_{\text{obs}}} \left[ H(\bar{p}) - \sum_{v \in \text{Bins}(c)} \frac{k_v}{k} H(\bar{p}_v) \right]$$

---

## 3. Dataset Provenance, Coding, and Partitions

### 3.1 Datasets
* **2024 Source:** CDC BRFSS 2024 survey (`LLCP2024.XPT`, SHA-256 `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c`). Total records: 457,670; valid binary stroke history: **456,218** (20,661 positive, prevalence **4.5285%**; 1,452 dropped due to missing/refused outcome).
* **2025 Source:** CDC BRFSS 2025 survey (`LLCP2025.XPT`, SHA-256 `d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2`). Total records: 356,158; valid labels: **354,950** (16,353 positive, prevalence **4.6071%**; 1,208 dropped).
* **2025 Common-Jurisdiction Primary Cohort:** Restricting to the 49 jurisdictions present in both 2024 and 2025 yields **349,241** respondents (16,082 positive, weighted prevalence **3.49%**).

### 3.2 Canonical 15 Predictor Variables
1. `SEXVAR` (Sex: 1=Male, 2=Female)
2. `_AGEG5YR` (14 five-year age categories, 1=18-24 to 14=80+)
3. `GENHLTH` (General health: 1=Excellent to 5=Poor)
4. `EXERANY2` (Exercise in past 30 days: 1=Yes, 2=No)
5. `_SMOKER3` (Smoking status: 1=Current every day, 2=Some days, 3=Former, 4=Never)
6. `DIABETE4` (Diabetes: 1=Yes, 2=Yes pregnancy, 3=No, 4=Borderline)
7. `CVDINFR4` (History of myocardial infarction: 1=Yes, 2=No)
8. `CVDCRHD4` (Coronary heart disease/angina: 1=Yes, 2=No)
9. `CHCKDNY2` (Kidney disease: 1=Yes, 2=No)
10. `HAVARTH4` (Arthritis diagnosis: 1=Yes, 2=No)
11. `EDUCA` (Education level: 1=Never/kindergarten to 6=College 4+ years)
12. `INCOME3` (Household income: 11 categories; 77=Don't know, 99=Refused retained as separate categories)
13. `PHYSHLTH` (Days physical health not good in past 30 days: 1-30; 88=None recoded to 0)
14. `MENTHLTH` (Days mental health not good in past 30 days: 1-30; 88=None recoded to 0)
15. `_BMI5` (Body Mass Index: continuous with two implied decimal places)

### 3.3 Four Predefined Availability Scenarios
* **Full (15 features):** All 15 canonical predictors.
* **Without Health State (12 features):** Excludes contemporaneous state variables `GENHLTH`, `PHYSHLTH`, `MENTHLTH`.
* **Demographics + History (7 features):** `SEXVAR`, `_AGEG5YR`, `DIABETE4`, `CVDINFR4`, `CVDCRHD4`, `CHCKDNY2`, `HAVARTH4`.
* **Demographics (2 features):** `SEXVAR`, `_AGEG5YR`.

---

## 4. Distinct Experimental Setups & Exact Results

### 4.1 Study 1: Temporal Survey-Weighted Availability Study (Core Manuscript)
* **Execution Script:** [`truststroke/run_availability_study.py`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/run_availability_study.py)
* **Artifact Files:** [`docs/experiments/availability_paper/all_metrics.csv`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/availability_paper/all_metrics.csv), [`docs/experiments/availability_paper/transport2025.json`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/availability_paper/transport2025.json)
* **Cohort:** Fitted on 2024 development ($N_{\text{fit}}=60,000$, $N_{\text{stop}}=10,000$, $N_{\text{cal}}=45,622$); evaluated on 2025 common-jurisdiction ($N=349,241$).
* **Model Families:** Histogram Gradient Boosting Trees (`HistGradientBoostingClassifier`, max 200 trees, lr=0.05, 31 leaves) and Multi-Layer Perceptron (MLP, 128 $\to$ 64 ReLU, dropout 0.1, AdamW, batch 1024). Three seeds (42, 43, 44) per family.
* **Four Controls:** FG (Fixed Global calibration), SC (Fixed Scenario recalibration), MT (Mixed-Mask Training), SR (Subset Retraining).

#### Exact Results: 2025 Common-Jurisdiction Weighted Primary Comparison (7 Features: Demographics + History)
*Source: `all_metrics.csv`, `scenario=demographics_history`, `cohort=transport2025_common`*

| Family | Procedure | Brier Score $\downarrow$ | Average Precision (AP) $\uparrow$ | AUROC $\uparrow$ |
|---|---|:---:|:---:|:---:|
| **Trees** | FG (Fixed Global) | 0.0332212 | 0.149023 | 0.804412 |
| **Trees** | SC (Scenario Recal.) | 0.0329937 | 0.149023 | 0.804412 |
| **Trees** | MT (Mixed-Mask Train) | **0.0328371** | 0.154378 | **0.811364** |
| **Trees** | SR (Subset Retrain) | 0.0327834 | 0.157712 | 0.810729 |
| **MLP** | FG (Fixed Global) | 0.0372145 | 0.144189 | 0.804791 |
| **MLP** | SC (Scenario Recal.) | 0.0330218 | 0.144189 | 0.804791 |
| **MLP** | MT (Mixed-Mask Train) | **0.0327891** | 0.154812 | **0.812105** |
| **MLP** | SR (Subset Retrain) | 0.0327614 | 0.157820 | 0.811685 |

* **Primary Contrast (Trees MT minus SC):**  
  $\hat\theta = -0.000156566$ (Brier reduction of $\approx 0.48\%$).
* **Singleton Strata Disclosure:** 93 of 2,034 strata contain $m_h = 1$ PSU. The prespecified Taylor linearized interval is strictly omitted. Post-hoc assumption-based sensitivities: Adjust 95% CI: $[-0.000254, -0.000059]$; Average 95% CI: $[-0.000256, -0.000057]$.
* **Post-Hoc Fitting-Sample Robustness:** Three independent 60,000 draws from 2024 fitting pool give contrasts: $-0.000163$, $-0.000040$, $-0.000175$.

---

### 4.2 Study 2: Trust Metric Diagnostic Pilots (DSI, CAS, RAI)
* **Execution Script:** [`truststroke/trust.py`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/trust.py), [`truststroke/trust_experiment.py`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/trust_experiment.py)
* **Artifact Files:** [`docs/experiments/trust/trust.json`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/trust/trust.json), [`docs/experiments/trust/attribution_convergence.json`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/trust/attribution_convergence.json), [`docs/experiments/trust/availability.json`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/docs/experiments/trust/availability.json)
* **Cohort:** 2,048 natural-prevalence development records from 2024. Model: 5-member MLP ensemble (3 random seeds 42–44, 2 bootstrap resamples 45–46).

#### Exact Diagnostic Results:
1. **DSI (Decision Stability Index):**
   * *Monte Carlo Sampling Repeat Stability (Same model, 128 explanation records):*
     * 16 permutations: $\text{Mean DSI} = \mathbf{0.7365}$ (95% CI $[0.7212, 0.7509]$)
     * 64 permutations: $\text{Mean DSI} = \mathbf{0.8538}$ (95% CI $[0.8445, 0.8624]$)
     * 256 permutations: $\text{Mean DSI} = \mathbf{0.9278}$ (95% CI $[0.9232, 0.9322]$)
   * *Retrained Model Explanation Stability (at 256 permutations):*
     * Three random seeds: $\text{DSI} = \mathbf{0.8902}$, Absolute-rank Spearman = $\mathbf{0.8260}$, Top-5 Jaccard = $\mathbf{0.7375}$
     * Bootstrap samples: $\text{DSI} = \mathbf{0.8557}$, Absolute-rank Spearman = $\mathbf{0.7974}$, Top-5 Jaccard = $\mathbf{0.6952}$
2. **CAS (Confidence Attribution Scheme):**
   * Mean Predictive Entropy: $0.215213\text{ bits}$
   * Mean Member Entropy ($H_{\text{within}}$): $0.214682\text{ bits}$
   * Mean Ensemble Disagreement ($H_{\text{between}}$): $0.000531\text{ bits}$
   * Error Detection AUROC (at threshold $\theta = 0.14$):
     * Standard predictive entropy $H_{\text{total}}$: $\text{AUROC} = \mathbf{0.9365}$
     * Ensemble disagreement $H_{\text{between}}$: $\text{AUROC} = \mathbf{0.8302}$
   * *Key Finding:* Conventional predictive entropy detects misclassifications substantially better than ensemble disagreement.
3. **RAI (Resource Adaptability Index):**
   * Full-model Brier $B_{\text{full}} = 0.030689$; null baseline $B_{\text{null}} = 0.033135$.
   * Aggregate 7-Scenario $\text{RAI} = \mathbf{0.717437}$.
   * Scenario Headroom Retention: Demographics = $0.3622$, Demographics+History = $0.5590$, Without Health State = $0.6171$, MCAR 10% = $0.9309$, MCAR 30% = $0.9951$, MCAR 50% = $0.6135$, Age-Biased BMI Missing = $0.9467$.

---

### 4.3 Study 3: Adaptive Decision Policy (Full 2024 Evaluation)
* **Execution Script:** [`truststroke/evaluate_adaptive.py`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/evaluate_adaptive.py)
* **Artifact File:** [`results/brfss_adaptive_full/evaluation_results.json`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/results/brfss_adaptive_full/evaluation_results.json)
* **Cohort:** 2024 BRFSS stratified 60/20/20 split ($N_{\text{train}}=273,730$, $N_{\text{dev}}=91,244$, locked $N_{\text{test}}=\mathbf{91,244}$). Model: Logistic regression base classifier.
* **Tuned Policy Hyperparameters (on Dev):** Acquisition trigger $\tau_{\text{acq}} = 0.30\text{ bits}$, Abstention trigger $\tau_{\text{abstain}} = 0.85\text{ bits}$, Max steps $K_{\max} = 1$, Decision threshold $\theta = 0.14$. Replicates: 200.

#### Exact Results Across All Four Scenarios ($N_{\text{test}} = 91,244$):

| Scenario | Policy | Coverage | Abstain | Acq. Freq. | Full Brier $\downarrow$ | Full AUROC $\uparrow$ | Full AP $\uparrow$ | Full ECE $\downarrow$ | Covered Sens $\uparrow$ | Covered Spec $\uparrow$ | Covered F1 $\uparrow$ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full** (15) | No Intervention | 1.0000 | 0.0000 | 0.0000 | 0.04024 | 0.8067 | 0.1731 | 0.0030 | 0.2904 | 0.9490 | 0.2455 |
| | Fixed Recal. | 1.0000 | 0.0000 | 0.0000 | 0.04026 | 0.8067 | 0.1731 | 0.0033 | 0.2945 | 0.9479 | 0.2462 |
| | **Adaptive Policy** | **0.9783** | **0.0217** | **0.0000** | **0.04053** | **0.8075** | **0.1723** | **0.0048** | **0.2250** | **0.9547** | **0.1955** |
| | *Paired Diff* | \multicolumn{11}{c}{$\text{Adaptive} - \text{Fixed Brier} = \mathbf{+0.000269}$ (95% CI: $[\mathbf{+0.000158}, \mathbf{+0.000370}]$) $\to$ **Adaptive Statistically Worse**} |
| **Without Health State** (12) | No Intervention | 1.0000 | 0.0000 | 0.0000 | 0.04074 | 0.7844 | 0.1562 | 0.0092 | 0.2585 | 0.9515 | 0.2267 |
| | Fixed Recal. | 1.0000 | 0.0000 | 0.0000 | 0.04071 | 0.7844 | 0.1562 | 0.0025 | 0.2335 | 0.9578 | 0.2200 |
| | **Adaptive Policy** | **0.9719** | **0.0281** | **0.3143** | **0.04115** | **0.7914** | **0.1662** | **0.0094** | **0.2287** | **0.9476** | **0.1816** |
| | *Paired Diff* | \multicolumn{11}{c}{$\text{Adaptive} - \text{Fixed Brier} = \mathbf{+0.000438}$ (95% CI: $[\mathbf{+0.000233}, \mathbf{+0.000602}]$) $\to$ **Adaptive Statistically Worse**} |
| **Demogr. + History** (7) | No Intervention | 1.0000 | 0.0000 | 0.0000 | 0.04114 | 0.7665 | 0.1405 | 0.0045 | 0.1576 | 0.9739 | 0.1845 |
| | Fixed Recal. | 1.0000 | 0.0000 | 0.0000 | 0.04107 | 0.7665 | 0.1405 | 0.0021 | 0.2064 | 0.9620 | 0.2057 |
| | **Adaptive Policy** | **0.9809** | **0.0191** | **0.2728** | **0.04089** | **0.7752** | **0.1576** | **0.0032** | **0.1938** | **0.9623** | **0.1869** |
| | *Paired Diff* | \multicolumn{11}{c}{$\text{Adaptive} - \text{Fixed Brier} = \mathbf{-0.000183}$ (95% CI: $[\mathbf{-0.000310}, \mathbf{-0.000061}]$) $\to$ **Adaptive Statistically Better**} |
| **Demographics** (2) | No Intervention | 1.0000 | 0.0000 | 0.0000 | 0.04272 | 0.6991 | 0.0831 | 0.0151 | 0.0000 | 1.0000 | 0.0000 |
| | Fixed Recal. | 1.0000 | 0.0000 | 0.0000 | 0.04233 | 0.6991 | 0.0831 | 0.0014 | 0.0000 | 1.0000 | 0.0000 |
| | **Adaptive Policy** | **1.0000** | **0.0000** | **0.3592** | **0.04183** | **0.7196** | **0.1093** | **0.0013** | **0.1459** | **0.9685** | **0.1612** |
| | *Paired Diff* | \multicolumn{11}{c}{$\text{Adaptive} - \text{Fixed Brier} = \mathbf{-0.000503}$ (95% CI: $[\mathbf{-0.000614}, \mathbf{-0.000394}]$) $\to$ **Adaptive Statistically Better**} |

---

### 4.4 Study 4: Feature Acquisition Benchmark (Dynamic vs. Static Heuristics)
* **Execution Script:** [`truststroke/benchmark_acquisition.py`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/benchmark_acquisition.py)
* **Artifact File:** [`results/acquisition_benchmark/benchmark_results.json`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/results/acquisition_benchmark/benchmark_results.json)
* **Cohort:** 2024 BRFSS ($N_{\text{total}}=5,000$, $N_{\text{test}}=1,000$). Comparison of 4 strategies: No Acquisition, `ConditionalDonors`, Static `CVDINFR4`, Static `GENHLTH`. Replicates: 100.

#### Exact Results in `Demographics` Scenario ($N_{\text{test}} = 1,000$):
* **No Acquisition:** Brier = $0.05662$, AUROC = $0.6991$, Latency = $5.42\text{ ms/record}$, Acq. Freq. = $0.000$
* **ConditionalDonors:** Brier = $0.05518$, AUROC = $0.7281$, Latency = **$18.39\text{ ms/record}$**, Acq. Freq. = $0.620$
* **Static CVDINFR4:** Brier = **$0.05500$**, AUROC = **$0.7455$**, Latency = **$8.13\text{ ms/record}$**, Acq. Freq. = $0.623$
* **Static GENHLTH:** Brier = $0.05511$, AUROC = $0.7313$, Latency = $10.65\text{ ms/record}$, Acq. Freq. = $0.626$
* **Paired Difference ($\text{ConditionalDonors} - \text{Static CVDINFR4}$):**  
  $\text{Mean Diff} = \mathbf{+0.000174}$ (95% CI: $[-0.001437, +0.001836]$).  
  *Finding:* Static querying of prior heart attack (`CVDINFR4`) outperforms dynamic donor mutual information in AUROC ($+0.0174$) and Brier score at less than half the computational latency ($8.13\text{ ms}$ vs. $18.39\text{ ms}$).

---

## 5. Critical Limitations, Negative Findings, and Unresolved Nuances

1. **Symmetric Shannon Entropy in Rare Diseases (The Abstention Defect):**
   * Binary entropy peaks at $p = 0.5$. In BRFSS stroke history (prevalence $4.5\%$), predicted probabilities $p \in [0.28, 0.72]$ trigger the $\tau_{\text{abstain}} = 0.85$ safety gate.
   * In the `full` scenario, this abstains on 1,978 respondents containing **532 confirmed stroke cases** (a positive prevalence of **26.9%**). Rather than filtering uninformative epistemic noise, the symmetric entropy gate discards $12.8\%$ of all stroke cases in the test cohort, driving covered sensitivity down from $29.45\%$ to $22.50\%$.
2. **Post-Acquisition Calibrator Fallback Mismatch:**
   * Calibrators are fitted only for the 4 predefined scenario subsets (15, 12, 7, 2 features).
   * When an adaptive step acquires a feature in `demographics` (yielding 3 observed features: `SEXVAR`, `_AGEG5YR`, `GENHLTH`), the policy falls back to the largest registered subset, which is the 2-feature `demographics` calibrator. Logits from a 3-feature model are passed through a 2-feature Platt scaler, distorting post-acquisition probabilities and worsening ECE in `without_health_state` ($0.0025 \to 0.0094$).
3. **Temporal Survey Weighting vs. I.I.D. Splits:**
   * The core manuscript study ([`manuscript.tex`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/paper/manuscript.tex)) is survey-weighted with cluster-robust design variance across 2024 $\to$ 2025.
   * The adaptive evaluation ([`evaluate_adaptive.py`](file:///c:/Users/jorda/OneDrive/Desktop/sem 7/SM/Project/truststroke/evaluate_adaptive.py)) is unweighted I.I.D. random sampling on 2024. These two sets of numbers must never be merged or averaged into a single table.
