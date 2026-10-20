# References and Empirical Claims Mapping

---

## 1. Verified Bibliography Entries & Supported Claims

```bibtex
@misc{cdc24,
  title = {2024 BRFSS Survey Data and Documentation},
  author = {{Centers for Disease Control and Prevention}},
  year = {2025},
  url = {https://www.cdc.gov/brfss/annual_data/annual_2024.html},
  note = {Official public data release. LLCP2024.XPT SHA-256: 7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c}
}
```
* **Supported Claims:** Official 2024 BRFSS source dataset, sampling methodology, survey weight `_LLCPWT`, stratum `_STSTR`, cluster `_PSU`, and 15 predictor variable definitions.

```bibtex
@misc{cdc25,
  title = {2025 BRFSS Survey Data and Documentation},
  author = {{Centers for Disease Control and Prevention}},
  year = {2026},
  url = {https://www.cdc.gov/brfss/annual_data/annual_2025.html},
  note = {Official public data release. LLCP2025.XPT SHA-256: d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2}
}
```
* **Supported Claims:** Official 2025 BRFSS forward temporal transport dataset and variable coding verification.

```bibtex
@article{ding2025,
  title = {Cross-sectional study on smoking types and stroke risk: development of a predictive model for identifying stroke risk},
  author = {Ding, Chao and Yuan, Minjia and Cheng, Jiwei and Wen, Junkai},
  journal = {Frontiers in Physiology},
  volume = {16},
  pages = {1528910},
  year = {2025},
  doi = {10.3389/fphys.2025.1528910},
  url = {https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1528910/full}
}
```
* **Supported Claims:** Primary domain comparison for BRFSS-based stroke-history classification using conventional ML models and post-hoc SHAP explanations. Demonstrates that survey models classify self-reported history rather than incident stroke.

```bibtex
@article{moulaei2024,
  title = {Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models},
  author = {Moulaei, Khadijeh and Afshari, Lida and Moulaei, Reza and Sabet, Babak and Mousavi, Seyed Mohammad and Afrash, Mohammad Reza},
  journal = {Scientific Reports},
  volume = {14},
  pages = {31392},
  year = {2024},
  doi = {10.1038/s41598-024-82931-5},
  url = {https://www.nature.com/articles/s41598-024-82931-5}
}
```
* **Supported Claims:** Prior art comparing clinical ML/DL with XAI. Demonstrates that headline accuracy/AUROC scores in specialized hospital cohorts cannot be directly compared to national survey screening.

```bibtex
@inproceedings{realmlp,
  title = {Better by default: Strong pre-tuned MLPs and boosted trees on tabular data},
  author = {Holzm{\"u}ller, David and Grinsztajn, L{\'e}o and Steinwart, Ingo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume = {37},
  year = {2024},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html}
}
```
* **Supported Claims:** Justifies selecting competitive boosted trees (HistGradientBoosting) and properly tuned MLPs as strong tabular baselines over unvalidated complex neural architectures.

```bibtex
@article{janssen,
  title = {Dealing with missing predictor values when applying clinical prediction models},
  author = {Janssen, K. J. M. and Vergouwe, Y. and Donders, A. R. T. and Harrell, F. E. and Chen, Q. and Grobbee, D. E. and Moons, K. G. M.},
  journal = {Clinical Chemistry},
  volume = {55},
  number = {5},
  pages = {994--1001},
  year = {2009},
  doi = {10.1373/clinchem.2008.115345}
}
```
* **Supported Claims:** Foundational methodological comparator for unavailable predictors, establishing submodels and calibration intercept/slope assessment.

```bibtex
@article{mercaldo,
  title = {Missing data and prediction: the pattern submodel},
  author = {Fletcher-Mercaldo, S. and Blume, J. D.},
  journal = {Biostatistics},
  volume = {21},
  number = {2},
  pages = {236--252},
  year = {2020},
  doi = {10.1093/biostatistics/kxy040}
}
```
* **Supported Claims:** Methodological basis for pattern submodels fitted under specific missingness sets. Distinguishes our subset refitting (using all eligible fitting data) from pattern submodels fitted only within observed missingness subsets.

```bibtex
@article{hoogland,
  title = {Handling missing predictor values when validating and applying a prediction model to new patients},
  author = {Hoogland, J. and IntHout, J. and Vergouwe, Y. and Rovers, M. M. and Debray, T. P. A.},
  journal = {Statistics in Medicine},
  volume = {39},
  pages = {3591--3607},
  year = {2020},
  doi = {10.1002/sim.8682}
}
```
* **Supported Claims:** Compares observed-predictor submodels, marginalization, and imputation for deployment to new patients.

```bibtex
@article{sisk,
  title = {Imputation and missing indicators for handling missing data in the development and deployment of clinical prediction models: A simulation study},
  author = {Sisk, R. and Collins, G. S. and Lanyon, P. and Grainge, M. and Abhishek, A.},
  journal = {Statistical Methods in Medical Research},
  volume = {32},
  number = {8},
  pages = {1461--1477},
  year = {2023},
  doi = {10.1177/09622802231165001}
}
```
* **Supported Claims:** Methodological analysis of missingness indicators and imputation across model development and deployment.

```bibtex
@inproceedings{zhou,
  title = {Domain adaptation under missingness shift},
  author = {Zhou, H. and Balakrishnan, S. and Lipton, Z.},
  booktitle = {Proc. AISTATS},
  volume = {206},
  pages = {9577--9606},
  year = {2023}
}
```
* **Supported Claims:** Theoretical formalization of domain adaptation when feature availability changes between development and deployment.

```bibtex
@article{cooray,
  title = {Temporal validation and updating of a prediction model for the diagnosis of gestational diabetes mellitus},
  author = {Cooray, S. D. and et al.},
  journal = {Journal of Clinical Epidemiology},
  volume = {164},
  pages = {54--64},
  year = {2023},
  doi = {10.1016/j.jclinepi.2023.08.020}
}
```
* **Supported Claims:** Clinical epidemiology prior art for temporal validation across sequential years and staged recalibration updates.

```bibtex
@article{tripod,
  title = {TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods},
  author = {Collins, G. S. and et al.},
  journal = {BMJ},
  volume = {385},
  pages = {e078378},
  year = {2024},
  doi = {10.1136/bmj-2023-078378}
}
```
* **Supported Claims:** Reporting integrity standard informing our cohort documentation, transparent limitation disclosure, and separation of development from temporal evaluation.

```bibtex
@inproceedings{meyer2023,
  title = {On Minimizing the Impact of Dataset Shifts on Actionable Explanations},
  author = {Meyer, Anna P. and Ley, Dan and Srinivas, Suraj and Lakkaraju, Himabindu},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)},
  volume = {216},
  pages = {1434--1444},
  year = {2023},
  url = {https://proceedings.mlr.press/v216/meyer23a.html}
}
```
* **Supported Claims:** Direct methodological predecessor for explanation stability (DSI). Proves that explanation sensitivity across model retraining is an established research problem, not an original invention of this project.

```bibtex
@inproceedings{lakshminarayanan2017,
  title = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  author = {Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume = {30},
  year = {2017},
  url = {https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html}
}
```
* **Supported Claims:** Foundational prior art for the Confidence Attribution Scheme (CAS), establishing ensemble entropy decomposition into predictive entropy and disagreement.

```bibtex
@inproceedings{lundberg2017,
  title = {A Unified Approach to Interpreting Model Predictions},
  author = {Lundberg, Scott M. and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume = {30},
  year = {2017},
  url = {https://proceedings.neurips.cc/paper/7062-a-unified-approach-tointerpreting-model-predictions.pdf}
}
```
* **Supported Claims:** Theoretical definition of Shapley additive feature attributions and efficiency/additivity axioms.

```bibtex
@inproceedings{covert2023,
  title = {Learning to Maximize Mutual Information for Dynamic Feature Selection},
  author = {Covert, Ian Connick and Qiu, Wei and Lu, Mingyu and Kim, Na Yoon and White, Nathan J. and Lee, Su-In},
  booktitle = {International Conference on Machine Learning (ICML)},
  volume = {2023},
  pages = {6424--6447},
  year = {2023},
  url = {https://proceedings.mlr.press/v202/covert23a.html}
}
```
* **Supported Claims:** Prior art for dynamic information acquisition; establishes mutual information maximization for sequential feature selection.

---

## 2. Claim-to-Source-Artifact Mapping

| Scientific / Quantitative Claim | Exact Numerical Value | Source Artifact File | Exact Key / Column |
|---|---|---|---|
| 2024 BRFSS Labeled Cohort Size | 456,218 records (20,661 positive; 4.5285%) | `docs/experiments/availability_paper/cohort.json` | `["labeled2024"]`, `["positive2024"]` |
| 2025 BRFSS Labeled Cohort Size | 354,950 records (16,353 positive; 4.6071%) | `docs/experiments/availability_paper/cohort.json` | `["labeled2025"]`, `["positive2025"]` |
| 2025 Common-Jurisdiction Size | 349,241 records (16,082 positive, 49 states) | `docs/experiments/availability_paper/cohort.json` | `["common2025"]`, `["common_positive2025"]` |
| Primary Contrast (Trees MT vs. SC) | $\hat\theta = -0.000156566$ ($\approx 0.48\%$ Brier reduction) | `docs/experiments/availability_paper/transport2025.json` | `["primary_contrast"]["mean"]` |
| 2025 Trees 7-Variable Brier (FG) | $0.0332212$ | `docs/experiments/availability_paper/all_metrics.csv` | `cohort=transport2025_common, weighted_brier` |
| 2025 Trees 7-Variable Brier (SC) | $0.0329937$ | `docs/experiments/availability_paper/all_metrics.csv` | `cohort=transport2025_common, weighted_brier` |
| 2025 Trees 7-Variable Brier (MT) | $0.0328371$ | `docs/experiments/availability_paper/all_metrics.csv` | `cohort=transport2025_common, weighted_brier` |
| 2025 Trees 7-Variable Brier (SR) | $0.0327834$ | `docs/experiments/availability_paper/all_metrics.csv` | `cohort=transport2025_common, weighted_brier` |
| Singleton Strata Count | 93 strata with $m_h = 1$ PSU out of 2,034 | `docs/experiments/availability_paper/singleton_sensitivity.json` | `["singleton_strata_count"]` |
| Post-hoc Average Sensitivity CI | $[-0.0002563, -0.0000569]$ | `docs/experiments/availability_paper/singleton_sensitivity.json` | `["ci95_average"]` |
| Same-Model MC DSI (256 paths) | $0.927764$ (95% CI $[0.923155, 0.932162]$) | `docs/experiments/trust/attribution_convergence.json` | `["repeats"][2]["same_model_repeat"]["dsi"]` |
| Retrained 3-Seed Pair DSI | $0.890181$ (Spearman $0.8260$, Jaccard $0.7375$) | `docs/experiments/trust/attribution_convergence.json` | `["seed_retraining_256"]["dsi"]` |
| CAS Disagreement Error AUROC | $0.830233$ (vs. Entropy Error AUROC $0.936487$) | `docs/experiments/trust/trust.json` | `["cas"]["disagreement_error_detection_auroc"]` |
| Aggregate 7-Scenario RAI | $0.717437$ (Full Brier $0.030689$, Null $0.033135$) | `docs/experiments/trust/availability.json` | `["rai"]["aggregate"]` |
| Adaptive Policy Full-Pop Brier (Full) | $0.040532$ (Fixed Recal: $0.040262$) | `results/brfss_adaptive_full/evaluation_results.json` | `["scenarios"]["full"]["adaptive_policy"]["full_population_probability_metrics"]["brier"]` |
| Adaptive Paired Diff (Full Scenario) | $+0.000269$ (95% CI $[+0.000158, +0.000370]$) | `results/brfss_adaptive_full/evaluation_results.json` | `["scenarios"]["full"]["paired_brier_diff_adaptive_minus_fixed"]` |
| Adaptive Paired Diff (Demographics) | $-0.000503$ (95% CI $[-0.000614, -0.000394]$) | `results/brfss_adaptive_full/evaluation_results.json` | `["scenarios"]["demographics"]["paired_brier_diff_adaptive_minus_fixed"]` |
| Discarded Strokes under Full Abstention | 532 confirmed stroke cases ($26.9\%$ prevalence) | `results/brfss_adaptive_full/evaluation_results.json` | Difference between total test positives ($4,132$) and covered positives ($810+2,790=3,600$) |
| Acquisition Benchmark Latency | Static CVDINFR4 $8.13\text{ ms}$ vs. Donors $18.39\text{ ms}$ | `results/acquisition_benchmark/benchmark_results.json` | `["scenarios"]["demographics"]["strategies"]["*"]["ms_per_record"]` |
| Benchmark AUROC in Demographics | Static CVDINFR4 $0.7455$ vs. Donors $0.7281$ | `results/acquisition_benchmark/benchmark_results.json` | `["scenarios"]["demographics"]["strategies"]["*"]["full_population_probability_metrics"]["auroc"]` |

---

## 3. Unverified Leads & Literature Gaps

1. **Absence Claims:** Do not claim that "no prior stroke prediction paper evaluates trust, calibration, or missingness." Many clinical risk scores (Framingham, CHA2DS2-VASc) evaluate calibration, and recent ML works (e.g., Moulaei 2024, Zhi 2024, Melnykova 2025) explore explainability and class imbalance.
2. **Novelty of DSI, CAS, and RAI:** These must be presented as **provisional trust summaries and diagnostic metrics**, not foundational algorithmic inventions. DSI is an $L_1$-normalized cosine-style attribution similarity; CAS is Jensen-gap entropy decomposition; RAI is normalized headroom retention.
3. **TabNet Clinical Superiority:** TabNet was investigated (`docs/TABNET_TRUST_RESULTS.md`) but achieved selection AP of $0.1675$, which was inferior to the tree/MLP blend ($0.1938$). Do not claim TabNet as the superior stroke model.
