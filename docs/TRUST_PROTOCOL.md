# Provisional trust framework: definitions and execution protocol

These definitions are research proposals built from conventional metrics. They are not established novel measures or validated clinical quantities. Implementation: truststroke/trust.py, acquisition.py and trust_experiment.py. All new experiments use development records; the previously inspected final-test partition is excluded.

## Decision Stability Index (DSI)

Let a_r(x) in R^15 be the probability-scale attribution vector of retrained model r for the same record x, class 1 (self-reported stroke history), feature order, background population and explanation protocol. Define u_r=a_r/||a_r||_1 when ||a_r||_1>epsilon, epsilon=1e-12.

DSI_(r,s)(x) = 1 - (1/2)||u_r(x)-u_s(x)||_1, with range [0,1]. Positive scaling leaves the score unchanged; reversing every attribution sign gives zero. Both zero vectors produce an undefined score plus a noninformative flag; one zero vector gives zero by declared convention. This convention is a choice, not a derived clinical interpretation.

For each record, average uniformly over declared retraining pairs; then report the distribution and bootstrap the RECORD averages, rather than treating model pairs as independent respondents. Report random-seed retraining, bootstrap retraining and input perturbation separately. Do not average these heterogeneous interventions into a single trust score.

Explanations are Monte Carlo permutation Shapley estimates on original variable groups. Each permutation samples a training donor and replaces variables in a random order. Shared donor/permutation paths across retrained models reduce explanation-sampling confounding. The signed increments sum to f(x)-E_background[f(background)] along the sampled paths. Local accuracy does NOT prove that individual attribution estimates have converged.

Controls: repeat explanations on the SAME model with a different path seed, report conventional absolute-attribution rank Spearman and top-5 overlap, and report local-accuracy residuals. Interventional coalitions can violate observed feature dependencies; no causal explanation claim is supported. Small perturbations change healthy-day counts by +/-1 (bounded 0–30) and BMI by +/-0.2 (lower bounded 10), preserving missingness and categories; they are numerical sensitivity probes, not longitudinal patient trajectories.

Signed normalized-L1 similarity is not intrinsically new. An information-addition claim requires an error/failure-detection comparison against the constituent similarity measures with independent validation. Constant explanations can be stable but useless.

## Confidence Attribution Score (CAS)

For calibrated member probabilities p_m(x), m=1,...,M, define pbar=M^-1 sum_m p_m and binary entropy h(p)=-p log2(p)-(1-p)log2(1-p), with h(0)=h(1)=0.

CAS(x) is a REPORT VECTOR:

- Total predictive entropy U_total=h(pbar), in [0,1] bits.
- Mean member entropy U_within=M^-1 sum_m h(p_m), in [0,1] bits.
- Ensemble disagreement U_between=U_total-U_within, in [0,1] bits, nonnegative by concavity (roundoff clipped at zero).
- Entropy confidence C=1-U_total, in [0,1].

U_total=U_within+U_between is an algebraic mixture identity. It does not uniquely identify epistemic versus aleatoric uncertainty. The five-model experiment mixes three initialization seeds and two training bootstrap samples; its distribution is a finite empirical ensemble, not a proven posterior. No arbitrary percentage is called a causal contribution. Feature masking can increase OR decrease entropy, so masking-associated entropy change is reported with its sign and alongside proper scoring loss.

Validation: ensemble Brier score, calibration bins/ECE, Spearman associations with absolute prediction error and AUROC for detecting errors at the declared .14 research threshold. These are descriptive associations; ranking error via entropy may partly reflect distance from the threshold. Post-hoc summaries do not improve the underlying classifier merely by being attached.

## Resource Adaptability Index (RAI)

On the SAME evaluation records, let B_full be full-input Brier loss, B_s the loss under feature-availability scenario s, and B_null the loss of the constant FITTING-prevalence predictor. If B_null<=B_full+epsilon, RAI is undefined because there is no positive Brier headroom to retain.

Otherwise define r_s = 1 - clip((B_s-B_full)/(B_null-B_full),0,1), then RAI=sum_s w_s r_s, with nonnegative normalized w_s. The pilot uses equal weights across seven declared scenarios; these are experimental weights, not measured deployment frequencies.

Range [0,1]. Full-reference performance or improvement gives retention 1; null performance or worse gives 0. Improvements and catastrophic degradations saturate, so raw losses and UNCLIPPED relative excess are always saved. RAI is a normalized degradation summary; usefulness beyond simpler loss curves remains unvalidated.

Scenarios: demographics only; demographics plus five disease-history variables; removal of general/physical/mental health state; independent MCAR masking at 10%, 30%, 50%; and age-associated BMI masking (probability .6 for age-group>=10, .15 otherwise). The informative scenario is declared synthetic, not a claim about clinical behavior. All fixed models use FITTING-derived imputation. Subset retraining is a different experiment and is not represented by these results. Added masking rates include naturally missing values in the reported total missing fraction.

## Conditional questionnaire-item acquisition

Start with known sex and age group on held-out COMPLETE development records. Obtain an empirical conditional distribution from 64 nearest complete FITTING donors within an 8192-donor training pool. Distance sums category mismatches and numeric absolute distances divided by training IQR and clipped to 1. No hidden evaluation feature or outcome is used in donor selection or recommendation.

For observed set O, approximate p(y=1|x_O) by averaging model predictions over donor completions. For candidate j, group donor completions by categorical value or four numeric quantile bins, and compute:

G_j(O) = h(mean_k p_k) - sum_b q_b h(mean_(k in b) p_k).

This is an empirical model-implied conditional information-gain proxy, not a true conditional mutual-information estimate from known population distributions. Recommend argmax_j G_j among unavailable features, then reveal only that chosen feature's actual held-out value and repeat.

Compare with random acquisition and a global ranking derived from mean absolute TRAINING-record Shapley values. All policies use the SAME conditional-completion predictor and the SAME held-out records. Report proper scoring loss, discrimination, ECE and entropy by acquisition step. Paired bootstrap effect sizes compare final Brier losses on the same records. Entropy reduction alone is not success: it may indicate increased overconfidence. There is no validated acquisition-cost model, so only feature counts are used. Complete-case evaluation and donor selection can cause selection bias. These are survey/questionnaire items, not invented CT/MRI/laboratory measurements.

## Evidence gates before manuscript claims

1. Complete exploratory runs and retain negative findings.
2. Test sensitivity to explanation permutation counts, training background, subgroup support and numerical conventions.
3. Compare DSI to rank/overlap measures; CAS to entropy/variance baselines; RAI to raw degradation curves; acquisition to random/global and stronger conditional acquisition methods.
4. Separate full-model fixed masking from subset retraining and missingness-aware training.
5. Validate on genuinely independent data, ideally another BRFSS release with harmonized coding; this is population/year transport validation, not incident-stroke forecasting.
6. Add design-aware survey intervals, matched repeated-training protocols, and appropriate uncertainty on paired effects. Single seeds/test cohorts are not independent patient studies.
7. Select conference and use its actual current template only after results and central contribution are defensible. No target conference was supplied, and no submission-ready manuscript is claimed.
