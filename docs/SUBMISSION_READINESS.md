# Submission readiness and reporting review

## Completed

The project now includes an executed forward-year empirical study, a frozen pre-evaluation protocol, real-data provenance and checksums, saved model manifests and aggregate results, controls separating recalibration from training, secondary domain analyses, a reconstructed revised IEEE-format PDF, editable source, figures, a reproduction notebook and tested code. Earlier null acquisition and TabNet results are preserved rather than rewritten as improvements.

## Required before an actual submission

1. **Real authors and affiliations.** The draft is anonymous. Supply the agreed author order, institution names and contact details; do not infer authorship from GitHub names. Each author must review and approve the work.
2. **Target conference.** Apply its actual template, page limit, anonymity policy, supplementary-material rules and disclosure requirements. The supplied generic IEEEtran layout is not confirmation that a particular venue accepts five pages or this topic.
3. **Scientific and statistical review.** The prespecified primary interval was unavailable because of singleton strata. The post-hoc average-contribution sensitivity is transparent but assumption-dependent and has now been independently verified with R survey, with separate full-design adjust/average sensitivities and Python agreement. A supervisor/statistical reviewer should assess the survey estimand, missing-outcome exclusions, singleton handling and limits on any superiority statement. Do not substitute the sensitivity for a prespecified confirmatory interval.
4. **Contribution judgment.** The contribution is empirical control/transport evidence, not a new algorithm. Whether that is sufficient depends on the venue. Do not reinstate claims that DSI/CAS/RAI, mask training or DIME-style acquisition are independently novel.
5. **Disclosures and institutional determinations.** The manuscript discloses AI assistance. Authors should verify that disclosure and supply funding, conflicts and any institutional determination required for secondary public-data analysis. No such determination or ethics approval is invented in the draft.
6. **Public release if desired.** The working branch is committed locally; the source package is delivered here. It has not been pushed or submitted externally. Use the supplied source package or publish the reviewed branch before claiming the final study code is publicly available.

These are concrete manuscript-review and submission decisions, not missing model runs. There is no guarantee of IEEE publication. No model, calibration or hyperparameter should now be tuned against 2025 and described as independently confirmed on that same year.

## Reporting map (TRIPOD+AI-informed; not a certified compliance audit)

| Topic | Location / status |
|---|---|
| Study purpose and endpoint | Abstract, Introduction, Data; lifetime self-report, no incident prediction |
| Data sources, dates, selection and counts | Data section; cohort/provenance JSON; known-outcome selection explicit |
| Predictors and coding | Data section and OUTCOME_AND_FEATURE_PROTOCOL.md |
| Missingness | Natural nonresponse retained; simulated unavailability distinguished; no causal MNAR inference |
| Training, stopping, calibration and evaluation | Experimental Protocol; split configuration and model manifest |
| Model parameters and ensemble | Experimental Protocol; frozen config; full histories retained |
| Primary estimand and uncertainty | Equations and frozen config; unavailable interval disclosed; sensitivity marked post-hoc |
| Model evaluation and subgroup scope | Results; all_metrics.csv; sex/age/income only, not comprehensive fairness |
| Limitations and applicability | Discussion/Limitations; cross-sectional, known masks, unweighted training, one new year |
| Code/data access and reproducibility | Public CDC links/checksums; accompanying source; no final remote release claimed |
| Real author contributions, funding/conflicts, institutional determinations | Authors must supply before submission |

The public-data analyses support an empirical research draft. They do not establish clinical deployment readiness, future-stroke prediction or safety of automated medical decisions.
