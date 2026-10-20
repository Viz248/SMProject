"""Comparative benchmark: ConditionalDonors vs. Static Feature Acquisition Heuristics.

Evaluates feature acquisition strategies under identical missingness scenarios and test records:
1. No Acquisition: Decision policy without feature querying.
2. ConditionalDonors: Dynamic information-gain acquisition via k-nearest donor distributions.
3. Static Heuristic (CVDINFR4): Always queries CVDINFR4 (history of heart attack) when missing.
4. Static Heuristic (GENHLTH): Always queries GENHLTH (general health status) when missing.

All policies operate under identical training, development calibrators, and locked test partitions.
Acquired feature values are revealed strictly on-demand only after selection.
"""
import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .baseline import FEATURES, CATEGORICAL, NUMERIC, clean, build_model, metrics, synthetic
from .optimize import load_data, probability_logit
from .acquisition import ConditionalDonors
from .adaptive_layer import AdaptiveDecisionPolicy, DecisionRecord
from .evaluate_adaptive import (
    SCENARIOS,
    create_partitions,
    mask_record,
    fit_scenario_calibrators,
    paired_bootstrap_brier_diff,
)


class StaticFeatureRecommender:
    """Mock/wrapper providing recommendation of a single pre-specified feature."""
    def __init__(self, target_feature: str):
        self.target_feature = target_feature

    def recommend(self, observed: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        if self.target_feature in observed:
            return None, {"expected_information_gain": {}}
        return self.target_feature, {
            "expected_information_gain": {self.target_feature: 1.0}
        }


def evaluate_acquisition_strategy(
    policy: AdaptiveDecisionPolicy,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    allowed_cols: List[str],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a policy on evaluation records, measuring latency, predictions, and metrics."""
    decisions: List[DecisionRecord] = []
    t0 = time.perf_counter()

    for i in range(len(X_eval)):
        row = X_eval.iloc[i].to_dict()
        observed_init = mask_record(row, allowed_cols)

        # Strictly on-demand ground truth callback: unobserved values are not accessible before call
        def respondent_provider(feat: str) -> Any:
            return row.get(feat, np.nan)

        decision = policy.decide(observed_init, ground_truth_provider=respondent_provider)
        decisions.append(decision)

    elapsed_sec = time.perf_counter() - t0
    probs = np.array([d.probability for d in decisions], dtype=float)
    actions = [d.action for d in decisions]
    steps = np.array([d.acquisition_steps for d in decisions], dtype=float)

    n_total = len(y_eval)
    classified_mask = np.array([a == "classify" for a in actions], dtype=bool)
    n_classified = int(classified_mask.sum())
    coverage = float(n_classified / n_total) if n_total > 0 else 0.0
    abstention_rate = 1.0 - coverage

    m_full = metrics(y_eval, probs, threshold=threshold)

    if n_classified > 0 and len(np.unique(y_eval[classified_mask])) == 2:
        m_cov = metrics(y_eval[classified_mask], probs[classified_mask], threshold=threshold)
        cov_metrics = {
            "n": n_classified,
            "prevalence": m_cov.get("prevalence"),
            "sensitivity": m_cov.get("sensitivity"),
            "specificity": m_cov.get("specificity"),
            "precision": m_cov.get("precision"),
            "f1": m_cov.get("f1"),
            "confusion_matrix": m_cov.get("confusion_matrix"),
            "brier": m_cov.get("brier"),
            "auroc": m_cov.get("auroc"),
            "average_precision": m_cov.get("average_precision"),
        }
    else:
        cov_metrics = {
            "n": n_classified,
            "prevalence": float(np.mean(y_eval[classified_mask])) if n_classified > 0 else None,
            "sensitivity": None,
            "specificity": None,
            "precision": None,
            "f1": None,
            "confusion_matrix": None,
            "brier": float(np.mean((probs[classified_mask] - y_eval[classified_mask]) ** 2)) if n_classified > 0 else None,
            "auroc": None,
            "average_precision": None,
        }

    return {
        "n_total": n_total,
        "n_covered": n_classified,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "mean_acquisition_steps": float(steps.mean()),
        "acquisition_frequency": float((steps > 0).mean()),
        "compute_time_seconds": float(elapsed_sec),
        "ms_per_record": float((elapsed_sec / n_total) * 1000.0) if n_total > 0 else 0.0,
        "full_population_probability_metrics": {
            "brier": m_full.get("brier"),
            "auroc": m_full.get("auroc"),
            "average_precision": m_full.get("average_precision"),
            "ece": m_full.get("ece"),
        },
        "covered_decision_metrics": cov_metrics,
        "probabilities": probs,
    }


def run_benchmark(args) -> Dict[str, Any]:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed

    if args.smoke:
        print("Running acquisition benchmark in synthetic SMOKE mode...")
        raw, digest = synthetic(n=args.smoke_records, seed=rng_seed), "SYNTHETIC_SMOKE"
        is_smoke = True
    else:
        if not args.data:
            raise ValueError("--data path must be supplied when --smoke is not set")
        print(f"Loading data from {args.data}...")
        raw, digest = load_data(args.data)
        is_smoke = False

    X, y, design = clean(raw)
    if args.limit and len(X) > args.limit:
        print(f"Limiting cohort to {args.limit} records...")
        X = X.iloc[:args.limit].reset_index(drop=True)
        y = y.iloc[:args.limit].reset_index(drop=True)

    parts = create_partitions(y, seed=rng_seed)
    train_idx, dev_idx, test_idx = parts["train"], parts["dev"], parts["test"]
    print(f"Cohort sizes: train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}")

    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_dev, y_dev = X.iloc[dev_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True)
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    print("Fitting baseline model...")
    base_model = build_model(name="logistic", seed=rng_seed)
    base_model.fit(X_train, y_train)

    print("Fitting scenario-specific calibrators on dev partition...")
    calibrators = fit_scenario_calibrators(base_model, X_dev, y_dev, SCENARIOS)
    calibrator_map = {}
    for sc_name, cols in SCENARIOS.items():
        cal = calibrators[sc_name]
        calibrator_map[sc_name] = cal
        calibrator_map[tuple(sorted(cols))] = cal

    print("Setting up donor pool...")
    donor_pool = X_train.dropna(subset=FEATURES).copy()
    if len(donor_pool) > 2000:
        donor_pool = donor_pool.sample(n=2000, random_state=rng_seed).reset_index(drop=True)

    donor_engine = ConditionalDonors(
        predict=lambda df: base_model.predict_proba(df)[:, 1],
        donors=donor_pool,
        neighbors=min(64, len(donor_pool)),
    )

    static_cvd = StaticFeatureRecommender("CVDINFR4")
    static_gen = StaticFeatureRecommender("GENHLTH")

    decision_threshold = 0.14
    tau_acq = args.tau_acq
    tau_abstain = args.tau_abstain

    # Policies to compare
    strategies = {
        "no_acquisition": AdaptiveDecisionPolicy(
            model=base_model,
            decision_threshold=decision_threshold,
            tau_entropy_acq=None,
            tau_entropy_abstain=tau_abstain,
            max_acquisition_steps=0,
            donor_engine=None,
            calibrators=calibrator_map,
        ),
        "conditional_donors": AdaptiveDecisionPolicy(
            model=base_model,
            decision_threshold=decision_threshold,
            tau_entropy_acq=tau_acq,
            tau_entropy_abstain=tau_abstain,
            max_acquisition_steps=1,
            donor_engine=donor_engine,
            calibrators=calibrator_map,
        ),
        "static_cvdinfr4": AdaptiveDecisionPolicy(
            model=base_model,
            decision_threshold=decision_threshold,
            tau_entropy_acq=tau_acq,
            tau_entropy_abstain=tau_abstain,
            max_acquisition_steps=1,
            donor_engine=static_cvd,
            calibrators=calibrator_map,
        ),
        "static_genhlth": AdaptiveDecisionPolicy(
            model=base_model,
            decision_threshold=decision_threshold,
            tau_entropy_acq=tau_acq,
            tau_entropy_abstain=tau_abstain,
            max_acquisition_steps=1,
            donor_engine=static_gen,
            calibrators=calibrator_map,
        ),
    }

    # Evaluate across scenarios
    scenario_evals = {}
    test_y_np = y_test.to_numpy()

    for sc_name, allowed_cols in SCENARIOS.items():
        print(f"\n--- Scenario: {sc_name} ({len(allowed_cols)}/15 observed) ---")
        strat_results = {}
        probs_by_strat = {}

        for strat_name, policy in strategies.items():
            res = evaluate_acquisition_strategy(
                policy=policy,
                X_eval=X_test,
                y_eval=y_test,
                allowed_cols=allowed_cols,
                threshold=decision_threshold,
            )
            probs_by_strat[strat_name] = res.pop("probabilities")
            strat_results[strat_name] = res

            fpm = res["full_population_probability_metrics"]
            cdm = res["covered_decision_metrics"]
            print(f"  [{strat_name:20s}] Time: {res['ms_per_record']:.2f}ms/rec | Acq: {res['acquisition_frequency']:.3f} | Cov: {res['coverage']:.3f} | Brier: {fpm['brier']:.5f} | AUROC: {fpm['auroc'] or 0:.4f} | AP: {fpm['average_precision'] or 0:.4f} | Sens: {cdm['sensitivity'] or 0:.4f}")

        # Compute paired bootstrap differences vs no_acquisition and vs static heuristics
        paired_contrasts = {}
        p_base = probs_by_strat["no_acquisition"]
        p_cd = probs_by_strat["conditional_donors"]

        paired_contrasts["conditional_donors_minus_no_acq"] = paired_bootstrap_brier_diff(
            test_y_np, p_base, p_cd, replicates=args.bootstrap, seed=rng_seed
        )
        for stat_name in ["static_cvdinfr4", "static_genhlth"]:
            p_stat = probs_by_strat[stat_name]
            paired_contrasts[f"conditional_donors_minus_{stat_name}"] = paired_bootstrap_brier_diff(
                test_y_np, p_stat, p_cd, replicates=args.bootstrap, seed=rng_seed
            )
            paired_contrasts[f"{stat_name}_minus_no_acq"] = paired_bootstrap_brier_diff(
                test_y_np, p_base, p_stat, replicates=args.bootstrap, seed=rng_seed
            )

        scenario_evals[sc_name] = {
            "strategies": strat_results,
            "paired_brier_differences": paired_contrasts,
        }

    summary = {
        "status": "COMPLETED",
        "data_mode": "SYNTHETIC_SMOKE" if is_smoke else "REAL_BRFSS",
        "data_sha256": digest,
        "sample_counts": {"total": len(X), "train": len(X_train), "dev": len(X_dev), "test": len(X_test)},
        "hyperparameters": {
            "decision_threshold": decision_threshold,
            "tau_acq": tau_acq,
            "tau_abstain": tau_abstain,
            "max_acquisition_steps": 1,
            "bootstrap_replicates": args.bootstrap,
        },
        "scenarios": scenario_evals,
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    out_file = out / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBenchmark finished. Saved to {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark ConditionalDonors vs Static Heuristic Acquisition")
    parser.add_argument("--data", type=str, help="Path to BRFSS .XPT or .csv")
    parser.add_argument("--smoke", action="store_true", help="Run synthetic smoke benchmark")
    parser.add_argument("--smoke-records", type=int, default=300, help="Number of synthetic smoke records")
    parser.add_argument("--limit", type=int, default=5000, help="Max records for fast pilot evaluation")
    parser.add_argument("--tau-acq", type=float, default=0.30, help="Entropy threshold for triggering acquisition")
    parser.add_argument("--tau-abstain", type=float, default=0.85, help="Entropy threshold for abstention")
    parser.add_argument("--bootstrap", type=int, default=100, help="Bootstrap replicates for paired diffs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/acquisition_benchmark", help="Output directory")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
