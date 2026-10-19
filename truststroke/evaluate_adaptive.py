"""Comparative evaluation for trust-aware adaptive decision policy.

Compares three intervention regimes across identical respondents and feature-availability scenarios:
1. No intervention: static base model with global imputation, no recalibration or acquisition.
2. Fixed scenario recalibration: static model calibrated specifically for the observed scenario.
3. Adaptive decision policy: AdaptiveDecisionPolicy with uncertainty routing, conditional donor
   acquisition (via ConditionalDonors), and safety abstention.

All thresholds and parameters are tuned exclusively on development/validation data;
the holdout test partition is evaluated once at the conclusion.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

from .baseline import FEATURES, CATEGORICAL, NUMERIC, clean, build_model, metrics, synthetic
from .optimize import load_data, probability_logit
from .acquisition import ConditionalDonors
from .adaptive_layer import AdaptiveDecisionPolicy, DecisionRecord

# Standard evaluation scenarios from availability_study_frozen.json
SCENARIOS = {
    "full": FEATURES,
    "without_health_state": [
        "SEXVAR", "_AGEG5YR", "EXERANY2", "_SMOKER3", "DIABETE4",
        "CVDINFR4", "CVDCRHD4", "CHCKDNY2", "HAVARTH4", "EDUCA", "INCOME3", "_BMI5"
    ],
    "demographics_history": [
        "SEXVAR", "_AGEG5YR", "DIABETE4", "CVDINFR4", "CVDCRHD4", "CHCKDNY2", "HAVARTH4"
    ],
    "demographics": ["SEXVAR", "_AGEG5YR"],
}


def create_partitions(y: pd.Series, seed: int = 42) -> Dict[str, np.ndarray]:
    """Partition indices disjointly into train (60%), development/cal (20%), and locked test (20%)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.40, stratify=y, random_state=seed)
    dev, test = train_test_split(rest, test_size=0.50, stratify=np.asarray(y)[rest], random_state=seed + 1)
    return {"train": train, "dev": dev, "test": test}


def mask_record(full_record: Dict[str, Any], allowed_features: List[str]) -> Dict[str, Any]:
    """Return dictionary containing only the observed features that are not NaN."""
    return {
        k: full_record[k]
        for k in allowed_features
        if k in full_record and not pd.isna(full_record[k])
    }


def fit_scenario_calibrators(
    base_model: Any,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    scenarios: Dict[str, List[str]],
) -> Dict[str, LogisticRegression]:
    """Fit Platt recalibrator for each scenario on the development split."""
    calibrators = {}
    for sc_name, allowed_cols in scenarios.items():
        # Mask out unavailable columns with NaN
        masked_X = X_dev.copy()
        missing_cols = [c for c in FEATURES if c not in allowed_cols]
        masked_X[missing_cols] = np.nan

        # Raw base model probabilities on masked development data
        p_raw = base_model.predict_proba(masked_X)[:, 1]
        logits = probability_logit(p_raw)

        # Fit isotonic/Platt logistic calibrator
        cal = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        cal.fit(logits, y_dev)
        calibrators[sc_name] = cal
    return calibrators


def evaluate_no_intervention(
    base_model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    allowed_cols: List[str],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate baseline: static model on masked features, no acquisition, no abstention."""
    masked_X = X_eval.copy()
    missing_cols = [c for c in FEATURES if c not in allowed_cols]
    masked_X[missing_cols] = np.nan

    p = base_model.predict_proba(masked_X)[:, 1]
    m = metrics(y_eval, p, threshold=threshold)
    m["coverage"] = 1.0
    m["abstention_rate"] = 0.0
    m["mean_acquisition_steps"] = 0.0
    m["probabilities"] = p
    return m


def evaluate_fixed_recalibration(
    base_model: Any,
    calibrator: LogisticRegression,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    allowed_cols: List[str],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate fixed recalibration: calibrated on scenario logit, no acquisition, no abstention."""
    masked_X = X_eval.copy()
    missing_cols = [c for c in FEATURES if c not in allowed_cols]
    masked_X[missing_cols] = np.nan

    p_raw = base_model.predict_proba(masked_X)[:, 1]
    p_cal = calibrator.predict_proba(probability_logit(p_raw))[:, 1]
    m = metrics(y_eval, p_cal, threshold=threshold)
    m["coverage"] = 1.0
    m["abstention_rate"] = 0.0
    m["mean_acquisition_steps"] = 0.0
    m["probabilities"] = p_cal
    return m


def evaluate_adaptive(
    policy: AdaptiveDecisionPolicy,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    allowed_cols: List[str],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate AdaptiveDecisionPolicy on evaluated records with simulated feature unavailability."""
    decisions: List[DecisionRecord] = []
    
    # Iterate through evaluation records
    for i in range(len(X_eval)):
        row = X_eval.iloc[i].to_dict()
        observed_init = mask_record(row, allowed_cols)

        # Ground truth provider: strictly on-demand callback to simulate acquiring from respondent
        def respondent_provider(feat: str) -> Any:
            return row.get(feat, np.nan)

        decision = policy.decide(observed_init, ground_truth_provider=respondent_provider)
        decisions.append(decision)

    probs = np.array([d.probability for d in decisions], dtype=float)
    actions = [d.action for d in decisions]
    steps = np.array([d.acquisition_steps for d in decisions], dtype=float)

    n_total = len(y_eval)
    classified_mask = np.array([a == "classify" for a in actions], dtype=bool)
    n_classified = int(classified_mask.sum())
    coverage = float(n_classified / n_total) if n_total > 0 else 0.0
    abstention_rate = 1.0 - coverage

    # Compute metrics on covered (classified) respondents if any, and on all respondents for ranking
    m_covered = metrics(y_eval[classified_mask], probs[classified_mask], threshold=threshold) if n_classified > 10 and len(np.unique(y_eval[classified_mask])) == 2 else {}
    m_all = metrics(y_eval, probs, threshold=threshold)

    result = {
        "n_total": n_total,
        "n_covered": n_classified,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "mean_acquisition_steps": float(steps.mean()),
        "acquisition_frequency": float((steps > 0).mean()),
        "auroc_all": m_all.get("auroc"),
        "average_precision_all": m_all.get("average_precision"),
        "brier_all": m_all.get("brier"),
        "brier_covered": m_covered.get("brier"),
        "sensitivity_covered": m_covered.get("sensitivity"),
        "probabilities": probs,
        "actions": actions,
        "steps": steps,
    }
    return result


def paired_bootstrap_brier_diff(
    y: np.ndarray,
    p_baseline: np.ndarray,
    p_candidate: np.ndarray,
    replicates: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap CI for candidate minus baseline Brier score difference."""
    y = np.asarray(y)
    b_diffs = []
    rng = np.random.default_rng(seed)
    n = len(y)
    diff = (p_candidate - y) ** 2 - (p_baseline - y) ** 2
    point_diff = float(diff.mean())

    for _ in range(replicates):
        idx = rng.integers(0, n, n)
        b_diffs.append(float(diff[idx].mean()))

    b_diffs = np.sort(b_diffs)
    return {
        "mean_difference": point_diff,
        "ci95_low": float(np.quantile(b_diffs, 0.025)),
        "ci95_high": float(np.quantile(b_diffs, 0.975)),
        "replicates": replicates,
    }


def tune_policy_thresholds(
    base_model: Any,
    calibrator: LogisticRegression,
    donor_engine: ConditionalDonors,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    scenarios: Dict[str, List[str]],
    decision_threshold: float = 0.14,
    sample_limit: int = 500,
) -> Dict[str, Any]:
    """Select adaptive policy thresholds on development data to balance Brier score and coverage."""
    # Subsample development data if large to keep threshold tuning fast
    if len(X_dev) > sample_limit:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(X_dev), sample_limit, replace=False)
        X_tune = X_dev.iloc[sub_idx].reset_index(drop=True)
        y_tune = y_dev.iloc[sub_idx].reset_index(drop=True)
    else:
        X_tune = X_dev.reset_index(drop=True)
        y_tune = y_dev.reset_index(drop=True)

    # Candidate grids for tuning
    candidate_tau_acq = [0.3, 0.5, 0.7]
    candidate_tau_abs = [0.85, 0.95, None]

    best_score = float("inf")
    best_cfg = {
        "tau_entropy_acq": 0.5,
        "tau_entropy_abstain": None,
        "max_acquisition_steps": 1,
    }

    # Evaluate across a representative degraded scenario ('demographics')
    test_scenario = scenarios["demographics"]

    for tau_acq in candidate_tau_acq:
        for tau_abs in candidate_tau_abs:
            policy = AdaptiveDecisionPolicy(
                model=base_model,
                decision_threshold=decision_threshold,
                tau_entropy_acq=tau_acq,
                tau_entropy_abstain=tau_abs,
                max_acquisition_steps=1,
                donor_engine=donor_engine,
            )
            res = evaluate_adaptive(policy, X_tune, y_tune, test_scenario, decision_threshold)
            # Penalize abstention heavily if coverage falls below 80%
            cov = res["coverage"]
            brier = res["brier_all"]
            effective_score = brier if cov >= 0.80 else brier + 1.0 * (0.80 - cov)

            if effective_score < best_score:
                best_score = effective_score
                best_cfg = {
                    "tau_entropy_acq": tau_acq,
                    "tau_entropy_abstain": tau_abs,
                    "max_acquisition_steps": 1,
                }

    return best_cfg


def run_experiment(args) -> Dict[str, Any]:
    """Execute comparative evaluation between no-intervention, fixed recalibration, and adaptive."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    rng_seed = args.seed

    if args.smoke:
        print("Running in synthetic SMOKE test mode...")
        raw, data_digest = synthetic(n=args.smoke_records, seed=rng_seed), "SYNTHETIC_SMOKE_FIXTURE"
        is_smoke = True
    else:
        if not args.data:
            raise ValueError("--data path must be provided unless --smoke is specified")
        print(f"Loading BRFSS dataset from {args.data}...")
        raw, data_digest = load_data(args.data)
        is_smoke = False

    print("Cleaning predictors and outcome...")
    X, y, design = clean(raw)
    print(f"Cohort ready: {len(X)} records, prevalence = {y.mean():.4f}")

    # Subsample if requested for quick execution
    if args.limit and len(X) > args.limit:
        print(f"Limiting total dataset to {args.limit} records...")
        X = X.iloc[:args.limit].reset_index(drop=True)
        y = y.iloc[:args.limit].reset_index(drop=True)

    # Disjoint splits: train, dev, locked test
    parts = create_partitions(y, seed=rng_seed)
    train_idx, dev_idx, test_idx = parts["train"], parts["dev"], parts["test"]
    print(f"Partitions: train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}")

    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_dev, y_dev = X.iloc[dev_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True)
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    print("Fitting baseline model on training split...")
    base_model = build_model(name="logistic", seed=rng_seed)
    base_model.fit(X_train, y_train)

    print("Fitting scenario-specific calibrators on development split...")
    calibrators = fit_scenario_calibrators(base_model, X_dev, y_dev, SCENARIOS)

    print("Initializing conditional donor engine on complete training donors...")
    donor_pool = X_train.dropna(subset=FEATURES).copy()
    if len(donor_pool) > 2000:
        donor_pool = donor_pool.sample(n=2000, random_state=rng_seed).reset_index(drop=True)
    donor_engine = ConditionalDonors(
        predict=lambda df: base_model.predict_proba(df)[:, 1],
        donors=donor_pool,
        neighbors=min(64, len(donor_pool)),
    )

    print("Tuning adaptive policy thresholds on development split...")
    decision_threshold = 0.14
    tuned_cfg = tune_policy_thresholds(
        base_model,
        calibrators["demographics"],
        donor_engine,
        X_dev,
        y_dev,
        SCENARIOS,
        decision_threshold=decision_threshold,
        sample_limit=args.tune_limit,
    )
    print(f"Tuned thresholds: {tuned_cfg}")

    # Build frozen adaptive policy with tuned thresholds
    policy = AdaptiveDecisionPolicy(
        model=base_model,
        decision_threshold=decision_threshold,
        tau_entropy_acq=tuned_cfg["tau_entropy_acq"],
        tau_entropy_abstain=tuned_cfg["tau_entropy_abstain"],
        max_acquisition_steps=tuned_cfg["max_acquisition_steps"],
        donor_engine=donor_engine,
    )

    # Evaluate across all scenarios on the LOCKED test partition
    print("Evaluating policies across scenarios on the locked test partition...")
    scenario_results = {}

    for sc_name, allowed_cols in SCENARIOS.items():
        print(f"\nEvaluating scenario: {sc_name} ({len(allowed_cols)}/{len(FEATURES)} features observed)")
        
        # 1. No Intervention
        res_no = evaluate_no_intervention(base_model, X_test, y_test, allowed_cols, decision_threshold)
        
        # 2. Fixed Recalibration
        res_fix = evaluate_fixed_recalibration(
            base_model, calibrators[sc_name], X_test, y_test, allowed_cols, decision_threshold
        )
        
        # 3. Adaptive Policy
        res_adapt = evaluate_adaptive(policy, X_test, y_test, allowed_cols, decision_threshold)

        # Paired bootstrap difference (Adaptive minus Fixed Recalibration Brier score)
        paired_diff = paired_bootstrap_brier_diff(
            y_test.to_numpy(),
            res_fix["probabilities"],
            res_adapt["probabilities"],
            replicates=args.bootstrap,
            seed=rng_seed,
        )

        scenario_results[sc_name] = {
            "features_observed": allowed_cols,
            "no_intervention": {k: v for k, v in res_no.items() if k not in ["probabilities"]},
            "fixed_recalibration": {k: v for k, v in res_fix.items() if k not in ["probabilities"]},
            "adaptive_policy": {
                k: v for k, v in res_adapt.items() if k not in ["probabilities", "actions", "steps"]
            },
            "paired_brier_diff_adaptive_minus_fixed": paired_diff,
        }

    duration = time.time() - start_time
    summary = {
        "status": "COMPLETED",
        "data_mode": "SYNTHETIC_SMOKE" if is_smoke else "REAL_BRFSS",
        "data_sha256": data_digest,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": float(duration),
        "seed": rng_seed,
        "sample_counts": {
            "total": len(X),
            "train": len(X_train),
            "dev": len(X_dev),
            "test": len(X_test),
        },
        "tuned_hyperparameters": tuned_cfg,
        "scenarios": scenario_results,
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
        },
    }

    result_path = out / "evaluation_results.json"
    with result_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation completed in {duration:.2f}s. Results written to {result_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trust-aware adaptive decision policy")
    parser.add_argument("--data", type=str, help="Path to BRFSS .XPT or .csv dataset")
    parser.add_argument("--smoke", action="store_true", help="Run with synthetic smoke data")
    parser.add_argument("--smoke-records", type=int, default=500, help="Number of records for smoke run")
    parser.add_argument("--limit", type=int, default=None, help="Limit total dataset records for faster execution")
    parser.add_argument("--tune-limit", type=int, default=500, help="Max records for tuning thresholds on dev")
    parser.add_argument("--bootstrap", type=int, default=100, help="Bootstrap replicates for paired diffs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/adaptive_eval", help="Output directory")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
