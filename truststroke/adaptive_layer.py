"""Modular deterministic adaptive decision policy.

Operates at inference time to route predictions across feature availability,
gauge uncertainty (CAS predictive entropy and ensemble disagreement), optionally
acquire features via ConditionalDonors, and abstain when uncertainty or decision margins
breach configured thresholds.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .baseline import FEATURES, CATEGORICAL, NUMERIC
from .trust import entropy, confidence_attribution
from .acquisition import ConditionalDonors


@dataclass(frozen=True)
class DecisionRecord:
    """Immutable record of an individual decision made by AdaptiveDecisionPolicy."""
    action: str  # 'classify', 'abstain'
    prediction: Optional[int]  # 0 or 1, or None if abstained
    probability: float  # predicted probability of positive class (stroke)
    confidence: float  # 1 - predictive_entropy
    predictive_entropy: float  # binary entropy in bits
    ensemble_disagreement: float  # ensemble disagreement in bits (0.0 for single models)
    initial_observed_features: Tuple[str, ...]
    final_observed_features: Tuple[str, ...]
    acquired_features: Tuple[str, ...]
    acquisition_steps: int
    reason: str


class AdaptiveDecisionPolicy:
    """Adaptive metric-driven decision policy.

    Routes prediction based on observed feature availability, uses CAS entropy and
    disagreement to assess prediction confidence, invokes ConditionalDonors
    to acquire informative missing features sequentially if permitted, and abstains
    if uncertainty or decision-margin criteria are met.

    DSI (attribution stability) and RAI (resource adaptability) are offline diagnostic
    metrics and are deliberately excluded from per-record online routing.
    """

    def __init__(
        self,
        model_or_ensemble: Optional[Union[Any, Sequence[Any]]] = None,
        model: Optional[Union[Any, Sequence[Any]]] = None,
        decision_threshold: float = 0.14,
        tau_entropy_acq: Optional[float] = None,
        tau_disagreement_acq: Optional[float] = None,
        tau_entropy_abstain: Optional[float] = None,
        margin_abstain: Optional[float] = None,
        max_acquisition_steps: int = 0,
        donor_engine: Optional[ConditionalDonors] = None,
        min_information_gain: float = 0.001,
        allowed_acquisition_features: Optional[Sequence[str]] = None,
        calibrators: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        model_or_ensemble : object or sequence of objects, optional
            Fitted estimator(s). Each must support `predict_proba(X)` where X is a DataFrame
            with FEATURES columns (possibly with NaNs/imputation handled by the model pipeline),
            or `predict(X, observed, key)` if using AvailabilityModel.
            If a sequence is provided, CAS ensemble decomposition is applied.
        model : object or sequence of objects, optional
            Alias for `model_or_ensemble`.
        decision_threshold : float
            Classification threshold for positive stroke risk (default: 0.14).
        tau_entropy_acq : float, optional
            Entropy threshold (in bits) above which acquisition is triggered.
        tau_disagreement_acq : float, optional
            Ensemble disagreement threshold (in bits) above which acquisition is triggered.
        tau_entropy_abstain : float, optional
            Predictive entropy threshold (in bits) above which the policy abstains.
        margin_abstain : float, optional
            Distance |p - decision_threshold| below which the policy abstains.
        max_acquisition_steps : int
            Maximum number of features to sequentially acquire (default: 0, no acquisition).
        donor_engine : ConditionalDonors, optional
            Instance of ConditionalDonors for computing expected information gain and recommending features.
        min_information_gain : float
            Minimum expected information gain (bits) required to execute an acquisition step.
        allowed_acquisition_features : sequence of str, optional
            Subset of FEATURES that are permissible to acquire. Defaults to all FEATURES.
        calibrators : dict, optional
            Mapping of scenario name or feature-tuple to a fitted calibrator (e.g. Platt LogisticRegression).
            Alternatively, if a single calibrator object is passed, it is used for calibration.
        """
        target_model = model_or_ensemble if model_or_ensemble is not None else model
        if target_model is None:
            raise ValueError("Must provide either model_or_ensemble or model.")

        if isinstance(target_model, (list, tuple)):
            if len(target_model) == 0:
                raise ValueError("Ensemble cannot be empty.")
            self.ensemble = list(target_model)
            self.is_ensemble = len(self.ensemble) > 1
        else:
            self.ensemble = [target_model]
            self.is_ensemble = False

        self.decision_threshold = float(decision_threshold)
        self.tau_entropy_acq = float(tau_entropy_acq) if tau_entropy_acq is not None else None
        self.tau_disagreement_acq = float(tau_disagreement_acq) if tau_disagreement_acq is not None else None
        self.tau_entropy_abstain = float(tau_entropy_abstain) if tau_entropy_abstain is not None else None
        self.margin_abstain = float(margin_abstain) if margin_abstain is not None else None
        self.max_acquisition_steps = int(max_acquisition_steps)
        self.donor_engine = donor_engine
        self.min_information_gain = float(min_information_gain)
        self.calibrators = calibrators

        if allowed_acquisition_features is None:
            self.allowed_acquisition_features = set(FEATURES)
        else:
            self.allowed_acquisition_features = set(allowed_acquisition_features)

    def _predict_prob(self, observed: Dict[str, Any]) -> Tuple[float, float, float]:
        """Compute mean calibrated probability, predictive entropy, and ensemble disagreement.

        Parameters
        ----------
        observed : dict
            Mapping from feature name to observed scalar value.

        Returns
        -------
        p_mean : float
        pred_entropy : float
        disagreement : float
        """
        # Construct single-row DataFrame containing all canonical FEATURES, filling unobserved with NaN
        row_dict = {col: [observed.get(col, np.nan)] for col in FEATURES}
        df_row = pd.DataFrame(row_dict)

        observed_cols = [c for c in FEATURES if c in observed and not pd.isna(observed[c])]

        member_probs = []
        for model in self.ensemble:
            if hasattr(model, "predict") and hasattr(model, "encoded") and hasattr(model, "calibrators"):
                # AvailabilityModel interface: predict(X, observed, key)
                key = "recalibrated" if "recalibrated" in model.calibrators else (
                    list(model.calibrators.keys())[0] if model.calibrators else None
                )
                if key:
                    p = float(model.predict(df_row, observed_cols, key)[0])
                else:
                    p = float(model.raw(df_row, observed_cols)[0])
            elif hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(df_row)
                p = float(prob_arr[0, 1] if prob_arr.ndim == 2 else prob_arr[0])
            elif callable(model):
                p_out = model(df_row)
                p = float(p_out[0] if hasattr(p_out, "__len__") else p_out)
            else:
                raise TypeError(f"Unsupported model type in policy: {type(model)}")
            member_probs.append(p)

        p_array = np.array(member_probs, dtype=float)

        # Apply calibration if calibrators are provided
        if self.calibrators is not None:
            cal = None
            if hasattr(self.calibrators, "predict_proba"):
                cal = self.calibrators
            elif isinstance(self.calibrators, dict):
                # Check for exact tuple match of sorted observed features
                obs_tuple = tuple(sorted(observed_cols))
                if obs_tuple in self.calibrators:
                    cal = self.calibrators[obs_tuple]
                else:
                    # Look for best subset or matching scenario name
                    for k, c in self.calibrators.items():
                        if isinstance(k, (list, tuple, set)) and set(k) == set(observed_cols):
                            cal = c
                            break
                        elif isinstance(k, str) and k in ["recalibrated", "default", "global"]:
                            cal = c
            if cal is not None and hasattr(cal, "predict_proba"):
                # Apply calibrator to logits of probabilities
                logits = np.log(np.clip(p_array, 1e-6, 1 - 1e-6) / (1 - np.clip(p_array, 1e-6, 1 - 1e-6))).reshape(-1, 1)
                p_array = cal.predict_proba(logits)[:, 1]

        p_mean = float(p_array.mean())

        if self.is_ensemble:
            # Shape (n_members, 1)
            cas = confidence_attribution(p_array[:, np.newaxis])
            pred_entropy = float(cas["predictive_entropy"][0])
            disagreement = float(cas["ensemble_disagreement"][0])
        else:
            pred_entropy = float(entropy(p_mean))
            disagreement = 0.0

        return p_mean, pred_entropy, disagreement

    def should_acquire(self, pred_entropy: float, disagreement: float) -> bool:
        """Evaluate whether current uncertainty signals satisfy acquisition criteria."""
        trigger = False
        if self.tau_entropy_acq is not None and pred_entropy >= self.tau_entropy_acq:
            trigger = True
        if self.tau_disagreement_acq is not None and disagreement >= self.tau_disagreement_acq:
            trigger = True
        return trigger

    def should_abstain(self, p_mean: float, pred_entropy: float) -> Tuple[bool, str]:
        """Evaluate whether prediction uncertainty or margin breaches abstention thresholds."""
        if self.tau_entropy_abstain is not None and pred_entropy >= self.tau_entropy_abstain:
            return True, f"predictive_entropy ({pred_entropy:.4f}) >= tau_entropy_abstain ({self.tau_entropy_abstain:.4f})"
        if self.margin_abstain is not None:
            margin = abs(p_mean - self.decision_threshold)
            if margin <= self.margin_abstain:
                return True, f"decision_margin ({margin:.4f}) <= margin_abstain ({self.margin_abstain:.4f})"
        return False, ""

    def decide(
        self,
        observed: Dict[str, Any],
        ground_truth_provider: Optional[Callable[[str], Any]] = None,
    ) -> DecisionRecord:
        """Execute the adaptive policy on a single record.

        Parameters
        ----------
        observed : dict
            Initial observed features and their values. Unobserved features are NOT provided.
        ground_truth_provider : callable, optional
            Function `provider(feature_name) -> value` representing safe external acquisition.
            Called strictly on-demand only after `recommend()` determines the feature to acquire.
            Never accessed if acquisition is not triggered.

        Returns
        -------
        DecisionRecord
        """
        # Validate that observed fields are valid features and non-null
        current_observed = {
            k: v for k, v in observed.items() if k in FEATURES and not pd.isna(v)
        }
        initial_features = tuple(sorted(current_observed.keys()))
        acquired: List[str] = []
        steps_taken = 0

        # Initial prediction & CAS uncertainty estimation
        p_mean, pred_entropy, disagreement = self._predict_prob(current_observed)

        # Sequential acquisition loop
        while steps_taken < self.max_acquisition_steps:
            if not self.should_acquire(pred_entropy, disagreement):
                break

            if self.donor_engine is None or ground_truth_provider is None:
                # Acquisition warranted but no engine/provider supplied; proceed to decide with current info
                break

            # Recommend next feature based on conditional donors
            recommended_feature, eval_result = self.donor_engine.recommend(current_observed)
            if recommended_feature is None:
                break

            if recommended_feature not in self.allowed_acquisition_features:
                # Find best allowed feature with non-zero gain
                gains = eval_result.get("expected_information_gain", {})
                allowed_candidates = [
                    (feat, gain)
                    for feat, gain in gains.items()
                    if feat in self.allowed_acquisition_features and feat not in current_observed
                ]
                if not allowed_candidates:
                    break
                allowed_candidates.sort(key=lambda x: x[1], reverse=True)
                recommended_feature, top_gain = allowed_candidates[0]
            else:
                top_gain = eval_result.get("expected_information_gain", {}).get(recommended_feature, 0.0)

            if top_gain < self.min_information_gain:
                break

            # Acquire true value from external provider
            new_value = ground_truth_provider(recommended_feature)
            if pd.isna(new_value):
                # Unobtainable or missing even upon inquiry; stop querying
                break

            current_observed[recommended_feature] = new_value
            acquired.append(recommended_feature)
            steps_taken += 1

            # Recalculate uncertainty after acquisition
            p_mean, pred_entropy, disagreement = self._predict_prob(current_observed)

        # Evaluate abstention
        abstain, abstain_reason = self.should_abstain(p_mean, pred_entropy)
        final_features = tuple(sorted(current_observed.keys()))

        if abstain:
            return DecisionRecord(
                action="abstain",
                prediction=None,
                probability=p_mean,
                confidence=1.0 - pred_entropy,
                predictive_entropy=pred_entropy,
                ensemble_disagreement=disagreement,
                initial_observed_features=initial_features,
                final_observed_features=final_features,
                acquired_features=tuple(acquired),
                acquisition_steps=steps_taken,
                reason=f"Abstained: {abstain_reason}",
            )

        binary_pred = 1 if p_mean >= self.decision_threshold else 0
        classify_reason = f"Classified as {binary_pred} (p={p_mean:.4f}, threshold={self.decision_threshold:.4f})"
        if steps_taken > 0:
            classify_reason += f" after acquiring {steps_taken} feature(s): {', '.join(acquired)}"

        return DecisionRecord(
            action="classify",
            prediction=binary_pred,
            probability=p_mean,
            confidence=1.0 - pred_entropy,
            predictive_entropy=pred_entropy,
            ensemble_disagreement=disagreement,
            initial_observed_features=initial_features,
            final_observed_features=final_features,
            acquired_features=tuple(acquired),
            acquisition_steps=steps_taken,
            reason=classify_reason,
        )
