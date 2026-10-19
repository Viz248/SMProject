import unittest
import numpy as np
import pandas as pd
from truststroke.baseline import FEATURES
from truststroke.adaptive_layer import AdaptiveDecisionPolicy, DecisionRecord


class DummyPredictor:
    """Mock model that returns fixed or feature-dependent probabilities."""
    def __init__(self, prob=0.10):
        self.prob = prob

    def predict_proba(self, X):
        # Return Nx2 array
        n = len(X)
        p = np.full(n, self.prob)
        return np.column_stack([1 - p, p])


class LeakageDetectorModel:
    """Model that records the exact columns and values received at prediction time."""
    def __init__(self, p_fn):
        self.p_fn = p_fn
        self.inspected_records = []

    def predict_proba(self, X):
        for idx in range(len(X)):
            row = X.iloc[idx].to_dict()
            self.inspected_records.append(row)
        p = np.array([self.p_fn(X.iloc[i]) for i in range(len(X))])
        return np.column_stack([1 - p, p])


class DummyDonorEngine:
    """Mock ConditionalDonors engine."""
    def __init__(self, recommendation='_BMI5', gain=0.05):
        self.recommendation = recommendation
        self.gain = gain
        self.calls = []

    def recommend(self, observed):
        self.calls.append(dict(observed))
        return self.recommendation, {
            'probability': 0.15,
            'predictive_entropy': 0.6,
            'expected_information_gain': {self.recommendation: self.gain}
        }


class TestAdaptiveDecisionPolicy(unittest.TestCase):
    def setUp(self):
        # A synthetic complete record dictionary
        self.complete_record = {
            'SEXVAR': 1.0, '_AGEG5YR': 5.0, 'GENHLTH': 2.0, 'EXERANY2': 1.0,
            '_SMOKER3': 4.0, 'DIABETE4': 3.0, 'CVDINFR4': 2.0, 'CVDCRHD4': 2.0,
            'CHCKDNY2': 2.0, 'HAVARTH4': 2.0, 'EDUCA': 6.0, 'INCOME3': 8.0,
            'PHYSHLTH': 0.0, 'MENTHLTH': 0.0, '_BMI5': 2500.0
        }

    def test_complete_feature_dictionary(self):
        model = DummyPredictor(prob=0.08)
        policy = AdaptiveDecisionPolicy(model=model, decision_threshold=0.14)
        decision = policy.decide(self.complete_record)

        self.assertIsInstance(decision, DecisionRecord)
        self.assertEqual(decision.action, 'classify')
        self.assertEqual(decision.prediction, 0)
        self.assertAlmostEqual(decision.probability, 0.08)
        self.assertEqual(decision.acquisition_steps, 0)
        self.assertEqual(len(decision.initial_observed_features), len(FEATURES))
        self.assertIn("Classified as 0", decision.reason)

    def test_sparse_feature_dictionary(self):
        sparse_input = {'SEXVAR': 1.0, '_AGEG5YR': 5.0}
        model = DummyPredictor(prob=0.20)
        policy = AdaptiveDecisionPolicy(model=model, decision_threshold=0.14)
        decision = policy.decide(sparse_input)

        self.assertEqual(decision.action, 'classify')
        self.assertEqual(decision.prediction, 1)
        self.assertAlmostEqual(decision.probability, 0.20)
        self.assertEqual(decision.initial_observed_features, ('SEXVAR', '_AGEG5YR'))
        self.assertEqual(decision.final_observed_features, ('SEXVAR', '_AGEG5YR'))
        self.assertEqual(decision.acquired_features, ())

    def test_high_entropy_triggers_acquisition(self):
        # p=0.5 produces maximum binary entropy (1.0 bit)
        model = DummyPredictor(prob=0.5)
        donor = DummyDonorEngine(recommendation='PHYSHLTH', gain=0.08)
        
        provider_calls = []
        def provider(feat):
            provider_calls.append(feat)
            return 10.0

        policy = AdaptiveDecisionPolicy(
            model=model,
            decision_threshold=0.14,
            tau_entropy_acq=0.5,
            max_acquisition_steps=1,
            donor_engine=donor
        )
        decision = policy.decide({'SEXVAR': 1.0}, ground_truth_provider=provider)

        self.assertEqual(len(donor.calls), 1)
        self.assertEqual(provider_calls, ['PHYSHLTH'])
        self.assertEqual(decision.acquisition_steps, 1)
        self.assertIn('PHYSHLTH', decision.acquired_features)
        self.assertIn('PHYSHLTH', decision.final_observed_features)

    def test_low_uncertainty_skips_acquisition(self):
        # p=0.02 has low entropy (~0.14 bits)
        model = DummyPredictor(prob=0.02)
        donor = DummyDonorEngine(recommendation='PHYSHLTH', gain=0.08)

        provider_calls = []
        def provider(feat):
            provider_calls.append(feat)
            return 10.0

        policy = AdaptiveDecisionPolicy(
            model=model,
            decision_threshold=0.14,
            tau_entropy_acq=0.5,
            max_acquisition_steps=2,
            donor_engine=donor
        )
        decision = policy.decide({'SEXVAR': 1.0}, ground_truth_provider=provider)

        self.assertEqual(len(donor.calls), 0)
        self.assertEqual(len(provider_calls), 0)
        self.assertEqual(decision.acquisition_steps, 0)
        self.assertEqual(decision.acquired_features, ())

    def test_abstention_on_entropy_and_margin(self):
        # Case A: Entropy abstention
        model_entropy = DummyPredictor(prob=0.5)
        policy_entropy = AdaptiveDecisionPolicy(
            model=model_entropy,
            decision_threshold=0.14,
            tau_entropy_abstain=0.8
        )
        decision_e = policy_entropy.decide({'SEXVAR': 1.0})
        self.assertEqual(decision_e.action, 'abstain')
        self.assertIsNone(decision_e.prediction)
        self.assertIn("tau_entropy_abstain", decision_e.reason)

        # Case B: Margin abstention (|0.145 - 0.14| = 0.005 <= margin_abstain=0.01)
        model_margin = DummyPredictor(prob=0.145)
        policy_margin = AdaptiveDecisionPolicy(
            model=model_margin,
            decision_threshold=0.14,
            margin_abstain=0.01
        )
        decision_m = policy_margin.decide({'SEXVAR': 1.0})
        self.assertEqual(decision_m.action, 'abstain')
        self.assertIsNone(decision_m.prediction)
        self.assertIn("decision_margin", decision_m.reason)

    def test_single_model_disagreement_is_zero(self):
        model = DummyPredictor(prob=0.25)
        policy = AdaptiveDecisionPolicy(model=model)
        decision = policy.decide({'SEXVAR': 1.0})

        self.assertEqual(decision.ensemble_disagreement, 0.0)
        self.assertFalse(policy.is_ensemble)

    def test_ensemble_disagreement_calculation(self):
        model_a = DummyPredictor(prob=0.1)
        model_b = DummyPredictor(prob=0.9)
        policy = AdaptiveDecisionPolicy(
            model_or_ensemble=[model_a, model_b],
            tau_disagreement_acq=0.1
        )
        self.assertTrue(policy.is_ensemble)
        decision = policy.decide({'SEXVAR': 1.0})
        # Mean p is 0.5 (entropy 1.0), within is ~0.469, between disagreement > 0.5 bits
        self.assertGreater(decision.ensemble_disagreement, 0.4)

    def test_determinism(self):
        model = DummyPredictor(prob=0.12)
        policy = AdaptiveDecisionPolicy(model=model, decision_threshold=0.14)
        input_data = {'SEXVAR': 1.0, 'GENHLTH': 3.0}

        dec_1 = policy.decide(input_data)
        dec_2 = policy.decide(input_data)

        self.assertEqual(dec_1, dec_2)

    def test_decision_record_structure(self):
        model = DummyPredictor(prob=0.15)
        policy = AdaptiveDecisionPolicy(model=model, decision_threshold=0.14)
        decision = policy.decide({'SEXVAR': 2.0})

        self.assertEqual(decision.action, 'classify')
        self.assertEqual(decision.prediction, 1)
        self.assertAlmostEqual(decision.probability, 0.15)
        self.assertAlmostEqual(decision.confidence, 1.0 - decision.predictive_entropy)
        self.assertEqual(decision.acquisition_steps, 0)
        self.assertIn("Classified as 1", decision.reason)

    def test_feature_leakage_protection(self):
        """Verify that unobserved features are never passed to the model before acquisition."""
        def p_fn(row):
            # If CVDINFR4 is observed, return 0.25, else 0.5 (high entropy to trigger acq)
            if not pd.isna(row['CVDINFR4']):
                return 0.25
            return 0.5

        detector = LeakageDetectorModel(p_fn)
        donor = DummyDonorEngine(recommendation='CVDINFR4', gain=0.1)

        ground_truth = {
            'SEXVAR': 1.0,
            'CVDINFR4': 2.0,
            '_BMI5': 2700.0
        }

        queried_features = []
        def safe_provider(feat):
            queried_features.append(feat)
            return ground_truth[feat]

        policy = AdaptiveDecisionPolicy(
            model=detector,
            decision_threshold=0.14,
            tau_entropy_acq=0.6,
            max_acquisition_steps=1,
            donor_engine=donor
        )

        # Initial input has only SEXVAR. CVDINFR4 and _BMI5 are unobserved.
        initial_input = {'SEXVAR': 1.0}
        decision = policy.decide(initial_input, ground_truth_provider=safe_provider)

        # The detector was called twice: once initially, once after acquisition
        self.assertEqual(len(detector.inspected_records), 2)

        # In pass 1 (before acquisition), all features except SEXVAR MUST be NaN
        record_pass_1 = detector.inspected_records[0]
        self.assertEqual(record_pass_1['SEXVAR'], 1.0)
        self.assertTrue(pd.isna(record_pass_1['CVDINFR4']))
        self.assertTrue(pd.isna(record_pass_1['_BMI5']))
        self.assertEqual(queried_features, ['CVDINFR4'])

        # In pass 2 (after acquisition), ONLY CVDINFR4 became observed; _BMI5 remains NaN
        record_pass_2 = detector.inspected_records[1]
        self.assertEqual(record_pass_2['SEXVAR'], 1.0)
        self.assertEqual(record_pass_2['CVDINFR4'], 2.0)
        self.assertTrue(pd.isna(record_pass_2['_BMI5']))

        # Decision record reflects that only CVDINFR4 was acquired
        self.assertEqual(decision.acquired_features, ('CVDINFR4',))
        self.assertEqual(decision.initial_observed_features, ('SEXVAR',))
        self.assertEqual(decision.final_observed_features, ('CVDINFR4', 'SEXVAR'))

    def test_missing_callbacks_and_empty_inputs(self):
        # Case 1: Empty input dictionary
        model = DummyPredictor(prob=0.05)
        policy = AdaptiveDecisionPolicy(model=model)
        decision = policy.decide({})
        self.assertEqual(decision.action, 'classify')
        self.assertEqual(decision.initial_observed_features, ())

        # Case 2: High entropy with donor engine but missing ground_truth_provider
        donor = DummyDonorEngine(recommendation='PHYSHLTH', gain=0.1)
        policy_no_cb = AdaptiveDecisionPolicy(
            model=DummyPredictor(prob=0.5),
            tau_entropy_acq=0.5,
            max_acquisition_steps=2,
            donor_engine=donor
        )
        decision_no_cb = policy_no_cb.decide({'SEXVAR': 1.0}, ground_truth_provider=None)
        # Should gracefully proceed to decide without failing or acquiring
        self.assertEqual(decision_no_cb.acquisition_steps, 0)
        self.assertEqual(decision_no_cb.acquired_features, ())

        # Case 3: High entropy with provider but missing donor engine
        policy_no_engine = AdaptiveDecisionPolicy(
            model=DummyPredictor(prob=0.5),
            tau_entropy_acq=0.5,
            max_acquisition_steps=2,
            donor_engine=None
        )
        decision_no_engine = policy_no_engine.decide(
            {'SEXVAR': 1.0},
            ground_truth_provider=lambda f: 1.0
        )
        self.assertEqual(decision_no_engine.acquisition_steps, 0)

        # Case 4: Invalid/unsupported feature names in input dictionary are safely ignored
        decision_invalid_feat = policy.decide({'NON_EXISTENT_FEATURE': 999, 'SEXVAR': 1.0})
        self.assertEqual(decision_invalid_feat.initial_observed_features, ('SEXVAR',))


class TestAdaptiveEvaluationHelpers(unittest.TestCase):
    def test_mask_record_removes_unobserved(self):
        from truststroke.evaluate_adaptive import mask_record
        full_rec = {'SEXVAR': 1.0, '_AGEG5YR': 5.0, 'GENHLTH': 2.0}
        masked = mask_record(full_rec, ['SEXVAR'])
        self.assertEqual(masked, {'SEXVAR': 1.0})

    def test_paired_bootstrap_brier_diff(self):
        from truststroke.evaluate_adaptive import paired_bootstrap_brier_diff
        y = np.array([0, 0, 1, 1])
        p1 = np.array([0.1, 0.1, 0.9, 0.9])
        p2 = np.array([0.2, 0.2, 0.8, 0.8])
        res = paired_bootstrap_brier_diff(y, p1, p2, replicates=50, seed=42)
        self.assertIn('mean_difference', res)
        self.assertIn('ci95_low', res)
        self.assertIn('ci95_high', res)
        self.assertGreater(res['mean_difference'], 0)  # p2 is worse than p1, diff > 0

    def test_run_experiment_smoke(self):
        import tempfile
        from truststroke.evaluate_adaptive import run_experiment
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                smoke=True,
                data=None,
                smoke_records=120,
                limit=None,
                tune_limit=50,
                bootstrap=20,
                seed=42,
                output=tmpdir
            )
            summary = run_experiment(args)
            self.assertEqual(summary['status'], 'COMPLETED')
            self.assertEqual(summary['data_mode'], 'SYNTHETIC_SMOKE')
            self.assertIn('scenarios', summary)
            self.assertIn('demographics', summary['scenarios'])
            self.assertIn('adaptive_policy', summary['scenarios']['demographics'])


if __name__ == '__main__':
    unittest.main()

