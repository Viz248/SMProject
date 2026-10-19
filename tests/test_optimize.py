import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from truststroke.baseline import clean, synthetic
from truststroke.survey import weighted_points
from truststroke.availability_refit import fit_subset
from truststroke.ensemble_search import search
from truststroke.evaluate_locked import evaluate
from truststroke.optimize import (partitions, fit_candidate, probability_logit, run, Predictor)

class OptimizationTests(unittest.TestCase):
    def test_five_partitions_cover_records_once(self):
        _, y, _ = clean(synthetic(2000))
        split = partitions(y, 42)
        all_ids = np.concatenate(list(split.values()))
        self.assertEqual(len(np.unique(all_ids)), len(y))
        self.assertEqual(len(all_ids), len(y))
        for key, ids in split.items():
            np.testing.assert_array_equal(ids, partitions(y, 42)[key])
            self.assertEqual(set(y.iloc[ids]), {0, 1})

    def test_weighted_metrics(self):
        result = weighted_points([0, 1], [.1, .9], [3, 1], .5)
        self.assertEqual(result['prevalence'], .25)
        self.assertAlmostEqual(result['brier'], .01)
        self.assertEqual(result['sensitivity'], 1)
        with self.assertRaises(ValueError):
            weighted_points([0, 1], [.1, .9], [1, -1], .5)

    def test_refuse_tuning_after_test_access(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'development.json').write_text('{"test_accessed":true}')
            with self.assertRaises(ValueError):
                search(root, 'must-not-be-read.XPT')

    def test_refuse_overwriting_test_result(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root, 'locked_test.json')
            marker.write_text('{"status":"existing"}')
            with self.assertRaises(FileExistsError):
                evaluate(root, 'must-not-be-read.XPT')
            self.assertEqual(marker.read_text(), '{"status":"existing"}')

    def test_subset_retraining_cannot_use_removed_features(self):
        X,y,_=clean(synthetic(1000))
        with threadpool_limits(limits=1):
            model=fit_subset(['SEXVAR','_AGEG5YR'],X.iloc[:600],y.iloc[:600],
                    X.iloc[600:750],y.iloc[600:750],X.iloc[750:900],y.iloc[750:900])
            changed=X.iloc[900:].copy();changed['GENHLTH']=999;changed['_BMI5']=1e12
            np.testing.assert_allclose(model.predict_proba(changed),model.predict_proba(X.iloc[900:]))

    def test_logit_finite_at_endpoints(self):
        self.assertTrue(np.isfinite(probability_logit([0, 1, .5])).all())
        self.assertAlmostEqual(probability_logit([.5])[0, 0], 0)

    def test_bottleneck_roundtrip_and_shape(self):
        X, y, _ = clean(synthetic(1000))
        s = partitions(y, 4)
        with threadpool_limits(limits=1):
            model, _ = fit_candidate(dict(family='bottleneck', widths=(64, 32, 15, 16), alpha=.01),
                X.iloc[s['fit']], y.iloc[s['fit']], X.iloc[s['stop']], y.iloc[s['stop']], 4, 3)
        self.assertEqual(model.transform(X.iloc[s['select']]).shape[1], 15)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(model.predict_proba(X.iloc[s['select']]),
                                       joblib.load(path).predict_proba(X.iloc[s['select']]))

    def test_default_search_does_not_predict_on_test(self):
        args = SimpleNamespace(seed=42, smoke=True, data=None, fit_limit=0, epochs=2,
                               threads=1, evaluate_test=False, bootstrap=5)
        _, y, _ = clean(synthetic(2000, 42))
        forbidden = set(partitions(y, 42)['test'])
        original = Predictor.predict_proba
        def guarded(model, X):
            self.assertTrue(forbidden.isdisjoint(X.index))
            return original(model, X)
        with tempfile.TemporaryDirectory() as root:
            args.output = str(Path(root) / 'run')
            with patch('truststroke.optimize.candidates', return_value=[dict(name='lr', family='logistic', C=1)]), \
                 patch.object(Predictor, 'predict_proba', guarded), threadpool_limits(limits=1):
                report = run(args)
            self.assertFalse(report['test_accessed'])
            self.assertNotIn('test', report)
            model = joblib.load(Path(args.output) / 'winner.joblib')
            self.assertIsNotNone(model.calibrator)

if __name__ == '__main__':
    unittest.main()
