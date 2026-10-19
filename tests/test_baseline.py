import unittest
import numpy as np
from truststroke.baseline import clean, synthetic, split_indices, build_model, metrics, bootstrap_ci

class BaselineTests(unittest.TestCase):
    def test_cleaning_and_target_exclusion(self):
        d = synthetic(100)
        d.loc[0, 'CVDSTRK3'] = 7
        d.loc[1, 'PHYSHLTH'] = 88
        d.loc[2, '_BMI5'] = 2500
        d['_LLCPWT'] = 1
        X, y, design = clean(d)
        self.assertEqual(len(y), 99)
        self.assertNotIn('CVDSTRK3', X)
        self.assertNotIn('_LLCPWT', X)
        self.assertIn('_LLCPWT', design)
        self.assertEqual(X.loc[1, 'PHYSHLTH'], 0)
        self.assertEqual(X.loc[2, '_BMI5'], 25)

    def test_partition_is_disjoint_and_reproducible(self):
        _, y, _ = clean(synthetic())
        a = split_indices(y)
        b = split_indices(y)
        self.assertEqual(len(set(np.concatenate(a))), len(y))
        for x, z in zip(a, b):
            np.testing.assert_array_equal(x, z)

    def test_test_values_cannot_change_fitted_preprocessing(self):
        X, y, _ = clean(synthetic())
        train, val, test = split_indices(y)
        model = build_model().fit(X.iloc[train], y.iloc[train])
        imputer = model[0].named_transformers_['numeric'][0]
        before = imputer.statistics_.copy()
        altered = X.iloc[test].copy()
        altered['_BMI5'] = 1e12
        altered['SEXVAR'] = 999
        model.predict_proba(altered)
        np.testing.assert_array_equal(imputer.statistics_, before)
        np.testing.assert_allclose(before, X.iloc[train][['PHYSHLTH', 'MENTHLTH', '_BMI5']].median().values)

    def test_metrics_edge_cases(self):
        m = metrics([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertEqual(m['auroc'], 1)
        self.assertEqual(m['brier'], 0)
        self.assertEqual(m['ece'], 0)
        self.assertIsNone(metrics([0, 0], [.1, .2])['auroc'])
        with self.assertRaises(ValueError):
            metrics([0, 1], [.2, np.nan])

    def test_bootstrap_repeatable(self):
        args = ([0, 1] * 10, [.2, .8] * 10, .5)
        self.assertEqual(bootstrap_ci(*args, repetitions=10), bootstrap_ci(*args, repetitions=10))

if __name__ == '__main__':
    unittest.main()
