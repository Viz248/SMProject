import unittest
import numpy as np
import pandas as pd
from truststroke.trust import (entropy,confidence_attribution,decision_stability,
 permutation_shapley,resource_adaptability,mean_interval)
from truststroke.acquisition import ConditionalDonors
from truststroke.baseline import clean,synthetic,FEATURES

class TrustTests(unittest.TestCase):
    def test_entropy_and_decomposition(self):
        np.testing.assert_allclose(entropy([0,.5,1]),[0,1,0],atol=1e-12)
        c=confidence_attribution([[.1,.1],[.9,.1]])
        np.testing.assert_allclose(c['predictive_entropy'],c['mean_member_entropy']+c['ensemble_disagreement'])
        self.assertGreater(c['ensemble_disagreement'][0],0)
        self.assertAlmostEqual(c['ensemble_disagreement'][1],0)
        with self.assertRaises(ValueError):entropy([1.1])

    def test_stability_sign_scale_and_zeros(self):
        a=np.array([1.,-2,3])
        self.assertAlmostEqual(decision_stability(a,10*a)['dsi'],1)
        self.assertAlmostEqual(decision_stability(a,-a)['dsi'],0)
        self.assertIsNone(decision_stability(a*0,a*0)['dsi'])
        self.assertEqual(decision_stability(a,a*0)['dsi'],0)
        with self.assertRaises(ValueError):decision_stability([np.nan],[1])

    def test_shapley_linear_exactness_and_efficiency(self):
        x=pd.DataFrame({'a':[1.,2.],'b':[3.,4.]}); bg=pd.DataFrame({'a':[0.],'b':[0.]})
        predict=lambda z:.1*z.a.to_numpy()+.05*z.b.to_numpy()
        a,base,residual=permutation_shapley(predict,x,bg,permutations=8)
        np.testing.assert_allclose(a,np.column_stack([.1*x.a,.05*x.b]),atol=1e-12)
        np.testing.assert_allclose(residual,0,atol=1e-12)
        a2,_,_=permutation_shapley(predict,x,bg,permutations=8)
        np.testing.assert_allclose(a,a2)

    def test_resource_edges(self):
        r=resource_adaptability(.1,[.1,.2,.3,.05],.2)
        np.testing.assert_allclose(r['retention'],[1,0,0,1])
        self.assertEqual(r['rai'],.5)
        self.assertIsNone(resource_adaptability(.2,[.3],.1)['rai'])
        with self.assertRaises(ValueError):resource_adaptability(.1,[.2],.3,weights=[-1])

    def test_conditional_acquisition_respects_known_values(self):
        d=clean(synthetic(200))[0].dropna()
        engine=ConditionalDonors(lambda z:np.where(z['CVDINFR4']==1,.8,.2),d,neighbors=64)
        recommendation,result=engine.recommend({'SEXVAR':1})
        self.assertNotIn('SEXVAR',result['expected_information_gain'])
        self.assertEqual(recommendation,'CVDINFR4')
        complete,_=engine.distribution({'SEXVAR':1})
        self.assertTrue((complete.SEXVAR==1).all())
        self.assertLessEqual(max(result['expected_information_gain'].values()),result['predictive_entropy']+1e-12)
        self.assertIsNone(engine.recommend(d.iloc[0].to_dict())[0])

    def test_confidence_mean_interval_reproducible(self):
        self.assertEqual(mean_interval([1,2,3]),mean_interval([1,2,3]))

if __name__=='__main__':unittest.main()
