import unittest
import numpy as np
import pandas as pd
from truststroke.availability_study import AvailabilityModel,linearized_mean

class AvailabilityStudyTests(unittest.TestCase):
    def design(self):
        return pd.DataFrame({'_STATE':[1]*4,'_STSTR':[1,1,2,2],'_PSU':[1,2,3,4],'_LLCPWT':[1.]*4})
    def test_stratified_psu_linearization_hand_calculation(self):
        r=linearized_mean(np.array([1.,3.,2.,4.]),self.design())
        self.assertAlmostEqual(r['estimate'],2.5)
        self.assertAlmostEqual(r['standard_error'],np.sqrt(.5))
        self.assertEqual(r['degrees_freedom'],2)
        self.assertEqual(r['singleton_strata'],0)
    def test_domain_retains_outside_psus(self):
        r=linearized_mean(np.array([1.,3.,2.,4.]),self.design(),[True,True,False,False])
        self.assertAlmostEqual(r['estimate'],2.)
        self.assertAlmostEqual(r['standard_error'],1.)
        self.assertEqual(r['psus'],4)
    def test_singleton_stratum_does_not_receive_confident_interval(self):
        d=self.design().iloc[:3]
        r=linearized_mean([1.,3.,2.],d)
        self.assertIsNone(r['ci95']);self.assertEqual(r['singleton_strata'],1)
    def test_group_mask_hides_original_values_and_missingness(self):
        X=pd.DataFrame({'SEXVAR':[1.,2.,1.,2.],'_BMI5':[20.,30.,np.nan,25.]})
        model=AvailabilityModel(['SEXVAR','_BMI5']);model.fit_preprocessor(X)
        a=model.encoded(X,['SEXVAR'])
        changed=X.copy();changed['_BMI5']=[100.,np.nan,-9.,200.]
        np.testing.assert_allclose(a,model.encoded(changed,['SEXVAR']))
        self.assertTrue(np.all(a[:,-1]==0))
    def test_subset_encoder_does_not_use_removed_variables(self):
        X=pd.DataFrame({'SEXVAR':[1.,2.],'_BMI5':[20.,30.]})
        model=AvailabilityModel(['SEXVAR']);model.fit_preprocessor(X)
        a=model.encoded(X,['SEXVAR']);X['_BMI5']=[9999.,-100.]
        np.testing.assert_array_equal(a,model.encoded(X,['SEXVAR']))

    def test_posthoc_average_singleton_sensitivity_hand_calculation(self):
        from truststroke.survey_sensitivity import average_singleton_interval
        design=pd.DataFrame({'_STATE':[1]*5,'_STSTR':[1,1,2,2,3],'_PSU':[1,2,3,4,5],'_LLCPWT':[1.]*5})
        r=average_singleton_interval([1.,3.,2.,4.,5.],design)
        self.assertAlmostEqual(r['estimate'],3.)
        self.assertAlmostEqual(r['variance_non_singleton'],.32)
        self.assertAlmostEqual(r['variance_singleton_imputed'],.16)
        self.assertAlmostEqual(r['se'],np.sqrt(.48))
        self.assertEqual(r['status'],'POST_HOC_SENSITIVITY_NOT_FROZEN_PRIMARY_CI')
