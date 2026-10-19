import unittest
import numpy as np
import torch
from truststroke.dime_pilot import state, masks, losses, paired_interval

class DimePilotTests(unittest.TestCase):
    def test_unobserved_values_cannot_enter_network(self):
        groups=torch.tensor([0,0,1,2]);m=torch.tensor([[1.,0.,1.]])
        a=torch.tensor([[2.,3.,4.,5.]]);b=a.clone();b[0,2]=9999
        torch.testing.assert_close(state(a,m,groups),state(b,m,groups))
        self.assertEqual(state(a,m,groups).shape[1],7)

    def test_initial_features_and_available_candidates(self):
        m=masks(100,15,[0,1],np.random.default_rng(42))
        self.assertTrue(np.all(m[:,:2]==1));self.assertTrue(np.all(m.sum(1)<=7))
        self.assertTrue(np.all((1-m).sum(1)>=8))

    def test_brier_improvement_sign(self):
        y=np.array([0,1]);before=losses(y,np.array([.5,.5]),'brier')
        after=losses(y,np.array([.1,.9]),'brier')
        np.testing.assert_allclose(before-after,.24)
        np.testing.assert_allclose(paired_interval(before-after,42),[.24,.24])
