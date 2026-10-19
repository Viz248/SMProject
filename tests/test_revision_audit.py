import unittest
import numpy as np
from scipy.special import expit,logit
from truststroke.revision_audit import calibration_diagnostics
class CalibrationTests(unittest.TestCase):
 def test_known_relation(self):
  p=np.linspace(.01,.8,150);y=expit(.3+.7*logit(p));r=calibration_diagnostics(y,p,np.ones(len(p)))
  self.assertAlmostEqual(r['intercept'],.3,places=5);self.assertAlmostEqual(r['slope'],.7,places=5)
 def test_perfect_calibration_weight_scaling(self):
  p=np.linspace(.02,.4,100);w=np.linspace(1,10,100);a=calibration_diagnostics(p,p,w);b=calibration_diagnostics(p,p,1000*w)
  self.assertAlmostEqual(a['citl'],0,places=8);self.assertAlmostEqual(a['slope'],1,places=8);self.assertAlmostEqual(a['weighted_ece'],0,places=10);self.assertAlmostEqual(a['citl'],b['citl'],places=10)
