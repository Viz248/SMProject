import unittest
import numpy as np
import torch
from truststroke.dime_matched import BinaryBrier

class MatchedDimeTests(unittest.TestCase):
    def test_binary_brier_reductions_and_gradient(self):
        logits=torch.tensor([[1.,2.],[3.,-1.]],requires_grad=True)
        y=torch.tensor([1,0])
        expected=(logits.softmax(1)[:,1]-y.float())**2
        torch.testing.assert_close(BinaryBrier('none')(logits,y),expected)
        mean=BinaryBrier()(logits,y)
        torch.testing.assert_close(mean,expected.mean())
        mean.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0,1]),0)
        self.assertGreater(float(logits.grad[1,1]),0)

    def test_logits_shift_preserves_probability_loss(self):
        logits=torch.tensor([[1.,2.],[-3.,1.]])
        y=torch.tensor([1,0])
        torch.testing.assert_close(BinaryBrier('none')(logits,y),BinaryBrier('none')(logits+12,y))

    def test_saved_predictor_and_value_reproduce_predictions(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        import joblib
        from truststroke.dime_matched import EncodedPredictor
        from truststroke.dime_pilot import Value,network
        from truststroke.dime_teacher_diagnostics import restore_predictor,restore_value
        X=pd.DataFrame({'category':[1,2,1,2],'number':[2.,4.,1.,3.]})
        original=EncodedPredictor(X,['category']);original.net.eval()
        m=np.array([[1,0],[0,1],[1,1],[0,0]],dtype=np.float32)
        value=Value.__new__(Value);value.predictor=original;value.scale=.03
        value.net=network(original.net[0].in_features,2);value.net.eval()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);joblib.dump(original.prep,root/'preprocessor.joblib')
            torch.save({'predictor':original.net.state_dict(),'groups':original.groups},root/'selected_state.pt')
            torch.save({'state_dict':value.net.state_dict(),'scale':value.scale},root/'value.pt')
            loaded=restore_predictor(root);loaded_value=restore_value(root/'value.pt',loaded)
            np.testing.assert_allclose(original.predict(X,m),loaded.predict(X,m))
            np.testing.assert_allclose(value.predict(X,m),loaded_value.predict(X,m))
