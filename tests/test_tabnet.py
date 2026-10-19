import importlib.util
import unittest
import tempfile
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
from truststroke.baseline import clean,synthetic

@unittest.skipUnless(importlib.util.find_spec('pytorch_tabnet') and importlib.util.find_spec('torch'),
                     'Optional TabNet dependencies not installed')
class TabNetTests(unittest.TestCase):
    def test_train_vocabulary_unknowns_and_numeric_isolation(self):
        from truststroke.tabnet_search import TabularEncoder
        X,_,_=clean(synthetic(200))
        encoder=TabularEncoder().fit(X.iloc[:100])
        before=encoder.scaler.mean_.copy()
        new=X.iloc[100:].copy();new['SEXVAR']=999;new['_BMI5']=1e12
        encoded=encoder.transform(new)
        self.assertEqual(encoded.shape,(100,18))
        self.assertTrue((encoded[:,0]==0).all())
        np.testing.assert_array_equal(before,encoder.scaler.mean_)
        self.assertTrue(np.isfinite(encoded).all())

    def test_native_checkpoint_reloads_in_fresh_process(self):
        import joblib,torch
        from threadpoolctl import threadpool_limits
        from truststroke.tabnet_search import fit_tabnet,configurations
        X,y,_=clean(synthetic(1000))
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            model=fit_tabnet(configurations()[0],X.iloc[:700],y.iloc[:700],X.iloc[700:850],y.iloc[700:850],epochs=1)
            expected=model.predict_proba(X.iloc[850:])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            joblib.dump(model,path/'model.joblib')
            X.iloc[850:].to_csv(path/'data.csv',index=False)
            np.save(path/'expected.npy',expected)
            script="""import sys,joblib,numpy as np,pandas as pd,torch
from pathlib import Path
torch.set_num_threads(1)
p=Path(sys.argv[1])
m=joblib.load(p/'model.joblib')
np.testing.assert_allclose(m.predict_proba(pd.read_csv(p/'data.csv')),np.load(p/'expected.npy'),atol=1e-7,rtol=0)
"""
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            subprocess.run([sys.executable,'-c',script,str(path)],check=True,env=env,capture_output=True)

if __name__=='__main__':unittest.main()
