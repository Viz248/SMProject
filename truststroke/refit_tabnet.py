"""Recover a predictor checkpoint by deterministic SAME-setting refit, not retuning."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .baseline import clean
from .optimize import load_data,probability_logit,sha256
from .tabnet_search import fit_tabnet


def run(a):
    root=Path(a.run);ledger=json.loads((root/'development.json').read_text())
    raw,digest=load_data(a.data)
    if digest!=ledger['dataset_sha256']:raise ValueError('Dataset mismatch')
    X,y,_=clean(raw)
    split=np.load(root/'splits.npz');fit,stop,select,cal=[split[k] for k in ['fit','stop','select','calibrate']]
    row=next(r for r in ledger['candidates'] if r['configuration']['name']==ledger['winner'])
    model=fit_tabnet(row['configuration'],X.iloc[fit],y.iloc[fit],X.iloc[stop],y.iloc[stop],ledger['epochs'],ledger['seed'])
    p=model.predict_proba(X.iloc[select])[:,1]
    ap=float(average_precision_score(y.iloc[select],p))
    if not np.isclose(ap,row['selection']['average_precision'],atol=1e-8,rtol=0):
        raise RuntimeError(f'Refit AP {ap} differs from recorded {row["selection"]["average_precision"]}; no silent replacement')
    model.calibrator=LogisticRegression(C=1,max_iter=1000,random_state=ledger['seed']).fit(
        probability_logit(model.predict_proba(X.iloc[cal])[:,1]),y.iloc[cal])
    joblib.dump(model,root/'winner.recovered.joblib')
    restored=joblib.load(root/'winner.recovered.joblib')
    np.testing.assert_allclose(restored.predict_proba(X.iloc[select[:128]]),model.predict_proba(X.iloc[select[:128]]),atol=1e-8,rtol=0)
    (root/'winner.recovered.joblib').replace(root/'winner.joblib')
    ledger['checkpoint_recovery']=dict(method='Native checkpoint serialization; identical configuration/seed/data refit',
        selection_ap=ap,original_selection_ap=row['selection']['average_precision'],
        model_sha256=sha256(root/'winner.joblib'),roundtrip_predictions_verified=True,
        note='No hyperparameters or calibration procedure changed after partial external evaluation')
    (root/'development.json').write_text(json.dumps(ledger,indent=2,allow_nan=False))
    print('Recovered native TabNet checkpoint; selection AP exactly reproduced',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--run',required=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(p.parse_args())
