"""Exploratory TabNet search restricted to existing development partitions.

The already inspected final test is excluded. No new final-test performance claim.
Optional dependency: torch and pytorch-tabnet (requirements-tabnet.txt).
"""
import argparse
from importlib.metadata import version
import json
import time
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .baseline import CATEGORICAL, NUMERIC, clean, metrics, synthetic
from .optimize import load_data, partitions, probability_logit


class AveragePrecision(Metric):
    def __init__(self):
        self._name, self._maximize = 'average_precision', True

    def __call__(self, y_true, y_score):
        return average_precision_score(y_true, y_score[:, 1])


class TabularEncoder:
    """Train-only categorical vocabulary; zero reserved for missing/unseen values."""
    def fit(self, X):
        self.vocabulary = {c: {v: i+1 for i, v in enumerate(sorted(X[c].dropna().unique()))}
                           for c in CATEGORICAL}
        self.medians = X[NUMERIC].median().fillna(0)
        self.scaler = StandardScaler().fit(X[NUMERIC].fillna(self.medians))
        self.cat_dims = [len(self.vocabulary[c])+1 for c in CATEGORICAL]
        return self

    def transform(self, X):
        categorical = np.column_stack([X[c].map(self.vocabulary[c]).fillna(0).to_numpy() for c in CATEGORICAL])
        numeric = self.scaler.transform(X[NUMERIC].fillna(self.medians))
        missing = X[NUMERIC].isna().to_numpy(dtype=float)
        return np.column_stack([categorical, numeric, missing]).astype(np.float32)


class TabNetPredictor:
    def __init__(self, encoder, model):
        self.encoder, self.model, self.calibrator = encoder, model, None

    def __getstate__(self):
        # Store native network/configuration, not training callbacks/custom metrics.
        # Native ZIP bytes are embedded in the predictor for a single portable artifact.
        with tempfile.TemporaryDirectory() as directory:
            archive=self.model.save_model(str(Path(directory)/'network'))
            payload=Path(archive).read_bytes()
        return dict(encoder=self.encoder,calibrator=self.calibrator,native_archive=payload)

    def __setstate__(self,state):
        self.encoder,self.calibrator=state['encoder'],state['calibrator']
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'network.zip';archive.write_bytes(state['native_archive'])
            self.model=TabNetClassifier(device_name='cpu',verbose=0)
            self.model.load_model(str(archive))

    def predict_proba(self, X):
        p = self.model.predict_proba(self.encoder.transform(X))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(probability_logit(p))[:, 1]
        return np.column_stack([1-p, p])


def configurations():
    return [dict(name='tabnet8', n_d=8, n_steps=3, lambda_sparse=.001, weight_decay=.0001),
            dict(name='tabnet16', n_d=16, n_steps=3, lambda_sparse=.001, weight_decay=.001),
            dict(name='tabnet16_sparse', n_d=16, n_steps=4, lambda_sparse=.01, weight_decay=.001)]


def fit_tabnet(spec, X, y, stop_X, stop_y, epochs=30, seed=42, batch_size=4096):
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TabularEncoder().fit(X)
    model = TabNetClassifier(n_d=spec['n_d'], n_a=spec['n_d'], n_steps=spec['n_steps'],
        gamma=1.5, lambda_sparse=spec['lambda_sparse'], cat_idxs=list(range(len(CATEGORICAL))),
        cat_dims=encoder.cat_dims, cat_emb_dim=2,
        grouped_features=[[12,15],[13,16],[14,17]],
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=.01, weight_decay=spec['weight_decay']),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=5, gamma=.8),
        seed=seed, device_name='cuda' if torch.cuda.is_available() else 'cpu', verbose=1)
    model.fit(encoder.transform(X), np.asarray(y,dtype=np.int64),
        eval_set=[(encoder.transform(stop_X), np.asarray(stop_y,dtype=np.int64))],
        eval_name=['stopping'], eval_metric=[AveragePrecision], max_epochs=epochs, patience=8,
        batch_size=min(batch_size,len(X)), virtual_batch_size=256, num_workers=0, drop_last=False)
    return TabNetPredictor(encoder,model)


def run(args):
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    raw,digest=(synthetic(2000,args.seed),None) if args.smoke else load_data(args.data)
    X,y,_=clean(raw)
    if args.smoke:
        split=partitions(y,42)
    else:
        source=Path(args.partitions_from)
        previous=json.loads((source/'development.json').read_text())
        if previous['dataset_sha256']!=digest:
            raise ValueError('Dataset hash differs from prior split source')
        with np.load(source/'splits.npz') as saved:
            split={k:saved[k] for k in saved.files}
    forbidden=set(split['test'])
    for key in ['fit','stop','select','calibrate']:
        if forbidden.intersection(split[key]):
            raise ValueError('Development split intersects inspected test')
    np.savez_compressed(output/'splits.npz',**split)
    fit,stop,select,cal=[split[k] for k in ['fit','stop','select','calibrate']]
    ledger=dict(status='SYNTHETIC_SMOKE_ONLY' if args.smoke else 'EXPLORATORY_TABNET_DEVELOPMENT',
       dataset_sha256=digest,seed=args.seed,torch=torch.__version__,pytorch_tabnet=version('pytorch-tabnet'),
       device='cuda' if torch.cuda.is_available() else 'cpu',epochs=args.epochs,
       source_run=args.partitions_from,split_counts={k:len(v) for k,v in split.items()},
       final_test_accessed=False,selection_rule='maximum average precision; tie minimum Brier',
       candidates=[],limitations=['Development partition reused from earlier search; exploratory findings',
       'No fresh confirmatory evaluation','No claim of TabNet novelty or automatic faithfulness of attention masks'])
    best,best_key=None,(-np.inf,-np.inf)
    for spec in configurations():
        started=time.perf_counter()
        model=fit_tabnet(spec,X.iloc[fit],y.iloc[fit],X.iloc[stop],y.iloc[stop],
                         args.epochs,args.seed,args.batch_size)
        score=metrics(y.iloc[select],model.predict_proba(X.iloc[select])[:,1])
        row=dict(configuration=spec,seconds=time.perf_counter()-started,
                 best_epoch=int(model.model.best_epoch),stopping_best_ap=float(model.model.best_cost),
                 selection=score,history={k:[float(v) for v in values] for k,values in model.model.history.history.items()})
        ledger['candidates'].append(row)
        key=(score['average_precision'],-score['brier'])
        if key>best_key:
            best,best_key=model,key; ledger['winner']=spec['name']
        (output/'development.json').write_text(json.dumps(ledger,indent=2,allow_nan=False))
        print(spec['name'], 'selection AP=',score['average_precision'],flush=True)
    best.calibrator=LogisticRegression(C=1,max_iter=1000,random_state=args.seed).fit(
        probability_logit(best.predict_proba(X.iloc[cal])[:,1]),y.iloc[cal])
    ledger['calibrated_selection']=metrics(y.iloc[select],best.predict_proba(X.iloc[select])[:,1])
    temporary=output/'winner.tmp.joblib'
    joblib.dump(best,temporary)
    joblib.load(temporary)
    temporary.replace(output/'winner.joblib')
    # Independent reload check on development inputs, without touching final test.
    loaded=joblib.load(output/'winner.joblib')
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[select[:16]]),best.predict_proba(X.iloc[select[:16]]))
    ledger['serialization_roundtrip_verified']=True
    (output/'development.json').write_text(json.dumps(ledger,indent=2,allow_nan=False))
    print('Selected',ledger['winner'],'; final test NOT accessed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data');p.add_argument('--smoke',action='store_true')
    p.add_argument('--partitions-from',default='results/cdc2024-development')
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=4096)
    a=p.parse_args()
    if not a.smoke and not a.data:p.error('--data or --smoke required')
    if a.epochs<1 or a.threads<1 or a.batch_size<2:p.error('Invalid epochs, threads or batch size')
    torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):run(a)

if __name__=='__main__':
    from truststroke.tabnet_search import main as imported_main
    imported_main()
