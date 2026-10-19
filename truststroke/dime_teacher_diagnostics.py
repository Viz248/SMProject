"""Post-hoc development diagnostic of teacher/deployment utility alignment; no fitting."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from threadpoolctl import threadpool_limits
from .baseline import clean
from .optimize import load_data
from .dime_pilot import Value,network,masks,targets
from .dime_matched import EncodedPredictor


def restore_predictor(root):
    model=EncodedPredictor.__new__(EncodedPredictor)
    model.prep=joblib.load(root/'preprocessor.joblib')
    saved=torch.load(root/'selected_state.pt',weights_only=True)
    model.groups=saved['groups'];model.d=int(model.groups.max())+1
    width=saved['predictor']['0.weight'].shape[1]
    model.net=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    model.net.load_state_dict(saved['predictor']);model.net.eval()
    return model


def restore_value(path,predictor):
    saved=torch.load(path,weights_only=True);v=Value.__new__(Value)
    v.predictor=predictor;v.scale=saved['scale'];v.net=network(predictor.net[0].in_features,predictor.d)
    v.net.load_state_dict(saved['state_dict']);v.net.eval();return v


def run(a):
    root=Path(a.run);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((root/'protocol.json').read_text())
    raw,digest=load_data(a.data)
    if digest!=protocol['brfss_sha256']:raise ValueError('Data hash mismatch')
    X,y,_=clean(raw);y=np.asarray(y)
    bc=load_breast_cancer(as_frame=True)
    datasets={'brfss2024':(X,y,[0,1]),'wisconsin':(bc.data,1-np.asarray(bc.target),[])}
    result=dict(status='POST_HOC_DEVELOPMENT_DIAGNOSTIC_NO_FITTING',final_test_accessed=False,brfss2025_accessed=False,datasets={})
    for name,(X,y,initial) in datasets.items():
        with np.load(root/f'{name}_positions.npz') as f:ix=f['evaluation']
        E=X.iloc[ix].reset_index(drop=True);ye=y[ix]
        m=masks(len(E),E.shape[1],initial,np.random.default_rng(9081));eligible=m==0
        rows=[]
        for seed in protocol['seeds']:
            seedroot=root/f'{name}_seed{seed}';full=restore_predictor(seedroot/'native_brier')
            delta,_=targets(full,E,ye,m,'brier')
            td=np.mean([targets(restore_predictor(seedroot/f'teacher{k}'),E,ye,m,'brier')[0] for k in range(3)],axis=0)
            v=restore_value(seedroot/'cf_brier.pt',full).predict(E,m)
            ins=restore_value(seedroot/'insample_brier.pt',full).predict(E,m)
            mse=lambda p,t:float(np.mean((p[eligible]-t[eligible])**2))
            # Center candidates within each record to remove common pre-loss terms.
            def centered(t):
                mean=(t*(1-m)).sum(1)/(1-m).sum(1)
                return t-mean[:,None]
            row=dict(seed=seed,teacher_mean_delta_minus_full=float(np.mean((td-delta)[eligible])),
                teacher_full_delta_rmse=float(np.sqrt(mse(td,delta))),
                cf_mse_full=mse(v,delta),cf_mse_teacher_mean=mse(v,td),
                insample_mse_full=mse(ins,delta),insample_mse_teacher_mean=mse(ins,td),
                cf_centered_mse_full=mse(centered(v),centered(delta)),
                insample_centered_mse_full=mse(centered(ins),centered(delta)),
                teacher_centered_drift_rmse=float(np.sqrt(mse(centered(td),centered(delta)))))
            rows.append(row)
        result['datasets'][name]=rows
    out.write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(p.parse_args())
