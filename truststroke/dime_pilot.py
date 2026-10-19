"""Development-only fixed-predictor DIME-style target-estimation pilot.

Not an exact reproduction of DIME's joint optimization or published scores.
DIME: Gadgil, Covert, Lee, ICLR 2024, Eq. 3; Apache-2.0 reference
https://github.com/suinleelab/DIME commit 06ba70f7bcd54dbd8ccd64bed398dfef90d09da5.
"""
import argparse, copy, json, time, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from .baseline import clean, FEATURES, CATEGORICAL
from .optimize import load_data, partitions


def state(z, mask, groups):
    """Only observed feature values and mask enter either network."""
    return torch.cat((z * mask[:, groups], mask), dim=1)


def masks(n,d,initial,rng):
    m=np.zeros((n,d),np.float32);m[:,initial]=1
    candidates=np.array([j for j in range(d) if j not in initial])
    for row in m:
        k=int(rng.integers(0,min(5,len(candidates))+1))
        row[rng.choice(candidates,k,replace=False)]=1
    return m


def network(inp,out):
    return nn.Sequential(nn.Linear(inp,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,out))


class Predictor:
    def __init__(self, X, y, categorical, seed, epochs):
        torch.manual_seed(seed);self.d=X.shape[1]
        blocks=[];groups=[]
        for j,c in enumerate(X.columns):
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False) if c in categorical else StandardScaler()
            blocks.append((str(j),enc,[c]))
        self.prep=ColumnTransformer(blocks)
        z=self.prep.fit_transform(X).astype(np.float32)
        for j,(_,enc,cols) in enumerate(self.prep.transformers_):
            groups.extend([j]*enc.transform(X[cols].iloc[:1]).shape[1])
        self.groups=torch.tensor(groups,dtype=torch.long)
        self.net=network(z.shape[1]+self.d,1)
        opt=torch.optim.AdamW(self.net.parameters(),lr=.002,weight_decay=.001)
        z=torch.tensor(z);target=torch.tensor(np.asarray(y),dtype=torch.float32)
        rng=np.random.default_rng(seed)
        self.net.train()
        for _ in range(epochs):
            order=rng.permutation(len(y))
            for a in range(0,len(y),256):
                ix=order[a:a+256]
                m=torch.tensor((rng.random((len(ix),self.d))<rng.uniform(.1,.9,(len(ix),1))).astype(np.float32))
                logits=self.net(state(z[ix],m,self.groups)).squeeze(1)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,target[ix])
                opt.zero_grad();loss.backward();opt.step()
        self.net.eval()

    def predict(self,X,m):
        z=torch.tensor(self.prep.transform(X).astype(np.float32));m=torch.tensor(m,dtype=torch.float32)
        with torch.no_grad():return self.net(state(z,m,self.groups)).squeeze(1).sigmoid().numpy()


def losses(y,p,kind):
    if kind=='brier':return (y-p)**2
    p=np.clip(p,1e-6,1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def targets(model,X,y,m,kind):
    before=losses(y,model.predict(X,m),kind);post=np.zeros_like(m)
    for j in range(m.shape[1]):
        after=m.copy();after[:,j]=1
        post[:,j]=losses(y,model.predict(X,after),kind)
    return before[:,None]-post,post


class Value:
    def __init__(self,predictor,X,m,t,seed,epochs):
        torch.manual_seed(seed);self.predictor=predictor
        z=torch.tensor(predictor.prep.transform(X).astype(np.float32));mask=torch.tensor(m)
        self.net=network(z.shape[1]+m.shape[1],m.shape[1])
        opt=torch.optim.AdamW(self.net.parameters(),lr=.002,weight_decay=.001)
        # Constant scaling improves optimization without changing action rankings.
        self.scale=max(float(np.std(t[m==0])),1e-4)
        target=torch.tensor(t/self.scale,dtype=torch.float32);available=1-mask
        rng=np.random.default_rng(seed)
        for _ in range(epochs):
            order=rng.permutation(len(X))
            for a in range(0,len(X),256):
                ix=order[a:a+256];out=self.net(state(z[ix],mask[ix],predictor.groups))
                loss=(((out-target[ix])**2)*available[ix]).sum()/available[ix].sum().clamp_min(1)
                opt.zero_grad();loss.backward();opt.step()
        self.net.eval()

    def predict(self,X,m):
        z=torch.tensor(self.predictor.prep.transform(X).astype(np.float32))
        with torch.no_grad():return self.net(state(z,torch.tensor(m),self.predictor.groups)).numpy()*self.scale


def paired_interval(diff,seed):
    rng=np.random.default_rng(seed)
    vals=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(500)]
    return [float(v) for v in np.quantile(vals,[.025,.975])]


def evaluate(model,policy,X,y,initial,seed,static=None,post=False):
    m=np.zeros((len(X),X.shape[1]),np.float32);m[:,initial]=1
    rng=np.random.default_rng(seed);curve=[];actions=[];p_all=[]
    for b in range(4):
        p=model.predict(X,m);p_all.append(p)
        curve.append(dict(budget=b,brier=float(brier_score_loss(y,p)),average_precision=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p))))
        if b==3:break
        if policy=='random':v=rng.random(m.shape)
        elif policy=='static':v=np.broadcast_to(static,m.shape).copy()
        else:v=policy.predict(X,m);v=-v if post else v
        v[m==1]=-np.inf;j=v.argmax(1)
        assert np.all(m[np.arange(len(m)),j]==0)
        actions.append(j);m[np.arange(len(m)),j]=1
    return curve,p_all,actions


def run_dataset(name,X,y,categorical,fit_ids,eval_ids,initial,out,a):
    result={'dataset':name,'feature_names':list(X.columns),'initial_features':[X.columns[j] for j in initial],
            'records_fit':len(fit_ids),'records_development_evaluation':len(eval_ids),'positive_fit':int(y[fit_ids].sum()),'positive_evaluation':int(y[eval_ids].sum()),'seeds':[]}
    np.savez_compressed(out/f'{name}_positions.npz',fit=fit_ids,evaluation=eval_ids)
    for seed in a.seeds:
        start=time.time();rng=np.random.default_rng(seed)
        train=X.iloc[fit_ids].reset_index(drop=True);yt=y[fit_ids];ev=X.iloc[eval_ids].reset_index(drop=True);ye=y[eval_ids]
        full=Predictor(train,yt,categorical,seed,a.epochs)
        # Multiple states per respondent. Every fold excludes all copies of its held-out rows.
        owner=np.tile(np.arange(len(train)),3);episode=train.iloc[owner].reset_index(drop=True);ey=yt[owner]
        m=masks(len(owner),X.shape[1],initial,rng)
        inside,insidepost=targets(full,episode,ey,m,'brier');ce,_=targets(full,episode,ey,m,'ce')
        cf=np.zeros_like(inside);cfpost=np.zeros_like(inside);matched=np.zeros_like(inside)
        folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(train,yt))
        teachers=[]
        for k,(tr,held) in enumerate(folds):
            teacher=Predictor(train.iloc[tr],yt[tr],categorical,seed+k+1,a.epochs);teachers.append(teacher)
            ix=np.flatnonzero(np.isin(owner,held));cf[ix],cfpost[ix]=targets(teacher,episode.iloc[ix],ey[ix],m[ix],'brier')
        # Each row gets a same-size teacher that DID train on it; contrast to CF
        # separates respondent reuse from teacher sample size (not joint DIME training).
        for k,(_,held) in enumerate(folds):
            ix=np.flatnonzero(np.isin(owner,held));matched[ix],_=targets(teachers[(k+1)%3],episode.iloc[ix],ey[ix],m[ix],'brier')
        # Split-sample control: teacher's fold held out for the value learner.
        tr,held=folds[0];sx=np.flatnonzero(np.isin(owner,held))
        policy_targets={'insample_brier':inside,'crossfit_brier':cf,'matched_size_insample_brier':matched,'crossfit_postloss':cfpost,'insample_ce':ce}
        policies={k:Value(full,episode,m,t,seed,a.value_epochs) for k,t in policy_targets.items()}
        policies['split_brier']=Value(full,episode.iloc[sx],m[sx],cf[sx],seed,a.value_epochs)
        rows={};predictions={};acts={};calibration={}
        for key,policy in list(policies.items())+[('random','random'),('static','static')]:
            rows[key],predictions[key],acts[key]=evaluate(full,policy,ev,ye,initial,seed,static=np.mean(inside,axis=0),post=key=='crossfit_postloss')
        em=masks(len(ev),X.shape[1],initial,np.random.default_rng(12345))
        realized,_=targets(full,ev,ye,em,'brier')
        for key in ['insample_brier','crossfit_brier','matched_size_insample_brier','split_brier']:
            estimates=policies[key].predict(ev,em);v=estimates[em==0];r=realized[em==0]
            bins=[]
            for ix in np.array_split(np.argsort(v),5):
                bins.append(dict(n=len(ix),predicted_mean=float(v[ix].mean()),realized_mean=float(r[ix].mean())))
            calibration[key]=dict(mse=float(np.mean((v-r)**2)),mean_bias=float(np.mean(v-r)),bins=bins)
        diff=(ye-predictions['insample_brier'][3])**2-(ye-predictions['crossfit_brier'][3])**2
        pair=dict(benefit_cf_over_insample=float(diff.mean()),iid_record_bootstrap_ci=paired_interval(diff,seed))
        row=dict(seed=seed,curves=rows,value_calibration=calibration,primary=pair,
             action_agreement_cf_delta_post=float(np.mean(np.array(acts['crossfit_brier'])==np.array(acts['crossfit_postloss']))),seconds=time.time()-start)
        result['seeds'].append(row)
        np.savez_compressed(out/f'{name}_seed{seed}_predictions.npz',labels=ye,**{k:np.array(v) for k,v in predictions.items()})
        (out/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(name,seed,pair,'seconds',round(row['seconds'],1),flush=True)
    return result


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='EXPLORATORY_DEVELOPMENT_PILOT',official_DIME_reproduction=False,
        base='DIME ICLR 2024 Eq.3',reference_commit='06ba70f7bcd54dbd8ccd64bed398dfef90d09da5',
        primary='Brier benefit of crossfit_brier versus insample_brier after exactly 3 acquisitions',
        implementation='Fixed CE-trained masked MLP, separate unconstrained value MLP, three-fold target estimation; no joint DIME optimization, no external calibration',
        folds=3,episodes_per_record=3,seeds=a.seeds,epochs=a.epochs,value_epochs=a.value_epochs,
        final_test_accessed=False,brfss2025_accessed=False,
        limitations=['Complete-case BRFSS sample; no survey-design inference','Historical development cohort already explored','Small bounded pilot; no hyperparameter search','Fold teachers trained on 2/3 while deployment predictor uses full fit; matched-size and split controls included','Split control has fewer distinct value-training records','Record bootstrap does not incorporate training uncertainty; seeds reported separately','Not an official DIME, GDFS, ACO or L2M benchmark reproduction'],
        python=platform.python_version(),torch=torch.__version__)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    bc=load_breast_cancer(as_frame=True);X=bc.data;y=np.asarray(bc.target)
    fit,ev=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=2026)
    # sklearn labels 1=benign; invert so positive class means malignancy.
    run_dataset('wisconsin_breast_cancer',X,1-y,[],fit,ev,[],out,a)
    raw,digest=load_data(a.data);X,y,_=clean(raw);y=np.asarray(y)
    split=partitions(y,42);complete=~X.isna().any(axis=1).to_numpy()
    rng=np.random.default_rng(2026)
    fit=rng.choice(split['fit'][complete[split['fit']]],a.fit_limit,replace=False)
    ev=rng.choice(split['select'][complete[split['select']]],a.eval_limit,replace=False)
    assert not set(fit)&set(ev);assert not (set(fit)|set(ev))&set(split['test'])
    protocol['dataset_sha256']=digest
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run_dataset('brfss2024',X,y,CATEGORICAL,fit,ev,[FEATURES.index(c) for c in ['SEXVAR','_AGEG5YR']],out,a)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--value-epochs',type=int,default=25);p.add_argument('--fit-limit',type=int,default=6000);p.add_argument('--eval-limit',type=int,default=1500)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(p.parse_args())
