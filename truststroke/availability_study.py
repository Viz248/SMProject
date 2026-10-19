"""Frozen availability-shift experiment. See configs/availability_study_frozen.json."""
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
from .baseline import clean,CATEGORICAL,metrics
from .optimize import load_data,partitions,probability_logit,sha256
from .survey import weighted_points


def linearized_mean(values,design,domain=None):
    """With-replacement stratified PSU Taylor interval for a weighted ratio mean.

    Full cohort PSUs retained when estimating a domain. No FPC; conditional on
    fitted predictions. Singleton strata make the interval unavailable.
    """
    values=np.asarray(values,dtype=float);w=design['_LLCPWT'].to_numpy(float)
    domain=np.ones(len(w),bool) if domain is None else np.asarray(domain,bool)
    if len(values)!=len(w) or not np.isfinite(values).all() or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid aligned survey values/weights')
    den=w[domain].sum()
    if den<=0:raise ValueError('Empty weighted domain')
    mean=float(np.sum(w[domain]*values[domain])/den)
    frame=design[['_STATE','_STSTR','_PSU']].copy()
    if frame.isna().any().any():raise ValueError('Missing survey design')
    frame['influence']=w*domain*(values-mean)/den
    psu=frame.groupby(['_STATE','_STSTR','_PSU'],observed=True)['influence'].sum()
    group=psu.groupby(level=[0,1]);sizes=group.size();singletons=int((sizes<2).sum());variance=0.
    for _,g in group:
        m=len(g)
        if m>1:variance+=m/(m-1)*float(np.sum((g-g.mean())**2))
    df=int(len(psu)-len(sizes));se=float(np.sqrt(variance))
    ci=None if singletons or df<=0 else [mean-float(student_t.ppf(.975,df))*se,mean+float(student_t.ppf(.975,df))*se]
    return dict(estimate=mean,standard_error=se if ci else None,ci95=ci,psus=len(psu),strata=len(sizes),singleton_strata=singletons,degrees_freedom=df,
        method='stratified-PSU Taylor ratio mean; with-replacement; no FPC; conditional on fitted models')


class AvailabilityModel:
    def __init__(self,columns):self.columns=list(columns);self.calibrators={}
    def fit_preprocessor(self,X):
        transformers=[];groups=[]
        for j,c in enumerate(self.columns):
            enc=make_pipeline(SimpleImputer(strategy='constant',fill_value=-1,keep_empty_features=True),OneHotEncoder(handle_unknown='ignore',sparse_output=False)) if c in CATEGORICAL else make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler())
            transformers.append((str(j),enc,[c]))
        self.prep=ColumnTransformer(transformers);z=np.asarray(self.prep.fit_transform(X),np.float32)
        for j,(_,enc,cols) in enumerate(self.prep.transformers_):
            if isinstance(enc,str):continue  # ColumnTransformer's dropped remainder.
            groups.extend([j]*enc.transform(X[cols].iloc[:1]).shape[1])
        self.groups=np.asarray(groups,int);return z
    def encoded(self,X,observed,z=None):
        z=np.asarray(self.prep.transform(X),np.float32) if z is None else z
        m=np.array([float(c in observed) for c in self.columns],np.float32)
        return np.concatenate((z*m[self.groups],np.broadcast_to(m,(len(z),len(m)))),axis=1).astype(np.float32)
    def raw(self,X,observed):
        # Bound transformation memory on full-year cohorts.
        output=[]
        for start in range(0,len(X),20000):
            z=self.encoded(X.iloc[start:start+20000],observed)
            if self.family=='mlp':
                self.estimator.eval()
                with torch.no_grad():p=self.estimator(torch.tensor(z)).squeeze(1).sigmoid().numpy()
            else:p=self.estimator.predict_proba(z)[:,1]
            output.append(p)
        return np.concatenate(output)
    def calibrate(self,p,key):return self.calibrators[key].predict_proba(probability_logit(p))[:,1]
    def predict(self,X,observed,key):return self.calibrate(self.raw(X,observed),key)


def train_model(X,y,S,sy,columns,patterns,family,seed,cfg):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=AvailabilityModel(columns);model.family=family
    z=model.fit_preprocessor(X);s=np.asarray(model.prep.transform(S),np.float32)
    A=np.concatenate([model.encoded(X,p,z) for p in patterns]);B=np.concatenate([model.encoded(S,p,s) for p in patterns])
    target=np.tile(np.asarray(y),len(patterns));starget=np.tile(np.asarray(sy),len(patterns))
    model.history=[]
    if family=='hist_gradient_boosting':
        model.estimator=HistGradientBoostingClassifier(**cfg['hgb'],scoring='loss',random_state=seed)
        model.estimator.fit(A,target,X_val=B,y_val=starget)
        model.history=[dict(iteration=i,training_score=float(tr),stopping_score=float(va)) for i,(tr,va) in enumerate(zip(model.estimator.train_score_,model.estimator.validation_score_))]
    else:
        h=cfg['mlp'];model.estimator=nn.Sequential(nn.Linear(A.shape[1],h['hidden'][0]),nn.ReLU(),nn.Dropout(h['dropout']),nn.Linear(h['hidden'][0],h['hidden'][1]),nn.ReLU(),nn.Linear(h['hidden'][1],1))
        opt=torch.optim.AdamW(model.estimator.parameters(),lr=h['learning_rate'],weight_decay=h['weight_decay'])
        A=torch.tensor(A);target=torch.tensor(target,dtype=torch.float32);B=torch.tensor(B);starget=torch.tensor(starget,dtype=torch.float32)
        best=np.inf;selected=None
        for epoch in range(h['epochs']):
            model.estimator.train();order=rng.permutation(len(A));total=0.
            for start in range(0,len(A),h['batch_size']):
                ix=order[start:start+h['batch_size']];logits=model.estimator(A[ix]).squeeze(1)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,target[ix]);opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(ix)
            model.estimator.eval()
            with torch.no_grad():val=float(nn.functional.binary_cross_entropy_with_logits(model.estimator(B).squeeze(1),starget))
            model.history.append(dict(epoch=epoch+1,training_nll=total/len(A),stopping_nll=val))
            if val<best:best=val;selected=copy.deepcopy(model.estimator.state_dict());model.selected_epoch=epoch+1
        model.estimator.load_state_dict(selected);model.estimator.eval()
    return model


def selected_positions(y,cfg):
    s=partitions(y,cfg['split_seed'])
    for key,limit in [('fit',cfg['fit_limit']),('stop',cfg['stopping_limit']),('select',cfg['development_evaluation_limit'])]:
        s[key],_=train_test_split(s[key],train_size=limit,stratify=y.iloc[s[key]],random_state=cfg['subsample_seed'])
    return s


def prediction_bundle(root,cfg,X):
    """All seed/condition probabilities; no refitting or model selection."""
    output={}
    for family in cfg['families']:
        for seed in cfg['seeds']:
            full=joblib.load(root/f'{family}_{seed}_full.joblib');partial=joblib.load(root/f'{family}_{seed}_mask.joblib')
            for scenario,observed in cfg['scenarios'].items():
                raw=full.raw(X,observed)
                output[f'{family}|{seed}|{scenario}|fixed_global_calibration']=full.calibrate(raw,'full').astype(np.float32)
                output[f'{family}|{seed}|{scenario}|fixed_scenario_calibration']=full.calibrate(raw,scenario).astype(np.float32)
                output[f'{family}|{seed}|{scenario}|mask_trained']=partial.predict(X,observed,scenario).astype(np.float32)
                subset=full if scenario=='full' else joblib.load(root/f'{family}_{seed}_subset_{scenario}.joblib')
                output[f'{family}|{seed}|{scenario}|subset_retrained']=subset.predict(X,observed,scenario).astype(np.float32)
    return output


def ensemble_bundle(predictions,cfg):
    return {f'{family}|{scenario}|{condition}':np.mean([predictions[f'{family}|{seed}|{scenario}|{condition}'] for seed in cfg['seeds']],axis=0,dtype=np.float64) for family in cfg['families'] for scenario in cfg['scenarios'] for condition in cfg['conditions']}


def score_bundle(predictions,y,design,cfg):
    ensembles=ensemble_bundle(predictions,cfg);rows={};w=design['_LLCPWT'].to_numpy(float)
    for key,p in ensembles.items():
        row=metrics(y,p,.14);row['nll']=float(log_loss(y,p,labels=[0,1]));row['weighted']=weighted_points(y,p,w,.14)
        row['weighted']['nll']=float(log_loss(y,p,sample_weight=w,labels=[0,1]));rows[key]=row
    return rows


def paired_primary(predictions,y,design,cfg,domain=None):
    ensemble=ensemble_bundle(predictions,cfg);family=cfg['primary']['family'];scenario=cfg['primary']['scenario']
    a=ensemble[f'{family}|{scenario}|mask_trained'];b=ensemble[f'{family}|{scenario}|fixed_scenario_calibration']
    diff=(y-a)**2-(y-b)**2
    return linearized_mean(diff,design,domain)


def fit(a):
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    raw,digest=load_data(a.data)
    if digest!=cfg['input_sha256_2024_xpt']:raise ValueError('2024 XPT hash mismatch')
    X,y,design=clean(raw);s=selected_positions(y,cfg);np.savez_compressed(out/'positions.npz',**s)
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2));start=time.time();timings=[]
    for family in cfg['families']:
        for seed in cfg['seeds']:
            for mode in ['full','mask']+[f'subset_{c}' for c in cfg['scenarios'] if c!='full']:
                t=time.time();scenario=mode.removeprefix('subset_')
                columns=cfg['features'] if mode in ['full','mask'] else cfg['scenarios'][scenario]
                patterns=list(cfg['scenarios'].values()) if mode=='mask' else [columns]
                model=train_model(X.iloc[s['fit']],y.iloc[s['fit']],X.iloc[s['stop']],y.iloc[s['stop']],columns,patterns,family,seed,cfg)
                keys=list(cfg['scenarios']) if mode in ['full','mask'] else [scenario]
                for key in keys:
                    p=model.raw(X.iloc[s['calibrate']],cfg['scenarios'][key])
                    model.calibrators[key]=LogisticRegression(C=1,max_iter=1000,random_state=seed).fit(probability_logit(p),y.iloc[s['calibrate']])
                path=out/f'{family}_{seed}_{mode}.joblib';joblib.dump(model,path)
                check=joblib.load(path);expected=model.predict(X.iloc[s['stop'][:100]],columns,keys[0]);actual=check.predict(X.iloc[s['stop'][:100]],columns,keys[0]);np.testing.assert_allclose(expected,actual)
                row=dict(family=family,seed=seed,mode=mode,seconds=time.time()-t,history=model.history,selected_epoch=getattr(model,'selected_epoch',None))
                timings.append(row);(out/'training.json').write_text(json.dumps(timings,indent=2))
                print('TRAINED',family,seed,mode,round(row['seconds'],1),flush=True)
    # Freeze exact model files before any 2025 predictive evaluation.
    manifest=dict(protocol_sha256=sha256(a.config),source_sha256=sha256(__file__),models={p.name:sha256(p) for p in out.glob('*.joblib')},
        dataset_2024_sha256=digest,elapsed_seconds=time.time()-start,models_frozen_before_2025_evaluation=True,training_reporting_states=sorted(design['_STATE'].unique().tolist()))
    (out/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Development scoring has no effect on already frozen models.
    ix=s['select'];pred=prediction_bundle(out,cfg,X.iloc[ix]);np.savez_compressed(out/'development_predictions.npz',labels=np.asarray(y.iloc[ix]),**pred)
    result=dict(status='DEVELOPMENT_ONLY_FROZEN_MODELS',metrics=score_bundle(pred,np.asarray(y.iloc[ix]),design.iloc[ix],cfg),n=len(ix),positives=int(y.iloc[ix].sum()))
    (out/'development.json').write_text(json.dumps(result,indent=2))


def evaluate(a):
    root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'model_manifest.json').read_text())
    if sha256(__file__)!=manifest['source_sha256']:raise ValueError('Source changed since freeze')
    for name,digest in manifest['models'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen model changed')
    if not manifest['models_frozen_before_2025_evaluation']:raise ValueError('No model freeze')
    (out/'evaluation_started.json').write_text(json.dumps(dict(status='STARTED_NO_RETUNING',model_manifest_sha256=sha256(root/'model_manifest.json')),indent=2))
    raw,digest=load_data(a.data)
    if digest!=cfg['input_sha256_2025_xpt']:raise ValueError('2025 XPT hash mismatch')
    X,y,design=clean(raw);y=np.asarray(y);p=prediction_bundle(root,cfg,X);np.savez_compressed(out/'predictions.npz',labels=y,**p)
    joblib.dump(design,out/'design.joblib')
    common=np.isin(design['_STATE'],manifest['training_reporting_states'])
    rows_all=score_bundle(p,y,design,cfg);pc={k:v[common] for k,v in p.items()};yc=y[common];dc=design.loc[common].reset_index(drop=True)
    rows_common=score_bundle(pc,yc,dc,cfg);primary=paired_primary(pc,yc,dc,cfg)
    ensembles=ensemble_bundle(pc,cfg);subgroups=[];XC=X.loc[common].reset_index(drop=True)
    domains={'sex_male':XC.SEXVAR==1,'sex_female':XC.SEXVAR==2,'age18_44':XC._AGEG5YR.between(1,5),'age45_64':XC._AGEG5YR.between(6,9),'age65plus':XC._AGEG5YR.between(10,13),'income_under35k':XC.INCOME3.between(1,5),'income35kplus':XC.INCOME3.between(6,11)}
    for name,domain in domains.items():
        d=np.asarray(domain);n=int(d.sum());pos=int(yc[d].sum())
        if n<200 or pos<20:continue
        record=dict(group=name,n=n,positives=pos,primary=paired_primary(pc,yc,dc,cfg,d),metrics={})
        for condition in cfg['conditions']:
            key=f"hist_gradient_boosting|demographics_history|{condition}";pv=ensembles[key]
            record['metrics'][condition]=weighted_points(yc[d],pv[d],dc.loc[d,'_LLCPWT'],.14)
        subgroups.append(record)
    seed_effects=[]
    for seed in cfg['seeds']:
        pre=f'hist_gradient_boosting|{seed}|demographics_history|'
        diff=(yc-pc[pre+'mask_trained'])**2-(yc-pc[pre+'fixed_scenario_calibration'])**2
        seed_effects.append(dict(seed=seed,contrast=linearized_mean(diff,dc)))
    result=dict(status='FROZEN_2025_TRANSPORT_COMPLETED',raw_records=len(raw),labeled_records=len(y),positives=int(y.sum()),excluded_outcomes=len(raw)-len(y),
       all_reporting_states=sorted(design['_STATE'].unique().tolist()),common_states=sorted(dc['_STATE'].unique().tolist()),common_records=len(yc),common_positives=int(yc.sum()),
       primary=primary,seed_effects=seed_effects,all_metrics=rows_all,common_metrics=rows_common,subgroups=subgroups,
       input_sha256=digest,model_manifest_sha256=sha256(root/'model_manifest.json'),no_external_refitting=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('PRIMARY',json.dumps(primary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    f=sub.add_parser('fit');f.add_argument('--config',required=True);f.add_argument('--data',required=True);f.add_argument('--output',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--run',required=True);e.add_argument('--data',required=True);e.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        fit(args) if args.stage=='fit' else evaluate(args)
