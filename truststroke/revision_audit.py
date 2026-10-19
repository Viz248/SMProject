"""Reconstruct post-review analyses; keep original frozen artifacts unchanged."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from .availability_study import ensemble_bundle,train_model,selected_positions,linearized_mean
from .baseline import clean
from .optimize import probability_logit,partitions,sha256
ROOT=Path(__file__).resolve().parents[1]

def calibration_diagnostics(y,p,w):
 y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float);w=w/w.sum();x=logit(np.clip(p,1e-6,1-1e-6))
 citl=brentq(lambda a:np.sum(w*(expit(a+x)-y)),-40,40)
 def fun(b):
  eta=b[0]+b[1]*x;q=expit(eta)
  return np.sum(w*(np.logaddexp(0,eta)-y*eta)),np.array([np.sum(w*(q-y)),np.sum(w*(q-y)*x)])
 fit=minimize(fun,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-10})
 if np.max(np.abs(fun(fit.x)[1]))>1e-7:raise RuntimeError('Diagnostic convergence')
 u,inv=np.unique(p,return_inverse=True);mass=np.bincount(inv,weights=w);bins=np.minimum(9,np.floor(10*(np.cumsum(mass)-mass/2)).astype(int))[inv];curve=[]
 for b in np.unique(bins):
  m=bins==b;v=w[m];sw=v.sum();curve.append(dict(bin=int(b),count=int(m.sum()),weight_fraction=float(sw),probability=float(np.sum(v*p[m])/sw),observed=float(np.sum(v*y[m])/sw)))
 return dict(citl=float(citl),intercept=float(fit.x[0]),slope=float(fit.x[1]),mean_prediction=float(np.sum(w*p)),prevalence=float(np.sum(w*y)),weighted_ece=float(sum(v['weight_fraction']*abs(v['probability']-v['observed']) for v in curve)),curve=curve)

def diagnostics(a):
 run=Path(a.run);ext=Path(a.external);out=Path(a.output);cfg=json.loads((run/'protocol.json').read_text());r=json.loads((ext/'results.json').read_text());D=joblib.load(ext/'design.joblib');common=D._STATE.isin(r['common_states']).to_numpy()
 with np.load(ext/'predictions.npz') as z:y=z['labels'][common];pred={k:z[k][common] for k in z.files if k!='labels'}
 ens=ensemble_bundle(pred,cfg);w=D.loc[common,'_LLCPWT'].to_numpy();rows={k:calibration_diagnostics(y,p,w) for k,p in ens.items()}
 (out/'calibration.json').write_text(json.dumps(dict(status='POST_HOC_DIAGNOSTICS_NEVER_DEPLOYED',metrics=rows),indent=2));pd.DataFrame([dict(key=k,**{n:v for n,v in row.items() if n!='curve'}) for k,row in rows.items()]).to_csv(out/'calibration.csv',index=False)
 cols=['_STATE','_STSTR','_PSU','_LLCPWT','CVDSTRK3'];parts=[]
 for chunk in pd.read_sas(a.data,format='xport',chunksize=10000):parts.append(chunk.loc[chunk._STATE.isin(r['common_states']),cols])
 full=pd.concat(parts);domain=full.CVDSTRK3.isin([1,2]).to_numpy();valid=full.loc[domain];np.testing.assert_array_equal(valid.index,D.loc[common].index);np.testing.assert_array_equal((valid.CVDSTRK3==1).to_numpy(),y)
 k='hist_gradient_boosting|demographics_history|';diff=(y-ens[k+'mask_trained'])**2-(y-ens[k+'fixed_scenario_calibration'])**2;values=np.zeros(len(full));values[domain]=diff;stat=linearized_mean(values,full,domain);np.testing.assert_allclose(stat['estimate'],r['primary']['estimate'],atol=1e-15,rtol=0)
 def counts(d):
  n=d.groupby(['_STATE','_STSTR'])._PSU.nunique();return dict(rows=len(d),strata=len(n),psus=int(n.sum()),singleton_strata=int(n.eq(1).sum()))
 (out/'design_audit.json').write_text(json.dumps(dict(status='POST_HOC_FULL_DESIGN_DOMAIN',full=counts(full),valid_outcome=counts(valid),domain_taylor=stat,frozen_primary=r['primary'],predictions_sha256=sha256(ext/'predictions.npz'),raw2025_sha256=sha256(a.data)),indent=2))
 export=full[cols[:4]].copy();export.columns=['state','stratum','psu','weight'];export['domain']=domain.astype(int);export['numerator']=domain*values;export.to_csv(ext/'posthoc_survey_domain.csv',index=False);print('Diagnostics and design audit completed',flush=True)

def robustness(a):
 run=Path(a.run);ext=Path(a.external);art=Path(a.output);out=Path(a.robustness_output);out.mkdir(parents=True,exist_ok=False);cfg=json.loads((run/'protocol.json').read_text());post=json.loads((ROOT/'configs/posthoc_revision.json').read_text())['training_sensitivity']
 if sha256(a.data)!=cfg['input_sha256_2024_xpt'] or sha256(a.data25)!=cfg['input_sha256_2025_xpt']:raise ValueError('Data identity')
 raw=pd.read_sas(a.data,format='xport');X,y,D=clean(raw);del raw
 raw=pd.read_sas(a.data25,format='xport');E,ey,ED=clean(raw);del raw
 common=ED._STATE.isin(json.loads((ext/'results.json').read_text())['common_states']).to_numpy();E=E.loc[common];ey=ey.loc[common].to_numpy();w=ED.loc[common,'_LLCPWT'].to_numpy();fixed=selected_positions(y,cfg);pool=partitions(y,cfg['split_seed'])['fit'];scenario='demographics_history';rows=[];indices=[]
 for sample_seed in post['fitting_subsample_seeds']:
  ix,_=train_test_split(pool,train_size=60000,stratify=y.iloc[pool],random_state=sample_seed);indices.append(set(ix.tolist()));np.save(out/f'fit_positions_{sample_seed}.npy',ix);preds={'full':[],'mask':[]};seeds=[]
  for seed in cfg['seeds']:
   for mode in ['full','mask']:
    m=train_model(X.iloc[ix],y.iloc[ix],X.iloc[fixed['stop']],y.iloc[fixed['stop']],cfg['features'],list(cfg['scenarios'].values()) if mode=='mask' else [cfg['features']],post['family'],seed,cfg)
    cal=m.raw(X.iloc[fixed['calibrate']],cfg['scenarios'][scenario]);m.calibrators[scenario]=LogisticRegression(C=1,max_iter=1000,random_state=seed).fit(probability_logit(cal),y.iloc[fixed['calibrate']]);joblib.dump(m,out/f'{sample_seed}_{seed}_{mode}.joblib');preds[mode].append(m.predict(E,cfg['scenarios'][scenario],scenario));print('TRAINED',sample_seed,seed,mode,flush=True)
   d=(ey-preds['mask'][-1])**2-(ey-preds['full'][-1])**2;seeds.append(dict(seed=seed,contrast=float(np.average(d,weights=w))))
  mt=np.mean(preds['mask'],axis=0);sc=np.mean(preds['full'],axis=0);d=(ey-mt)**2-(ey-sc)**2;np.savez_compressed(out/f'predictions_{sample_seed}.npz',MT=mt,SC=sc);rows.append(dict(fitting_subsample_seed=sample_seed,MT_brier=float(np.average((ey-mt)**2,weights=w)),SC_brier=float(np.average((ey-sc)**2,weights=w)),contrast=float(np.average(d,weights=w)),seed_contrasts=seeds));record=dict(status='RECONSTRUCTED_POST_HOC_NOT_FRESH_CONFIRMATION',protocol_sha256=sha256(ROOT/'configs/posthoc_revision.json'),rows=rows,complete=len(rows)==3);(art/'fitting_sample_robustness.json').write_text(json.dumps(record,indent=2))
 record['pairwise_overlap']=[dict(a=post['fitting_subsample_seeds'][i],b=post['fitting_subsample_seeds'][j],shared=len(indices[i]&indices[j])) for i in range(3) for j in range(i+1,3)];(art/'fitting_sample_robustness.json').write_text(json.dumps(record,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnostics','robustness']);p.add_argument('--data',required=True);p.add_argument('--data25');p.add_argument('--run',default=str(ROOT/'results/availability-paper-fit-v2'));p.add_argument('--external',default=str(ROOT/'results/availability-paper-2025'));p.add_argument('--output',default=str(ROOT/'docs/experiments/revision_audit'));p.add_argument('--robustness-output',default=str(ROOT/'results/posthoc-fitting-samples'));a=p.parse_args();Path(a.output).mkdir(parents=True,exist_ok=True)
 with threadpool_limits(limits=2):diagnostics(a) if a.stage=='diagnostics' else robustness(a)
