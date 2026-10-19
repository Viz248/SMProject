"""Post-evaluation singleton-stratum sensitivity; does not replace frozen inference.

Uses the average non-singleton stratum variance contribution described by the
survey package's surveyoptions manual. This Python implementation has unit
checks but is not independently executed in R survey.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import t
from .availability_study import ensemble_bundle

def average_singleton_interval(values,design,domain=None):
    values=np.asarray(values,float);w=np.asarray(design['_LLCPWT'],float)
    domain=np.ones(len(w),bool) if domain is None else np.asarray(domain,bool)
    den=w[domain].sum();theta=float(np.sum(w[domain]*values[domain])/den)
    d=design[['_STATE','_STSTR','_PSU']].copy();d['z']=w*domain*(values-theta)/den
    psu=d.groupby(['_STATE','_STSTR','_PSU'])['z'].sum();groups=psu.groupby(level=[0,1]);contributions=[];singleton=0
    for _,v in groups:
        if len(v)==1:singleton+=1
        else:contributions.append(len(v)/(len(v)-1)*float(np.sum((v-v.mean())**2)))
    variance=float(np.sum(contributions)+singleton*np.mean(contributions));df=len(psu)-len(groups)
    se=float(np.sqrt(variance));critical=float(t.ppf(.975,df))
    return dict(estimate=theta,se=se,ci95=[theta-critical*se,theta+critical*se],singleton_strata=singleton,
      variance_non_singleton=float(np.sum(contributions)),variance_singleton_imputed=float(singleton*np.mean(contributions)),
      status='POST_HOC_SENSITIVITY_NOT_FROZEN_PRIMARY_CI',method='replace singleton contributions by average contribution of non-singleton strata',
      assumptions='singleton strata comparable in variance contribution to non-singleton strata; no FPC; conditional on models; not independently cross-checked in R')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    cfg=json.loads((Path(a.run)/'protocol.json').read_text());r=json.loads((Path(a.external)/'results.json').read_text())
    D=joblib.load(Path(a.external)/'design.joblib');common=np.isin(D['_STATE'],r['common_states']);D=D.loc[common].reset_index(drop=True)
    with np.load(Path(a.external)/'predictions.npz') as z:
        y=z['labels'][common];pred={k:z[k][common] for k in z.files if k!='labels'}
    ens=ensemble_bundle(pred,cfg);diff=(y-ens['hist_gradient_boosting|demographics_history|mask_trained'])**2-(y-ens['hist_gradient_boosting|demographics_history|fixed_scenario_calibration'])**2
    stat=average_singleton_interval(diff,D);assert np.isclose(stat['estimate'],r['primary']['estimate'],atol=1e-12)
    result=dict(primary_frozen_interval=r['primary']['ci95'],sensitivity=stat,
        reference='https://r-survey.r-forge.r-project.org/survey/html/surveyoptions.html',reference_checked='2026-10-19',models_or_predictions_changed=False)
    out.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
