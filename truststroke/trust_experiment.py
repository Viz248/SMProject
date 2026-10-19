"""Development-only experiments for provisional DSI/CAS/RAI and acquisition.

No final test, external/clinical validation, or novelty claims. Uses the same
held-out development records across retraining/bootstrap members and interventions.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits
from .baseline import clean, metrics, FEATURES, CATEGORICAL, NUMERIC
from .optimize import load_data, fit_candidate, probability_logit
from .trust import (confidence_attribution,decision_stability,permutation_shapley,
                    entropy,resource_adaptability,mean_interval)
from .acquisition import ConditionalDonors


def mean_model(models,frame):
    return np.mean([m.predict_proba(frame)[:,1] for m in models],axis=0)


def stability_summary(first,second):
    rows=[[decision_stability(a,b) for a,b in zip(first[i],second[i])]
          for i in range(len(first))]
    # Paired-model scores are averaged WITHIN each record before respondent bootstrap.
    per_record={}
    for key in ['dsi','rank_spearman','top_k_jaccard']:
        values=[]
        for j in range(first[0].shape[0]):
            valid=[row[j][key] for row in rows if row[j][key] is not None]
            values.append(np.mean(valid) if valid else np.nan)
        per_record[key]=np.asarray(values)
    return {key:mean_interval(value) for key,value in per_record.items()},per_record


def safe_correlation(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    valid=np.isfinite(a)&np.isfinite(b)
    if valid.sum()<3 or np.ptp(a[valid])==0 or np.ptp(b[valid])==0:return None
    return float(spearmanr(a[valid],b[valid]).statistic)


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    raw,digest=load_data(args.data);X,y,design=clean(raw)
    source=Path(args.partitions_from)
    previous=json.loads((source/'development.json').read_text())
    if previous['dataset_sha256']!=digest:raise ValueError('Dataset hash differs')
    with np.load(source/'splits.npz') as saved: split={k:saved[k] for k in saved.files}
    forbidden=set(split['test'])
    for key in ['fit','stop','select','calibrate']:
        if forbidden.intersection(split[key]):raise ValueError('Development/test overlap')
    rng=np.random.default_rng(2026)
    fit,stop,select,cal=[split[k] for k in ['fit','stop','select','calibrate']]
    evaluation=rng.choice(select,min(args.records,len(select)),replace=False)
    eval_X,eval_y=X.iloc[evaluation],y.iloc[evaluation]
    np.savez_compressed(out/'development_positions.npz',evaluation=evaluation,**split)
    spec=dict(name='mlp_regularized',family='mlp',widths=(128,64,32),alpha=.1)
    ledger=dict(status='EXPLORATORY_DEVELOPMENT_TRUST_EXPERIMENT',dataset_sha256=digest,
       source_run=args.partitions_from,final_test_accessed=False,records=len(evaluation),
       model_specification=spec,epochs=args.epochs,member_runs=[],
       notes=['Unweighted, single-year internal exploratory study',
       'Selection partition was reused in prior model development; not independent confirmation',
       'DSI/CAS/RAI are proposed summaries of existing concepts, not established novelty',
       'No causal uncertainty attribution or clinical guidance'])
    def checkpoint(): (out/'trust.json').write_text(json.dumps(ledger,indent=2,allow_nan=False))
    models=[]
    for index in range(5):
        seed=42+index
        training=fit if index<3 else np.random.default_rng(seed).choice(fit,len(fit),replace=True)
        model,history=fit_candidate(spec,X.iloc[training],y.iloc[training],X.iloc[stop],y.iloc[stop],seed,args.epochs)
        model.calibrator=LogisticRegression(C=1,max_iter=1000,random_state=seed).fit(
           probability_logit(model.predict_proba(X.iloc[cal])[:,1]),y.iloc[cal])
        models.append(model);joblib.dump(model,out/f'member_{index}.joblib')
        ledger['member_runs'].append(dict(member=index,seed=seed,
           resampling='none' if index<3 else 'training respondent bootstrap',
           stopping_history=history,development=metrics(eval_y,model.predict_proba(eval_X)[:,1])))
        checkpoint();print(f'Trust member {index+1}/5 fitted',flush=True)
    p_members=np.array([m.predict_proba(eval_X)[:,1] for m in models])
    cas=confidence_attribution(p_members)
    mean_probability=cas['mean_probability']
    threshold=.14  # Reuse declared research threshold; no new threshold tuning.
    errors=((mean_probability>=threshold)!=eval_y.to_numpy()).astype(int)
    absolute_error=np.abs(eval_y.to_numpy()-mean_probability)
    ledger['cas']=dict(development_ensemble_metrics=metrics(eval_y,mean_probability,threshold),
        mean_summaries={k:float(v.mean()) for k,v in cas.items()},
        entropy_absolute_error_spearman=safe_correlation(cas['predictive_entropy'],absolute_error),
        disagreement_absolute_error_spearman=safe_correlation(cas['ensemble_disagreement'],absolute_error),
        entropy_error_detection_auroc=float(roc_auc_score(errors,cas['predictive_entropy'])) if len(np.unique(errors))==2 else None,
        disagreement_error_detection_auroc=float(roc_auc_score(errors,cas['ensemble_disagreement'])) if len(np.unique(errors))==2 else None,
        identity_max_residual=float(np.abs(cas['predictive_entropy']-cas['mean_member_entropy']-cas['ensemble_disagreement']).max()),
        interpretation='Entropy summaries of calibrated member probabilities; not unique epistemic/aleatoric or causal components')
    checkpoint();print('CAS decomposition and error associations complete',flush=True)
    # Shared background and path seeds isolate retraining effects from MC path randomness.
    explain_X=eval_X.iloc[:args.explain_records]
    background=X.iloc[rng.choice(fit,min(2048,len(fit)),replace=False)]
    attr=[];residuals=[]
    for i,model in enumerate(models):
        value,_,residual=permutation_shapley(lambda z:model.predict_proba(z)[:,1],explain_X,background,
                                           args.permutations,seed=923)
        attr.append(value);residuals.append(float(np.abs(residual).max()))
        print(f'Original-variable Shapley member {i+1}/5 complete',flush=True)
    seed_pairs=list(combinations(range(3),2))
    boot_pairs=[(0,3),(0,4)]
    seed_summary,seed_per=stability_summary([attr[i] for i,j in seed_pairs],[attr[j] for i,j in seed_pairs])
    boot_summary,_=stability_summary([attr[i] for i,j in boot_pairs],[attr[j] for i,j in boot_pairs])
    repeat,_,_=permutation_shapley(lambda z:models[0].predict_proba(z)[:,1],explain_X,background,args.permutations,seed=924)
    noise_summary,_=stability_summary([attr[0]],[repeat])
    changed=explain_X.copy()
    for c in ['PHYSHLTH','MENTHLTH']:
        changed[c]=(changed[c]+rng.choice([-1,1],len(changed))).clip(0,30)
    changed['_BMI5']=(changed['_BMI5']+rng.choice([-.2,.2],len(changed))).clip(lower=10)
    perturbed,_,_=permutation_shapley(lambda z:models[0].predict_proba(z)[:,1],changed,background,args.permutations,seed=923)
    perturb_summary,_=stability_summary([attr[0]],[perturbed])
    dsi=seed_per['dsi']; explain_error=np.abs(y.iloc[evaluation[:len(explain_X)]].to_numpy()-p_members[:3,:len(explain_X)].mean(axis=0))
    subgroup={}
    for value in [1,2]:
        mask=explain_X.SEXVAR.to_numpy()==value
        subgroup[str(value)]=mean_interval(dsi[mask])
    ledger['dsi']=dict(seed_retraining=seed_summary,bootstrap_retraining=boot_summary,
       attribution_sampling_repeat=noise_summary,small_numeric_perturbation=perturb_summary,
       seed_dsi_absolute_error_spearman=safe_correlation(dsi,explain_error),sex_group_descriptive=subgroup,
       maximum_local_accuracy_residual=max(residuals),permutations=args.permutations,
       explanation_records=len(explain_X),background_records=len(background),
       interpretation='Original 15-variable probability-scale interventional permutation Shapley; finite MC approximation',
       limitation='Mixed coalitions may violate correlations; 128/256-record CIs are IID exploratory, not survey-aware')
    np.savez_compressed(out/'attributions.npz',attributions=np.array(attr),records=evaluation[:len(explain_X)])
    checkpoint();print('DSI seeds, bootstrap, perturbation and MC-repeat controls complete',flush=True)
    scenarios={}
    demographic=['SEXVAR','_AGEG5YR']
    history=demographic+['DIABETE4','CVDINFR4','CVDCRHD4','CHCKDNY2','HAVARTH4']
    scenarios['demographics_only']=eval_X.copy().assign(**{c:np.nan for c in FEATURES if c not in demographic})
    scenarios['demographics_history']=eval_X.copy().assign(**{c:np.nan for c in FEATURES if c not in history})
    scenarios['without_health_state']=eval_X.copy().assign(GENHLTH=np.nan,PHYSHLTH=np.nan,MENTHLTH=np.nan)
    for rate in [.1,.3,.5]:
        mask=rng.random(eval_X.shape)<rate
        scenarios[f'mcar_{rate}']=eval_X.mask(mask)
    informative=eval_X.copy()
    informative.loc[rng.random(len(eval_X))<np.where(eval_X._AGEG5YR>=10,.6,.15),'_BMI5']=np.nan
    scenarios['age_associated_bmi_missing']=informative
    full=metrics(eval_y,mean_probability,threshold)
    rows=[]
    for name,frame in scenarios.items():
        p=mean_model(models,frame);c=confidence_attribution(np.array([m.predict_proba(frame)[:,1] for m in models]))
        rows.append(dict(scenario=name,metrics=metrics(eval_y,p,threshold),
                         mean_entropy_change=float((c['predictive_entropy']-cas['predictive_entropy']).mean()),
                         masking_fraction=float(frame.isna().to_numpy().mean())))
    null=brier_score_loss(eval_y,np.repeat(y.iloc[fit].mean(),len(eval_y)))
    rai=resource_adaptability(full['brier'],[r['metrics']['brier'] for r in rows],null)
    ledger['rai']=dict(mode='Fixed predictor with training-fitted imputation; NOT subset retraining',
         full=full,null_brier=float(null),scenarios=rows,summary=rai,
         interpretation='Equal weights across seven declared scenarios; saturation hides improvement, raw losses retained',
         limitation='Simulated MCAR/age-associated masking is not an estimate of real deployment missingness')
    checkpoint();print('Feature-availability scenarios and RAI complete',flush=True)
    # Donors and global ranking come ONLY from fitting records, never evaluation labels.
    complete_fit=X.iloc[fit].dropna()
    donors=complete_fit.sample(n=min(8192,len(complete_fit)),random_state=42)
    reference=lambda z:models[0].predict_proba(z)[:,1]
    training_explain=complete_fit.sample(n=min(64,len(complete_fit)),random_state=43)
    train_attr,_,_=permutation_shapley(reference,training_explain,background,permutations=8,seed=925)
    global_order=np.argsort(-np.abs(train_attr).mean(axis=0))
    engine=ConditionalDonors(reference,donors,neighbors=64)
    complete_eval=eval_X.dropna().iloc[:args.acquisition_records]
    target=y.loc[complete_eval.index].to_numpy()
    policies={p:[] for p in ['conditional_information','random','global_importance']}
    entropies={p:[] for p in policies};initial=[];counts={p:{} for p in policies}
    for row_index,(_,record) in enumerate(complete_eval.iterrows()):
        start={c:record[c] for c in demographic}
        for policy in policies:
            observed=start.copy();probabilities=[];hs=[]
            random=np.random.default_rng(42+row_index)
            for step in range(args.acquisition_steps+1):
                if policy=='conditional_information':
                    recommended,state=engine.recommend(observed)
                else:
                    complete,prob=engine.distribution(observed)
                    state=dict(probability=float(prob.mean()),predictive_entropy=float(entropy(prob.mean())))
                    available=[c for c in FEATURES if c not in observed]
                    recommended=(random.choice(available) if policy=='random' else
                                 next(FEATURES[i] for i in global_order if FEATURES[i] in available))
                probabilities.append(state['probability']);hs.append(state['predictive_entropy'])
                if step<args.acquisition_steps:
                    if step==0:counts[policy][recommended]=counts[policy].get(recommended,0)+1
                    observed[recommended]=record[recommended]  # Hidden value read ONLY AFTER recommendation.
            policies[policy].append(probabilities);entropies[policy].append(hs)
        if (row_index+1)%32==0:print(f'Acquisition {row_index+1}/{len(complete_eval)} complete',flush=True)
    acquisition={}
    for policy,p in policies.items():
        p=np.asarray(p);h=np.asarray(entropies[policy])
        before=(target-p[:,0])**2;after=(target-p[:,-1])**2
        acquisition[policy]=dict(steps=[metrics(target,p[:,i],threshold) for i in range(p.shape[1])],
             realized_brier_reduction=mean_interval(before-after),
             entropy_reduction=mean_interval(h[:,0]-h[:,-1]),first_acquisition_counts=counts[policy])
    adaptive=np.asarray(policies['conditional_information'])
    paired={}
    for other in ['random','global_importance']:
        comparator=np.asarray(policies[other])
        difference=(target-comparator[:,-1])**2-(target-adaptive[:,-1])**2
        paired[other]=mean_interval(difference)
    ledger['acquisition']=dict(records=len(complete_eval),steps=args.acquisition_steps,
        initial_known=demographic,donor_pool=len(donors),neighbors=64,
        policies=acquisition,paired_final_brier_advantage=paired,
        interpretation='kNN empirical conditional donors; expected model-implied entropy reduction, not causal/clinical value',
        limitations=['Complete-case selected evaluation cohort','No defensible measurement costs; unit feature counts',
        'Numeric donor values grouped into four quantile bins','No claim of clinical measurement recommendations'])
    checkpoint();print('Trust experiment complete; old final test NOT accessed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--partitions-from',default='results/cdc2024-development')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--records',type=int,default=2048)
    p.add_argument('--explain-records',type=int,default=128);p.add_argument('--permutations',type=int,default=16)
    p.add_argument('--acquisition-records',type=int,default=256);p.add_argument('--acquisition-steps',type=int,default=3)
    a=p.parse_args()
    if min(a.epochs,a.records,a.explain_records,a.permutations,a.acquisition_records,a.acquisition_steps)<1:
        p.error('Experiment sizes must be positive')
    if a.acquisition_steps>13:p.error('Only 13 features remain after demographics')
    with threadpool_limits(limits=2):run(a)

if __name__=='__main__':main()
