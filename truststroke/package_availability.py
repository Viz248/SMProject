"""Aggregate-only publication tables/figures from the frozen availability study."""
import argparse,json,csv,shutil,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42
from .availability_study import score_bundle,linearized_mean,ensemble_bundle
from .baseline import clean
from .optimize import load_data,sha256

SHORT={'fixed_global_calibration':'Global calibration','fixed_scenario_calibration':'Scenario calibration','mask_trained':'Mask training','subset_retrained':'Subset refit'}
SCENARIOS={'full':'All 15','without_health_state':'Without health state (12)','demographics_history':'Demographics + history (7)','demographics':'Demographics (2)'}

def run(a):
    root=Path(a.run);ext=Path(a.external);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((root/'protocol.json').read_text());r=json.loads((ext/'results.json').read_text());manifest=json.loads((root/'model_manifest.json').read_text())
    assert json.loads(Path('configs/availability_study_frozen.json').read_text())==cfg
    assert sha256('configs/availability_study_frozen.json')==manifest['protocol_sha256']
    for name in ['protocol.json','model_manifest.json','training.json','development.json']:shutil.copy2(root/name,out/name)
    shutil.copy2(ext/'results.json',out/'transport2025.json')
    # Common-jurisdiction development summary for descriptive temporal comparisons.
    raw,digest=load_data(a.development_data);X,y,design=clean(raw)
    assert digest==cfg['input_sha256_2024_xpt']
    with np.load(root/'positions.npz') as s:ix=s['select']
    with np.load(root/'development_predictions.npz') as p:
        assert np.array_equal(p['labels'],np.asarray(y.iloc[ix]))
        pred={k:p[k] for k in p.files if k!='labels'}
    common=np.isin(design.iloc[ix]['_STATE'],r['common_states']);pp={k:v[common] for k,v in pred.items()}
    dc=design.iloc[ix].loc[common].reset_index(drop=True);yc=np.asarray(y.iloc[ix])[common]
    devcommon=dict(n=len(yc),positives=int(yc.sum()),metrics=score_bundle(pp,yc,dc,cfg))
    (out/'development_common.json').write_text(json.dumps(devcommon,indent=2))
    cohort=dict(raw2024=len(raw),labeled2024=len(y),positive2024=int(y.sum()),excluded2024=len(raw)-len(y),
      raw2025=r['raw_records'],labeled2025=r['labeled_records'],positive2025=r['positives'],excluded2025=r['excluded_outcomes'],
      common2025=r['common_records'],common_positive2025=r['common_positives'],common_jurisdictions=len(r['common_states']),common_state_codes=r['common_states'],
      missing_predictor_fraction2024={c:float(X[c].isna().mean()) for c in cfg['features']})
    (out/'cohort.json').write_text(json.dumps(cohort,indent=2))
    rows=[]
    for cohortname,metrics in [('2024_development_common',devcommon['metrics']),('2025_common',r['common_metrics']),('2025_all',r['all_metrics'])]:
        for key,v in metrics.items():
            family,scenario,condition=key.split('|');rows.append(dict(cohort=cohortname,family=family,scenario=scenario,condition=condition,n=v['n'],auroc=v['auroc'],average_precision=v['average_precision'],brier=v['brier'],nll=v['nll'],ece=v['ece'],weighted_auroc=v['weighted']['auroc'],weighted_average_precision=v['weighted']['average_precision'],weighted_brier=v['weighted']['brier'],weighted_nll=v['weighted']['nll']))
    pd.DataFrame(rows).to_csv(out/'all_metrics.csv',index=False)
    # Training seed ranges are descriptive; ensemble probability is primary.
    with np.load(ext/'predictions.npz') as pred:
        labels=pred['labels'];D=joblib.load(ext/'design.joblib');mask=np.isin(D['_STATE'],r['common_states']);D=D.loc[mask].reset_index(drop=True);labels=labels[mask]
        predictions={k:pred[k][mask] for k in pred.files if k!='labels'}
    ensemble=ensemble_bundle(predictions,cfg)
    contrasts=[];verify=0
    for key,p in ensemble.items():
        expected=r['common_metrics'][key]['weighted']['brier'];actual=float(np.average((labels-p)**2,weights=D['_LLCPWT']))
        assert np.isclose(expected,actual,atol=1e-9);verify+=1
    for family in cfg['families']:
        for scenario in cfg['scenarios']:
            baseline=ensemble[f'{family}|{scenario}|fixed_scenario_calibration']
            for condition in ['mask_trained','subset_retrained','fixed_global_calibration']:
                p=ensemble[f'{family}|{scenario}|{condition}'];stat=linearized_mean((labels-p)**2-(labels-baseline)**2,D)
                contrasts.append(dict(family=family,scenario=scenario,condition=condition,contrast=stat))
    (out/'secondary_contrasts.json').write_text(json.dumps(contrasts,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    scenarios=list(cfg['scenarios']);x=np.arange(4)
    for ax,family in zip(axes,cfg['families']):
        for j,condition in enumerate(cfg['conditions']):
            val=[r['common_metrics'][f'{family}|{s}|{condition}']['weighted']['brier'] for s in scenarios]
            ax.plot(x,val,marker='o',label=SHORT[condition],linewidth=1.5)
        ax.set(xticks=x,xticklabels=['15','12','7','2'],xlabel='Available variables (predefined sets)',ylabel='Survey-weighted Brier loss',title='Boosted trees' if family=='hist_gradient_boosting' else 'MLP')
        ax.legend(fontsize=7)
    fig.savefig(out/'availability.pdf');fig.savefig(out/'availability.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.5,3.5),layout='constrained')
    maximum=.0
    for condition in cfg['conditions']:
        rows=r['common_metrics'][f'hist_gradient_boosting|demographics_history|{condition}']['calibration'];rows=[v for v in rows if v['count']>=100]
        p=[v['probability'] for v in rows];q=[v['observed'] for v in rows]
        if p:maximum=max(maximum,max(p),max(q));ax.plot(p,q,marker='o',label=SHORT[condition])
    lim=min(1,max(.2,maximum*1.1));ax.plot([0,lim],[0,lim],ls='--',color='grey',lw=1)
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Mean predicted probability',ylabel='Observed stroke-history proportion',title='2025: demographics + history')
    ax.legend(fontsize=7);fig.savefig(out/'calibration.pdf');fig.savefig(out/'calibration.png',dpi=200);plt.close(fig)
    groups=r['subgroups']
    fig,ax=plt.subplots(figsize=(5,3.5),layout='constrained')
    for i,g in enumerate(groups):
        q=g['primary']
        if q['ci95'] is not None:
            lo,hi=q['ci95'];ax.errorbar(q['estimate'],i,xerr=[[q['estimate']-lo],[hi-q['estimate']]],fmt='o',color='tab:blue',capsize=3)
        else:ax.plot(q['estimate'],i,'o',color='tab:blue')
    ax.axvline(0,color='grey',ls='--');ax.set(yticks=range(len(groups)),yticklabels=[{'sex_male':'Male','sex_female':'Female','age18_44':'Age 18-44','age45_64':'Age 45-64','age65plus':'Age 65+','income_under35k':'Income <35k','income35kplus':'Income >=35k'}[g['group']] for g in groups],xlabel='Brier difference (MT - SC)',title='Domain point contrasts');ax.invert_yaxis();ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
    fig.savefig(out/'subgroups.pdf');fig.savefig(out/'subgroups.png',dpi=200);plt.close(fig)
    provenance=dict(official_2024='https://www.cdc.gov/brfss/annual_data/annual_2024.html',official_2025='https://www.cdc.gov/brfss/annual_data/annual_2025.html',
      xpt2024=cfg['input_sha256_2024_xpt'],xpt2025=cfg['input_sha256_2025_xpt'],codebook2025='https://www.cdc.gov/brfss/annual_data/2025/zip/codebook25_llcp-v2-508.zip',
      codebook2025_zip_sha256='70d920a440fd651a12f9dc62c100626a15ca2503774128ba2cbe62b056f861f3',
      zip2025_sha256='d26a8b391aa2966b58dc2a3ffddf8da050e9f38eb143d0f5ca7b3d21ece08f98',
      scope='independently downloaded official CDC records, not byte-verified against original Drive copy',protocol_commit='4772823',
      historical_first_fit_failure='Dropped ColumnTransformer remainder was treated as a transformer; fixed before 2025 prediction, same scientific settings, preserved failed output directory')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (out/'validation.json').write_text(json.dumps(dict(ensemble_weighted_brier_checks=verify,protocol_content_matches_committed_config=True,config_hash_matches_manifest=True,source_hash_matches_manifest=sha256('truststroke/availability_study.py')==manifest['source_sha256'],
      raw_participant_data_in_package=False,full_external_predictions_stored_privately=True,models_fitted=30,ensemble_comparisons=32,
      first_model_fit_failed_before_external_evaluation=True,pre_evaluation_freeze=True),indent=2))
    print('PACKAGED',json.dumps(r['primary']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--external',required=True);p.add_argument('--development-data',required=True);p.add_argument('--output',required=True);run(p.parse_args())
