"""One-time frozen 2024-trained model comparison on a separate BRFSS year.

All models and development-based winner are frozen BEFORE loading external labels.
No fitting, calibration or threshold tuning occurs on external data.
"""
import argparse,json
from pathlib import Path
import joblib
import torch
import numpy as np
from threadpoolctl import threadpool_limits
from .baseline import clean,metrics
from .optimize import load_data,sha256
from .survey import weighted_points
from .trust import mean_interval


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    original=Path(args.original_run);tabnet=Path(args.tabnet_run)
    first=json.loads((original/'development.json').read_text())
    second=json.loads((tabnet/'development.json').read_text())
    # Blend AP recorded in original development-only search; calibration is monotonic.
    baseline_ap=max([r['selection']['average_precision'] for r in first['candidates']]+
                    [r['selection']['average_precision'] for r in first['ensemble_stage']['candidates']])
    tabnet_ap=second['calibrated_selection']['average_precision']
    declared='tabnet' if tabnet_ap>baseline_ap else 'original_blend'
    models={'original_blend':original/'winner.joblib','tabnet':tabnet/'winner.joblib'}
    manifest=dict(status='FROZEN_BEFORE_EXTERNAL_EVALUATION',selection_rule='higher 2024 selection AP',
        selected_model=declared,development_average_precision=dict(original_blend=baseline_ap,tabnet=tabnet_ap),
        model_artifact_sha256={k:sha256(v) for k,v in models.items()},
        training_year=2024,evaluation_year=args.year,threshold=.14,
        limitation='Reverse-calendar survey-year transport; NOT prospective validation or incident stroke prediction')
    (out/'frozen_protocol.json').write_text(json.dumps(manifest,indent=2))
    raw,digest=load_data(args.data);X,y,design=clean(raw)
    results={};predictions={}
    for name,path in models.items():
        model=joblib.load(path)
        p=model.predict_proba(X)[:,1];predictions[name]=p
        results[name]=dict(metrics=metrics(y,p,.14),
            survey_weighted_points=weighted_points(y,p,design['_LLCPWT'],.14) if '_LLCPWT' in design else None)
        print(name,results[name]['metrics']['auroc'],results[name]['metrics']['average_precision'],flush=True)
    a=(y.to_numpy()-predictions['original_blend'])**2
    b=(y.to_numpy()-predictions['tabnet'])**2
    report=dict(status='EXTERNAL_YEAR_TRANSPORT_EVALUATION',source='Official CDC BRFSS release',
        dataset_sha256=digest,raw_records=len(raw),labeled_records=len(y),positive_records=int(y.sum()),
        selected_before_external_evaluation=declared,results=results,
        paired_brier_advantage_tabnet_over_blend=mean_interval(a-b,replicates=500),
        limitations=['Models trained in 2024 evaluated in earlier 2023 survey, not prospective forecasting',
          'Different participating-state coverage and population mix',
          'IID paired Brier CI, not complex-survey design CI',
          'No external-label fitting, calibration or threshold selection; no clinical validation'])
    np.savez_compressed(out/'local_predictions.npz',target=y.to_numpy(),**predictions)
    (out/'external.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('External evaluation complete; do not tune against these outcomes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--original-run',required=True)
    p.add_argument('--tabnet-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--year',type=int,default=2023)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(p.parse_args())
