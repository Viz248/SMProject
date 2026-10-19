"""Quantify explanation MC noise before interpreting retraining DSI."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .baseline import clean
from .optimize import load_data
from .trust import permutation_shapley
from .trust_experiment import stability_summary


def run(a):
    root=Path(a.run);target=root/'attribution_convergence.json'
    if target.exists():raise FileExistsError(target)
    raw,digest=load_data(a.data);X,y,_=clean(raw)
    report=json.loads((root/'trust.json').read_text())
    if digest!=report['dataset_sha256']:raise ValueError('Dataset mismatch')
    with np.load(root/'development_positions.npz') as s:
        evaluation=s['evaluation'];fit=s['fit']
    # Match original deterministic RNG choices and shared background exactly.
    rng=np.random.default_rng(2026)
    with np.load(Path(report['source_run'])/'splits.npz') as s:
        expected=rng.choice(s['select'],min(report['records'],len(s['select'])),replace=False)
    np.testing.assert_array_equal(expected,evaluation)
    background=X.iloc[rng.choice(fit,min(2048,len(fit)),replace=False)]
    records=X.iloc[evaluation[:report['dsi']['explanation_records']]]
    models=[joblib.load(root/f'member_{i}.joblib') for i in range(5)]
    result=dict(status='DEVELOPMENT_ATTRIBUTION_MONTE_CARLO_CONTROL',dataset_sha256=digest,
                final_test_accessed=False,records=len(records),repeats=[])
    for permutations in [16,64,256]:
        first,_,_=permutation_shapley(lambda z:models[0].predict_proba(z)[:,1],records,background,permutations,923)
        second,_,_=permutation_shapley(lambda z:models[0].predict_proba(z)[:,1],records,background,permutations,924)
        summary,_=stability_summary([first],[second])
        result['repeats'].append(dict(permutations=permutations,same_model_repeat=summary))
        print('Same-model attribution repeat',permutations,summary['dsi'],flush=True)
        target.write_text(json.dumps(result,indent=2,allow_nan=False))
    attributions=[first]
    for i,model in enumerate(models[1:],1):
        value,_,_=permutation_shapley(lambda z:model.predict_proba(z)[:,1],records,background,256,923)
        attributions.append(value);print('High-budget attribution member',i+1,flush=True)
    pairs=[(0,1),(0,2),(1,2)]
    result['seed_retraining_256'],_=stability_summary([attributions[i] for i,j in pairs],[attributions[j] for i,j in pairs])
    result['bootstrap_retraining_256'],_=stability_summary([attributions[0]]*2,[attributions[3],attributions[4]])
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(root/'attributions_256.npz',attributions=np.array(attributions),positions=evaluation[:len(records)])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--run',required=True)
    with threadpool_limits(limits=2):run(p.parse_args())
