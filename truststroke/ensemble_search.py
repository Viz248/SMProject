"""Optional development-only blend of selected model and strongest direct MLP.

This is an exploratory second search stage, not an independent confirmation.
It is prohibited after final-test access. Uses the same fitting/stopping/selection/calibration splits.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .baseline import clean, metrics
from .optimize import load_data, fit_candidate, probability_logit


class ProbabilityBlend:
    def __init__(self, base, neural, neural_weight):
        self.base, self.neural, self.neural_weight = base, neural, neural_weight
        self.calibrator = None

    def predict_proba(self, X):
        p = ((1 - self.neural_weight) * self.base.predict_proba(X)[:, 1]
             + self.neural_weight * self.neural.predict_proba(X)[:, 1])
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(probability_logit(p))[:, 1]
        return np.column_stack([1 - p, p])


def search(directory, data):
    directory = Path(directory)
    ledger = json.loads((directory / 'development.json').read_text())
    if ledger.get('test_accessed') or (directory / 'locked_test.json').exists():
        raise ValueError('Cannot tune after final-test access')
    if ledger.get('ensemble_stage'):
        raise ValueError('Ensemble search already performed')
    raw, digest = load_data(data)
    if digest != ledger['dataset_sha256']:
        raise ValueError('Dataset hash differs')
    X, y, _ = clean(raw)
    split = np.load(directory / 'splits.npz')
    fit, stop, select, cal = [split[k] for k in ['fit', 'stop', 'select', 'calibrate']]
    options = [r for r in ledger['candidates'] if r['specification']['family'] == 'mlp']
    row = max(options, key=lambda r:(r['selection']['average_precision'], -r['selection']['brier']))
    neural, history = fit_candidate(row['specification'], X.iloc[fit], y.iloc[fit],
                                    X.iloc[stop], y.iloc[stop], ledger['seed'], ledger['epochs'])
    base = joblib.load(directory / 'winner.joblib')
    base.calibrator = None
    original = metrics(y.iloc[select], base.predict_proba(X.iloc[select])[:, 1])
    best, name, key = base, ledger['winner'], (original['average_precision'], -original['brier'])
    rows = []
    for weight in [.25, .5, .75]:
        blend = ProbabilityBlend(base, neural, weight)
        result = metrics(y.iloc[select], blend.predict_proba(X.iloc[select])[:, 1])
        rows.append(dict(neural_weight=weight, selection=result))
        print(f"neural_weight={weight}: selection AP={result['average_precision']:.5f}",flush=True)
        current = (result['average_precision'], -result['brier'])
        if current > key:
            best, key, name = blend, current, f"blend_neural_{weight}"
    pcal = best.predict_proba(X.iloc[cal])[:, 1]
    best.calibrator = LogisticRegression(C=1,max_iter=1000,random_state=ledger['seed']).fit(
                          probability_logit(pcal),y.iloc[cal])
    ps = best.predict_proba(X.iloc[select])[:, 1]
    grid = np.linspace(.01,.99,99)
    ledger['threshold'] = float(grid[np.argmax([f1_score(y.iloc[select],ps>=t,zero_division=0) for t in grid])])
    ledger['ensemble_stage'] = dict(status='EXPLORATORY_DEVELOPMENT_ONLY',
           original_winner=ledger['winner'], neural_specification=row['specification'],
           neural_stopping_history=history, candidates=rows)
    ledger['winner'] = name
    joblib.dump(best,directory/'winner.joblib')
    (directory/'development.json').write_text(json.dumps(ledger,indent=2,allow_nan=False))
    print('Selected:',name,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--data',required=True)
    args=p.parse_args()
    from truststroke.ensemble_search import search as imported_search
    with threadpool_limits(limits=2):
        imported_search(args.run,args.data)
