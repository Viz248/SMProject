"""Evaluate a completed development search once, without fitting or selection."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .baseline import clean, metrics, bootstrap_ci
from .optimize import load_data
from .survey import weighted_points


def evaluate(directory, data, bootstrap=500):
    directory = Path(directory)
    target = directory / 'locked_test.json'
    # Atomic reservation prevents accidental overwrite/repeated evaluation.
    with target.open('x') as reservation:
        reservation.write('{"status":"evaluation_in_progress"}')
    report = json.loads((directory / 'development.json').read_text())
    if report.get('test_accessed'):
        target.unlink()
        raise ValueError('Search has already evaluated its test partition')
    raw, digest = load_data(data)
    if digest != report['dataset_sha256']:
        target.unlink()
        raise ValueError('Input hash differs from development dataset')
    X, y, design = clean(raw)
    ids = np.load(directory / 'splits.npz')['test']
    model = joblib.load(directory / 'winner.joblib')
    p = model.predict_proba(X.iloc[ids])[:, 1]
    result = dict(status='ONE_TIME_INTERNAL_TEST', selected_model=report['winner'],
                  seed=report['seed'], dataset_sha256=digest,
                  threshold=report['threshold'], test=metrics(y.iloc[ids], p, report['threshold']),
                  iid_test_ci=bootstrap_ci(y.iloc[ids], p, report['threshold'], bootstrap, report['seed']),
                  interval_scope='IID respondents, conditional on fitted selected model; not survey-design CIs',
                  limitations=['Single-year internal validation', 'No external or clinical validation',
                               'No estimate of training variability from this single seed'])
    if '_LLCPWT' in design:
        result['survey_weighted_points'] = weighted_points(y.iloc[ids], p, design.iloc[ids]['_LLCPWT'], report['threshold'])
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    report['test_accessed'] = True
    report['status'] = 'FROZEN_INTERNAL_TEST_COMPLETED'
    report['test_result'] = 'locked_test.json'
    (directory / 'development.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.data, args.bootstrap), indent=2))
