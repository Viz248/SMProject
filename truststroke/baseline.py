"""Split-first baseline. Run with python -m truststroke.baseline --help."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score, f1_score)

CATEGORICAL = ['SEXVAR', '_AGEG5YR', 'GENHLTH', 'EXERANY2', '_SMOKER3',
               'DIABETE4', 'CVDINFR4', 'CVDCRHD4', 'CHCKDNY2', 'HAVARTH4', 'EDUCA', 'INCOME3']
NUMERIC = ['PHYSHLTH', 'MENTHLTH', '_BMI5']
FEATURES = CATEGORICAL + NUMERIC
DESIGN = ['_LLCPWT', '_STSTR', '_PSU', '_STATE']


def clean(frame):
    """Deterministic recodes retained from legacy code; no fitted statistics.

    Codebook review of every predictor remains a gate before real-data reporting.
    Survey fields are retained separately and never used as predictors.
    """
    required = FEATURES + ['CVDSTRK3']
    absent = set(required) - set(frame.columns)
    if absent:
        raise ValueError(f'Missing variables: {sorted(absent)}')
    if not frame.index.is_unique:
        raise ValueError('Record index must be unique')
    d = frame.loc[frame.CVDSTRK3.isin([1, 2]), required + [c for c in DESIGN if c in frame]].copy()
    for c in CATEGORICAL:
        codes = {'_AGEG5YR': [14], '_SMOKER3': [9], 'EDUCA': [9],
                 'INCOME3': [77, 99]}.get(c, [7, 9])
        d[c] = d[c].replace(codes, np.nan)
    for c in ['PHYSHLTH', 'MENTHLTH']:
        d[c] = d[c].replace({88: 0, 77: np.nan, 99: np.nan})
    d['_BMI5'] = d['_BMI5'].replace([0, 9999], np.nan) / 100
    return d[FEATURES], d.CVDSTRK3.map({1: 1, 2: 0}).astype(int), d[[c for c in DESIGN if c in d]]


def split_indices(y, seed=42):
    train, rest = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=seed)
    validation, test = train_test_split(rest, test_size=.5, stratify=np.asarray(y)[rest], random_state=seed)
    return train, validation, test


def build_model(name='logistic', seed=42):
    prep = ColumnTransformer([
        ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                                     OneHotEncoder(handle_unknown='ignore', sparse_output=False)), CATEGORICAL),
        ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True),
                                 StandardScaler()), NUMERIC)])
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=seed),
        'forest': RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                        n_jobs=2, random_state=seed),
        'mlp': MLPClassifier(hidden_layer_sizes=(64, 32, 15, 16), early_stopping=True,
                             max_iter=80, n_iter_no_change=8, random_state=seed)}
    return make_pipeline(prep, models[name])


def calibration(y, p, bins=10):
    """Equal-width positive-probability ECE; empty bins omitted."""
    y, p = np.asarray(y), np.asarray(p)
    ids = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = ids == b
        if mask.any():
            rows.append(dict(bin=b, count=int(mask.sum()), probability=float(p[mask].mean()),
                             observed=float(y[mask].mean())))
    ece = sum(r['count'] * abs(r['probability'] - r['observed']) for r in rows) / len(y)
    return float(ece), rows


def metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty aligned one-dimensional arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0,1]')
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ece, curve = calibration(y, p)
    both = len(np.unique(y)) == 2
    return dict(n=len(y), prevalence=float(y.mean()), threshold=float(threshold),
                auroc=float(roc_auc_score(y, p)) if both else None,
                average_precision=float(average_precision_score(y, p)) if both else None,
                brier=float(brier_score_loss(y, p)), ece=ece, calibration=curve,
                precision=float(precision_score(y, pred, zero_division=0)),
                sensitivity=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                f1=float(f1_score(y, pred, zero_division=0)),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def bootstrap_ci(y, p, threshold, repetitions=200, seed=42):
    """IID respondent percentile intervals, conditional on fitted model.

    NOT survey-design intervals, nor training-variability intervals.
    """
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ['auroc', 'average_precision', 'brier', 'sensitivity', 'specificity', 'f1']}
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        m = metrics(np.asarray(y)[idx], np.asarray(p)[idx], threshold)
        for key in values:
            if m[key] is not None:
                values[key].append(m[key])
    return {k: {'low': float(np.quantile(v, .025)), 'high': float(np.quantile(v, .975)),
                'valid_replicates': len(v)} for k, v in values.items() if v}


def synthetic(n=1000, seed=42):
    """Artificial software smoke fixture, not a BRFSS distribution simulator."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.integers(1, 3, n).astype(float) for c in CATEGORICAL})
    d['PHYSHLTH'] = rng.integers(0, 31, n)
    d['MENTHLTH'] = rng.integers(0, 31, n)
    d['_BMI5'] = rng.integers(1800, 4000, n)
    p = 1 / (1 + np.exp(-(-3 + .07 * d.PHYSHLTH)))
    d['CVDSTRK3'] = np.where(rng.random(n) < p, 1, 2)
    d.loc[rng.random(n) < .1, '_BMI5'] = np.nan
    return d


def load_data(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return pd.read_csv(path), h.hexdigest()
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(block)
    digest = hasher.hexdigest()
    frames = []
    for chunk in pd.read_sas(path, format='xport', chunksize=20000):
        columns = FEATURES + ['CVDSTRK3'] + [c for c in DESIGN if c in chunk]
        frames.append(chunk[columns])
    return pd.concat(frames, ignore_index=True), digest


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    if args.smoke:
        raw, digest = synthetic(seed=args.seed), None
    else:
        if not args.data:
            raise ValueError('--data is required unless --smoke is selected')
        raw, digest = load_data(args.data)
    X, y, design = clean(raw)
    train, val, test = split_indices(y, args.seed)
    model = build_model(args.model, args.seed)
    model.fit(X.iloc[train], y.iloc[train])
    pv = model.predict_proba(X.iloc[val])[:, 1]
    grid = np.linspace(.01, .99, 99)
    threshold = float(grid[np.argmax([f1_score(y.iloc[val], pv >= t, zero_division=0) for t in grid])])
    pt = model.predict_proba(X.iloc[test])[:, 1]
    report = dict(status='SYNTHETIC_SMOKE_ONLY' if args.smoke else 'INTERNAL_UNWEIGHTED_BASELINE',
                  task='self-reported stroke-history classification', model=args.model, seed=args.seed,
                  dataset_sha256=digest, python=platform.python_version(), sklearn=sklearn.__version__,
                  numpy=np.__version__, pandas=pd.__version__, split_sizes=list(map(len, [train, val, test])),
                  survey_fields_present=list(design.columns), threshold_selection='maximum validation F1; smallest threshold on ties',
                  validation=metrics(y.iloc[val], pv, threshold), test=metrics(y.iloc[test], pt, threshold),
                  iid_test_ci=bootstrap_ci(y.iloc[test], pt, threshold, args.bootstrap, args.seed),
                  limitations=['No external validation', 'No complex-survey variance estimation',
                               'Predictor recodes require complete official codebook verification',
                               'No calibration fit or hyperparameter search in this first baseline'])
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(out / 'split_positions.npz', train=train, validation=val, test=test)
    # Record-level data and fitted models stay local and are gitignored.
    joblib.dump(model, out / 'model.joblib')
    print(json.dumps({k: report[k] for k in ['status', 'split_sizes', 'test']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['logistic', 'forest', 'mlp'], default='logistic')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=200)
    run(parser.parse_args())
