"""Controlled tabular model search with a locked final-test partition.

Five disjoint partitions: fit 55%, stopping 10%, selection 15%, calibration 10%, test 10%.
The test set is evaluated only if --evaluate-test is supplied. No test-based ranking.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from .baseline import clean, build_model, synthetic, metrics, bootstrap_ci, FEATURES, DESIGN


def partitions(y, seed):
    positions = np.arange(len(y))
    fit, remaining = train_test_split(positions, test_size=.45, stratify=y, random_state=seed)
    stop, remaining = train_test_split(remaining, train_size=10/45,
                                      stratify=np.asarray(y)[remaining], random_state=seed + 1)
    select, remaining = train_test_split(remaining, train_size=15/35,
                                        stratify=np.asarray(y)[remaining], random_state=seed + 2)
    calibrate, test = train_test_split(remaining, test_size=.5,
                                      stratify=np.asarray(y)[remaining], random_state=seed + 3)
    return dict(fit=fit, stop=stop, select=select, calibrate=calibrate, test=test)


def probability_logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Predictor:
    """Serializable original-input predictor; optional encoded selection and bottleneck."""
    def __init__(self, prep, model, selector=None, neural=None, latent_scaler=None):
        self.prep, self.model, self.selector = prep, model, selector
        self.neural, self.latent_scaler = neural, latent_scaler
        self.calibrator = None

    def transform(self, X):
        encoded = np.asarray(self.prep.transform(X), dtype=np.float32)
        if self.selector is not None:
            encoded = self.selector.transform(encoded)
        if self.neural is not None:
            encoded = latent(self.neural, encoded)
            encoded = self.latent_scaler.transform(encoded)
        return encoded

    def predict_proba(self, X):
        p = self.model.predict_proba(self.transform(X))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(probability_logit(p))[:, 1]
        return np.column_stack([1 - p, p])


def latent(model, encoded):
    value = encoded
    for w, b in zip(model.coefs_[:3], model.intercepts_[:3]):
        value = np.maximum(value @ w + b, 0)
    return value


def fit_neural(X, y, stop_X, stop_y, widths, alpha, seed, epochs, weighted=False):
    """Epoch-wise Adam; early stopping by held-out average precision, then log loss.

    No preprocessing is learned from the stopping partition. Natural stopping prevalence.
    """
    model = MLPClassifier(hidden_layer_sizes=widths, activation='relu', solver='adam',
                          alpha=alpha, batch_size=min(512, len(y)), learning_rate_init=.001,
                          early_stopping=False, max_iter=1, random_state=seed)
    weights = compute_sample_weight('balanced', y) if weighted else None
    best, best_key, patience = None, (-np.inf, -np.inf), 0
    history = []
    for epoch in range(epochs):
        model.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=weights)
        p = model.predict_proba(stop_X)[:, 1]
        ap, loss = average_precision_score(stop_y, p), log_loss(stop_y, p, labels=[0, 1])
        history.append(dict(epoch=epoch + 1, stopping_ap=float(ap), stopping_log_loss=float(loss)))
        key = (float(ap), -float(loss))
        if key > best_key:
            best, best_key, patience = copy.deepcopy(model), key, 0
        else:
            patience += 1
        if patience >= 8:
            break
    return best, history


def candidates():
    # Predeclared compact search. Different families have different candidate counts;
    # this is an engineering pilot, not an equal-budget benchmark.
    return [
        dict(name='logistic_C0.1', family='logistic', C=.1),
        dict(name='logistic_C1', family='logistic', C=1),
        dict(name='logistic_balanced', family='logistic', C=1, weighted=True),
        dict(name='l1_selected_logistic', family='selected_logistic', C=1),
        dict(name='boosted_15', family='boosted', leaves=15, l2=10.),
        dict(name='boosted_31', family='boosted', leaves=31, l2=1.),
        dict(name='mlp_shallow', family='mlp', widths=(64, 32), alpha=.01),
        dict(name='mlp_deep', family='mlp', widths=(128, 64, 32), alpha=.01),
        dict(name='mlp_regularized', family='mlp', widths=(128, 64, 32), alpha=.1),
        dict(name='mlp_balanced', family='mlp', widths=(64, 32), alpha=.01, weighted=True),
        dict(name='bottleneck15_logistic', family='bottleneck', widths=(64, 32, 15, 16), alpha=.01),
    ]


def fit_candidate(spec, X, y, stop_X, stop_y, seed, epochs):
    prep = clone(build_model()[0])
    fit = np.asarray(prep.fit_transform(X, y), dtype=np.float32)
    stop = np.asarray(prep.transform(stop_X), dtype=np.float32)
    history, selector, neural, scaler = [], None, None, None
    family = spec['family']
    if family == 'selected_logistic':
        # This selects encoded dimensions, not necessarily entire clinical variables.
        selector = SelectFromModel(LogisticRegression(l1_ratio=1, solver='liblinear',
                                   C=.1, max_iter=1000, random_state=seed), threshold=1e-8)
        selected = selector.fit_transform(fit, y)
        if selected.shape[1] == 0:
            raise ValueError('L1 selector retained zero dimensions; candidate invalid')
        fit, stop = selected, selector.transform(stop)
    if family in ['mlp', 'bottleneck']:
        model, history = fit_neural(fit, y, stop, stop_y, spec['widths'], spec['alpha'],
                                   seed, epochs, spec.get('weighted', False))
        if family == 'bottleneck':
            from sklearn.preprocessing import StandardScaler
            neural = model
            scaler = StandardScaler().fit(latent(neural, fit))
            fit = scaler.transform(latent(neural, fit))
            model = LogisticRegression(C=1, max_iter=1000, random_state=seed).fit(fit, y)
    elif family == 'boosted':
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                    max_leaf_nodes=spec['leaves'], l2_regularization=spec['l2'],
                    early_stopping=True, scoring='loss', n_iter_no_change=15, random_state=seed)
        model.fit(fit, y, X_val=stop, y_val=stop_y)
    else:
        model = LogisticRegression(C=spec['C'], max_iter=1000, random_state=seed,
                                   class_weight='balanced' if spec.get('weighted') else None).fit(fit, y)
    return Predictor(prep, model, selector, neural, scaler), history


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path), sha256(path)
    # Chunked XPT reads avoid retaining all 345 survey columns in memory.
    frames = []
    for chunk in pd.read_sas(path, format='xport', chunksize=20000):
        columns = FEATURES + ['CVDSTRK3'] + [c for c in DESIGN if c in chunk]
        frames.append(chunk[columns])
    return pd.concat(frames, ignore_index=True), sha256(path)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw, digest = (synthetic(2000, args.seed), None) if args.smoke else load_data(args.data)
    X, y, design = clean(raw)
    splits = partitions(y, args.seed)
    # Fixed training-only subsample for feasible pilot runs. Other partitions unchanged.
    if args.fit_limit and len(splits['fit']) > args.fit_limit:
        splits['fit'], _ = train_test_split(splits['fit'], train_size=args.fit_limit,
                          stratify=y.iloc[splits['fit']], random_state=args.seed)
    np.savez_compressed(output / 'splits.npz', **splits)
    ledger = dict(status='SYNTHETIC_SMOKE_ONLY' if args.smoke else 'DEVELOPMENT_SEARCH',
        task='self-reported stroke-history classification', seed=args.seed, dataset_sha256=digest,
        python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
        source=str(args.data) if not args.smoke else 'artificial fixture',
        input_records=len(raw), labeled_records=len(X), excluded_target_records=len(raw)-len(X),
        prevalence=float(y.mean()), split_counts={k: len(v) for k, v in splits.items()},
        fit_limit=args.fit_limit, epochs=args.epochs, threads=args.threads,
        selection_rule='maximum selection average precision; tie: minimum Brier then candidate order',
        design_columns=list(design), candidates=[], failures=[], test_accessed=False,
        limitations=['Internal unweighted validation only; no population inference',
                     'Search budgets differ across families; pilot not a fair family benchmark',
                     'Codebook verification and consequence-sensitive ablations required'])
    fit, stop, select, cal, test = [splits[k] for k in ['fit', 'stop', 'select', 'calibrate', 'test']]
    winner, key_best, best_name = None, (-np.inf, -np.inf), None
    for spec in candidates():
        started = time.perf_counter()
        try:
            model, history = fit_candidate(spec, X.iloc[fit], y.iloc[fit], X.iloc[stop], y.iloc[stop],
                                            args.seed, args.epochs)
            p = model.predict_proba(X.iloc[select])[:, 1]
            score = metrics(y.iloc[select], p)
            names = list(model.prep.get_feature_names_out())
            if model.selector is not None:
                names = np.asarray(names)[model.selector.get_support()].tolist()
            row = dict(specification=spec, seconds=time.perf_counter() - started,
                       selection=score, stopping_history=history, selected_encoded_inputs=names)
            ledger['candidates'].append(row)
            key = (score['average_precision'], -score['brier'])
            if key > key_best:
                winner, key_best, best_name = model, key, spec['name']
            print(f"{spec['name']}: selection AP={score['average_precision']:.5f}, Brier={score['brier']:.5f}", flush=True)
        except Exception as error:
            ledger['failures'].append(dict(candidate=spec['name'], error=repr(error)))
            print(f"{spec['name']}: FAILED {error!r}", flush=True)
        (output / 'development.json').write_text(json.dumps(ledger, indent=2, allow_nan=False))
    if winner is None:
        raise RuntimeError('All candidates failed; inspect development.json')
    ledger['winner'] = best_name
    # Predetermined sigmoid calibration: natural-prevalence calibration partition.
    pcal = winner.predict_proba(X.iloc[cal])[:, 1]
    winner.calibrator = LogisticRegression(C=1, max_iter=1000, random_state=args.seed).fit(
                           probability_logit(pcal), y.iloc[cal])
    ps = winner.predict_proba(X.iloc[select])[:, 1]
    thresholds = np.linspace(.01, .99, 99)
    threshold = float(thresholds[np.argmax([f1_score(y.iloc[select], ps >= t, zero_division=0) for t in thresholds])])
    ledger['threshold'] = threshold
    ledger['calibration'] = 'L2 sigmoid on clipped probability logits; fitted only on calibration partition'
    ledger['threshold_rule'] = 'maximum calibrated selection F1; smallest threshold on ties'
    joblib.dump(winner, output / 'winner.joblib')
    if args.evaluate_test:
        pt = winner.predict_proba(X.iloc[test])[:, 1]
        ledger['test_accessed'] = True
        ledger['test'] = metrics(y.iloc[test], pt, threshold)
        ledger['iid_test_ci'] = bootstrap_ci(y.iloc[test], pt, threshold, args.bootstrap, args.seed)
    (output / 'development.json').write_text(json.dumps(ledger, indent=2, allow_nan=False))
    print(json.dumps({k: ledger[k] for k in ['status', 'winner', 'test_accessed']}, indent=2))
    return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', help='Local XPT or reduced CSV, preserving original BRFSS codes')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--fit-limit', type=int, default=0, help='Training-only pilot cap; 0 means all')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--evaluate-test', action='store_true', help='One-time final test of selected model')
    args = p.parse_args()
    if not args.smoke and not args.data:
        p.error('--data or --smoke is required')
    if args.epochs < 1 or args.threads < 1 or args.fit_limit < 0 or args.bootstrap < 1:
        p.error('epochs, threads and bootstrap must be positive; fit-limit must be nonnegative')
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    # Use importable class identities in joblib artifacts created via -m.
    from truststroke.optimize import main as imported_main
    imported_main()
