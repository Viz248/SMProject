"""Survey-weighted point metrics only; no design-based variance estimator."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix


def weighted_points(y, p, weights, threshold):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(weights, dtype=float)
    if y.shape != p.shape or y.shape != w.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Expected aligned finite positive survey weights')
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, sample_weight=w, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(prevalence=float(np.average(y, weights=w)),
                auroc=float(roc_auc_score(y, p, sample_weight=w)) if both else None,
                average_precision=float(average_precision_score(y, p, sample_weight=w)) if both else None,
                brier=float(brier_score_loss(y, p, sample_weight=w)),
                sensitivity=float(tp / (tp + fn)) if tp + fn else None,
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                precision=float(tp / (tp + fp)) if tp + fp else 0.,
                interpretation='Survey-weighted point estimates; no design-aware intervals')
