"""Provisional trust summaries; conventional estimators, no novelty/causal claims."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def entropy(probability):
    p=np.asarray(probability,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Probabilities must be finite in [0,1]')
    q=np.clip(p,1e-15,1-1e-15)
    result=-(q*np.log2(q)+(1-q)*np.log2(1-q))
    return np.where((p==0)|(p==1),0.,result)


def confidence_attribution(member_probabilities):
    """CAS is a VECTOR of entropy summaries, not identified uncertainty causes.

    H(mean p) = mean H(p) + Jensen gap. All in bits for Bernoulli output.
    Ensemble disagreement is an operational proxy, not proven epistemic uncertainty;
    expected member entropy is not uniquely identified aleatoric uncertainty.
    """
    p=np.asarray(member_probabilities,dtype=float)
    if p.ndim!=2 or p.shape[0]<2:raise ValueError('Need at least two members by records')
    total=entropy(p.mean(axis=0)); within=entropy(p).mean(axis=0)
    between=np.maximum(0,total-within)
    return dict(predictive_entropy=total,mean_member_entropy=within,
                ensemble_disagreement=between,entropy_confidence=1-total,
                mean_probability=p.mean(axis=0))


def decision_stability(a,b,top_k=5):
    """Signed L1-normalized attribution similarity plus conventional comparators.

    DSI=1-||a/||a||1-b/||b||1||1/2 in [0,1] for informative vectors.
    Both zero: undefined, not perfect; one zero: 0 by explicit convention.
    Absolute rank Spearman and top-k Jaccard reported separately, not aggregated.
    """
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected aligned finite one-dimensional attributions')
    norms=np.abs(a).sum(),np.abs(b).sum()
    if max(norms)<1e-12:return dict(dsi=None,rank_spearman=None,top_k_jaccard=None,informative=False)
    if min(norms)<1e-12:return dict(dsi=0.,rank_spearman=None,top_k_jaccard=0.,informative=False)
    dsi=float(np.clip(1-.5*np.abs(a/norms[0]-b/norms[1]).sum(),0,1))
    k=min(top_k,len(a))
    aa=set(np.argsort(-np.abs(a),kind='stable')[:k]); bb=set(np.argsort(-np.abs(b),kind='stable')[:k])
    corr=None
    if np.ptp(np.abs(a))>0 and np.ptp(np.abs(b))>0:
        corr=float(spearmanr(np.abs(a),np.abs(b)).statistic)
    return dict(dsi=dsi,rank_spearman=corr,top_k_jaccard=len(aa&bb)/len(aa|bb),informative=True)


def permutation_shapley(predict, records, background, permutations=16, seed=42):
    """Monte Carlo interventional permutation Shapley on original feature groups.

    Same paired donor/permutation paths should be reused between model runs.
    Returns probability-scale attributions, not log-odds. Does not estimate
    conditional/causal effects; mixed coalitions may break feature dependencies.
    Telescoping paths satisfy local accuracy up to numerical precision even at
    low permutation counts, but that is not evidence of Monte Carlo convergence.
    """
    if list(records.columns)!=list(background.columns) or not len(background) or permutations<1:
        raise ValueError('Aligned records/background and positive permutations required')
    rng=np.random.default_rng(seed); n,d=records.shape
    attribution=np.zeros((n,d)); baseline=np.zeros(n)
    target=predict(records)
    for _ in range(permutations):
        current=background.iloc[rng.integers(0,len(background),n)].copy()
        current.index=records.index
        order=rng.permutation(d)
        coalitions=[current.copy()]
        for column in order:
            current=current.copy(); current.iloc[:,column]=records.iloc[:,column]
            coalitions.append(current)
        predictions=predict(pd.concat(coalitions,ignore_index=True)).reshape(d+1,n)
        baseline+=predictions[0]/permutations
        for step,column in enumerate(order):
            attribution[:,column]+=(predictions[step+1]-predictions[step])/permutations
    residual=target-baseline-attribution.sum(axis=1)
    return attribution,baseline,residual


def resource_adaptability(full_brier,scenario_briers,null_brier,weights=None):
    """RAI = weighted mean clipped Brier-headroom retention across declared scenarios.

    r_s=1-clip((B_s-B_full)/(B_null-B_full),0,1).
    Undefined if full model does not outperform constant prevalence reference.
    Improvements saturate at 1 and worse-than-null at 0; raw losses must accompany it.
    This is a proposed summary of conventional degradation, not a validated new metric.
    """
    loss=np.asarray(scenario_briers,float)
    if not len(loss) or not np.isfinite(loss).all():raise ValueError('Finite scenario losses required')
    w=np.ones(len(loss))/len(loss) if weights is None else np.asarray(weights,float)
    if w.shape!=loss.shape or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid scenario weights')
    denominator=null_brier-full_brier
    if denominator<=1e-12:return dict(rai=None,retention=None,reason='Full model has no positive Brier headroom over null')
    retention=1-np.clip((loss-full_brier)/denominator,0,1)
    return dict(rai=float(np.average(retention,weights=w)),retention=retention.tolist(),
                unbounded_relative_excess=((loss-full_brier)/denominator).tolist(),
                normalized_weights=(w/w.sum()).tolist())


def mean_interval(values,seed=42,replicates=500):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return dict(mean=None,low=None,high=None,n=0)
    rng=np.random.default_rng(seed)
    draws=np.array([a[rng.integers(0,len(a),size=len(a))].mean() for _ in range(replicates)])
    return dict(mean=float(a.mean()),low=float(np.quantile(draws,.025)),
                high=float(np.quantile(draws,.975)),n=len(a))
