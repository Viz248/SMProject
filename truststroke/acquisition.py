"""Training-only conditional-donor acquisition prototype; questionnaire items, not clinical advice."""
import numpy as np
import pandas as pd
from .baseline import CATEGORICAL, NUMERIC, FEATURES
from .trust import entropy


class ConditionalDonors:
    """Empirical k-nearest complete donors conditioned only on observed inputs.

    Mixed distance: categorical mismatch, numeric absolute distance divided by
    training IQR (clipped to 1). Equal feature weights. Labels never enter donor
    retrieval. This is an approximation to conditional feature distributions,
    not a validated generative/clinical model.
    """
    def __init__(self,predict,donors,neighbors=64):
        if donors[FEATURES].isna().any().any():raise ValueError('Donors must be complete')
        if not len(donors):raise ValueError('No donors')
        self.predict=predict;self.donors=donors[FEATURES].reset_index(drop=True).copy()
        self.neighbors=min(neighbors,len(donors))
        iqr=self.donors[NUMERIC].quantile(.75)-self.donors[NUMERIC].quantile(.25)
        self.scale=iqr.replace(0,1)

    def distribution(self,observed):
        # Accept only supplied observed fields. No hidden full record is available here.
        if any(c not in FEATURES for c in observed):raise ValueError('Unknown observed field')
        distance=np.zeros(len(self.donors))
        for c,value in observed.items():
            if pd.isna(value):raise ValueError('Observed feature must have known value')
            if c in CATEGORICAL:distance+=(self.donors[c].to_numpy()!=value)
            else:distance+=np.minimum(1,np.abs(self.donors[c].to_numpy()-value)/self.scale[c])
        # Stable order makes ties deterministic; arbitrary donor order is a limitation.
        idx=np.argsort(distance,kind='stable')[:self.neighbors]
        complete=self.donors.iloc[idx].copy()
        for c,value in observed.items():complete[c]=value
        p=np.asarray(self.predict(complete))
        return complete,p

    def evaluate(self,observed):
        complete,p=self.distribution(observed)
        current=float(p.mean()); current_entropy=float(entropy(current))
        gains={}
        for c in FEATURES:
            if c in observed:continue
            values=complete[c].to_numpy()
            if c in NUMERIC:
                # Four conditional-donor quantile bins; approximation to numerical acquisition.
                edges=np.unique(np.quantile(values,[.25,.5,.75]))
                group=np.searchsorted(edges,values,side='right')
            else:group=values
            expected=0.
            for level in np.unique(group):
                subset=group==level
                expected+=subset.mean()*float(entropy(p[subset].mean()))
            gains[c]=max(0.,current_entropy-expected)  # Jensen gap; numerical roundoff clipped.
        return dict(probability=current,predictive_entropy=current_entropy,expected_information_gain=gains)

    def recommend(self,observed):
        result=self.evaluate(observed)
        if not result['expected_information_gain']:return None,result
        return max(result['expected_information_gain'],key=result['expected_information_gain'].get),result
