"""Matched feature-set retraining versus fixed-model masking (development only)."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .baseline import clean,FEATURES,CATEGORICAL,NUMERIC,metrics
from .optimize import load_data,Predictor,probability_logit


def fit_subset(columns,X,y,stop_X,stop_y,cal_X,cal_y):
    cat=[c for c in CATEGORICAL if c in columns];num=[c for c in NUMERIC if c in columns]
    prep=ColumnTransformer([
       ('categorical',make_pipeline(SimpleImputer(strategy='most_frequent',keep_empty_features=True),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cat),
       ('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True),StandardScaler()),num)])
    fit=np.asarray(prep.fit_transform(X),np.float32);stop=np.asarray(prep.transform(stop_X),np.float32)
    estimator=HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=31,
       l2_regularization=1.,early_stopping=True,scoring='loss',n_iter_no_change=15,random_state=42)
    estimator.fit(fit,y,X_val=stop,y_val=stop_y)
    model=Predictor(prep,estimator)
    model.calibrator=LogisticRegression(C=1,max_iter=1000,random_state=42).fit(
       probability_logit(model.predict_proba(cal_X)[:,1]),cal_y)
    return model


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    raw,digest=load_data(a.data);X,y,_=clean(raw)
    root=Path(a.trust_run);meta=json.loads((root/'trust.json').read_text())
    if digest!=meta['dataset_sha256']:raise ValueError('Dataset mismatch')
    with np.load(root/'development_positions.npz') as s:
        fit,stop,cal,evaluation=[s[k] for k in ['fit','stop','calibrate','evaluation']]
        forbidden=set(s['test'])
    if forbidden.intersection(evaluation):raise ValueError('Inspected test overlap')
    demo=['SEXVAR','_AGEG5YR']
    scenarios={'full':FEATURES,'demographics':demo,
       'demographics_history':demo+['DIABETE4','CVDINFR4','CVDCRHD4','CHCKDNY2','HAVARTH4'],
       'without_health_state':[c for c in FEATURES if c not in ['GENHLTH','PHYSHLTH','MENTHLTH']]}
    result=dict(status='DEVELOPMENT_AVAILABILITY_RETRAINING_ABLATION',dataset_sha256=digest,
        final_test_accessed=False,records=len(evaluation),family='histogram boosted trees',scenarios=[],
        note='Each subset refitted with identical declared hyperparameters/seed, train-only preprocessing and separate calibration; no hyperparameter retuning')
    full=None
    for name,columns in scenarios.items():
        model=fit_subset(columns,X.iloc[fit],y.iloc[fit],X.iloc[stop],y.iloc[stop],X.iloc[cal],y.iloc[cal])
        if name=='full':full=model
        masked=X.iloc[evaluation].copy()
        for c in FEATURES:
            if c not in columns:masked[c]=np.nan
        row=dict(scenario=name,features=columns,retrained=metrics(y.iloc[evaluation],model.predict_proba(X.iloc[evaluation])[:,1],.14),
                 fixed_mask=metrics(y.iloc[evaluation],full.predict_proba(masked)[:,1],.14))
        result['scenarios'].append(row);joblib.dump(model,out/f'{name}.joblib')
        (out/'availability.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(name,'retrained AP',row['retrained']['average_precision'],'fixed-mask AP',row['fixed_mask']['average_precision'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--trust-run',required=True);p.add_argument('--output',required=True)
    with threadpool_limits(limits=2):run(p.parse_args())
