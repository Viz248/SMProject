"""Independent PSU-total algebra versus executed R survey results."""
import json
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[1];out=root/'docs/experiments/revision_audit';d=pd.read_csv(root/'results/availability-paper-2025/posthoc_survey_domain.csv')
theta=np.sum(d.weight*d.numerator)/np.sum(d.weight*d.domain);d['z']=d.weight*(d.numerator-theta*d.domain)/np.sum(d.weight*d.domain)
p=d.groupby(['state','stratum','psu']).z.sum();groups=p.groupby(level=[0,1]);regular=[];lonely=[]
for _,g in groups:
 if len(g)==1:lonely.append(float(g.iloc[0]))
 else:regular.append(float(len(g)/(len(g)-1)*np.sum((g-g.mean())**2)))
v={'adjust':sum(regular)+sum((z-p.mean())**2 for z in lonely),'average':sum(regular)*(1+len(lonely)/len(regular))};rs=pd.read_csv(out/'r_survey.csv');checks=[]
for k,value in v.items():
 row=rs[rs.method==k].iloc[0];se=np.sqrt(value);np.testing.assert_allclose(se,row.se,rtol=1e-10,atol=1e-14);np.testing.assert_allclose(theta,row.estimate,atol=1e-15,rtol=0);checks.append(dict(method=k,python_se=float(se),r_se=float(row.se),agreement=True))
assert rs.loc[rs.method=='fail','se'].isna().all();(out/'variance_crosscheck.json').write_text(json.dumps(dict(checks=checks,df=len(p)-len(groups)),indent=2));print('R/Python agreement PASS')
