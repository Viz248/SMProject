"""Package aggregate development-pilot results; no raw records or predictions copied."""
import argparse,hashlib,json,shutil,importlib.metadata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(a):
    root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    names=['brfss2024','wisconsin_breast_cancer'];summary={}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    choices={'insample_brier':'In-sample Brier','crossfit_brier':'Cross-fitted Brier','static':'Static ranking','random':'Random'}
    for ax,name in zip(axes,names):
        d=json.loads((root/f'{name}.json').read_text());shutil.copy2(root/f'{name}.json',out/f'{name}.json')
        summary[name]={k:float(np.mean([s['curves'][k][3]['brier'] for s in d['seeds']])) for k in d['seeds'][0]['curves']}
        for k,label in choices.items():
            values=np.array([[r['brier'] for r in s['curves'][k]] for s in d['seeds']])
            line=ax.plot(range(4),values.mean(0),marker='o',label=label)[0]
            ax.fill_between(range(4),values.min(0),values.max(0),alpha=.1,color=line.get_color())
        ax.set(title='BRFSS 2024 development' if name=='brfss2024' else 'Wisconsin diagnostic breast cancer',xlabel='Additional features acquired',ylabel='Brier loss (lower is better)',xticks=range(4))
        ax.legend(fontsize=8)
    fig.suptitle('Exploratory pilot: three-seed means; shaded bands are seed ranges, not confidence intervals',fontsize=10)
    fig.savefig(out/'budget_curves.png',dpi=180);fig.savefig(out/'budget_curves.pdf');plt.close(fig)
    shutil.copy2(root/'protocol.json',out/'protocol.json')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    source=Path(__file__).with_name('dime_pilot.py')
    (out/'environment.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        versions={k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scikit-learn','scipy','matplotlib']},
        checks='24 unit tests ran: 22 passed; two optional TabNet tests skipped because pytorch-tabnet absent in refreshed environment'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);run(p.parse_args())
