"""Plot saved aggregate trust and TabNet development results; no model fitting."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate(trust_run,tabnet_run,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    trust=json.loads((Path(trust_run)/'trust.json').read_text())
    convergence=json.loads((Path(trust_run)/'attribution_convergence.json').read_text())
    tabnet=json.loads((Path(tabnet_run)/'development.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    r=convergence['repeats'];x=[v['permutations'] for v in r]
    m=[v['same_model_repeat']['dsi']['mean'] for v in r]
    low=[v['same_model_repeat']['dsi']['low'] for v in r];high=[v['same_model_repeat']['dsi']['high'] for v in r]
    axes[0,0].errorbar(x,m,yerr=[np.array(m)-low,np.array(high)-m],marker='o',capsize=4,color='#287d73')
    axes[0,0].set(xscale='log',xticks=x,xticklabels=x,ylim=(0,1),xlabel='Attribution permutations',
        ylabel='Same-model repeat DSI',title='Explanation sampling noise control')
    axes[0,0].axhline(convergence['seed_retraining_256']['dsi']['mean'],ls='--',color='#536a9e',label='Seed-retraining DSI (256)')
    axes[0,0].legend(fontsize=8)
    rai=trust['rai'];rows=rai['scenarios'];labels=[r['scenario'].replace('_',' ') for r in rows]
    axes[0,1].barh(labels,[r['metrics']['brier'] for r in rows],color='#536a9e')
    axes[0,1].axvline(rai['full']['brier'],color='#287d73',label='Full inputs')
    axes[0,1].axvline(rai['null_brier'],color='#aa5950',ls='--',label='Constant prevalence')
    axes[0,1].set(xlabel='Development Brier loss (lower is better)',title='Fixed-model availability scenarios')
    axes[0,1].tick_params(axis='y',labelsize=8);axes[0,1].legend(fontsize=8)
    for policy,result in trust['acquisition']['policies'].items():
        scores=[r['brier'] for r in result['steps']]
        axes[1,0].plot(range(len(scores)),scores,marker='o',label=policy.replace('_',' '))
    axes[1,0].set(xlabel='Additional questionnaire items acquired',ylabel='Development Brier loss',
                   title=f"Acquisition pilot (n={trust['acquisition']['records']}; no established benefit)")
    axes[1,0].legend(fontsize=8)
    names=[r['configuration']['name'] for r in tabnet['candidates']]
    scores=[r['selection']['average_precision'] for r in tabnet['candidates']]
    axes[1,1].bar(names,scores,color='#287d73')
    reference=json.loads((Path(trust['source_run'])/'development.json').read_text())
    if reference.get('ensemble_stage'):
        reference_ap=max(r['selection']['average_precision'] for r in reference['ensemble_stage']['candidates'])
        axes[1,1].axhline(reference_ap,ls='--',color='#536a9e',label='Prior blend selection AP')
    axes[1,1].set(ylabel='Selection average precision',title='TabNet bounded search (30 epochs)',ylim=(0,.22))
    axes[1,1].tick_params(axis='x',labelsize=8);axes[1,1].legend(fontsize=8)
    for ax in axes.ravel():ax.spines[['top','right']].set_visible(False)
    fig.suptitle('BRFSS 2024: exploratory development findings',fontsize=15)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'trust_overview.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trust-run',required=True);p.add_argument('--tabnet-run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();generate(a.trust_run,a.tabnet_run,a.output)
