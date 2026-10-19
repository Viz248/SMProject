"""Generate figures from saved aggregate metrics without refitting or reading raw records."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate(directory):
    root=Path(directory)
    development=json.loads((root/'development.json').read_text())
    labels=[r['specification']['name'] for r in development['candidates']]
    values=[r['selection']['average_precision'] for r in development['candidates']]
    stage=development.get('ensemble_stage',{})
    for row in stage.get('candidates',[]):
        labels.append(f"blend (neural {row['neural_weight']:.0%})")
        values.append(row['selection']['average_precision'])
    order=np.argsort(values)
    fig,ax=plt.subplots(figsize=(9,6))
    colors=['#4770a6' if 'mlp' not in labels[i] else '#26836c' for i in order]
    ax.barh(np.asarray(labels)[order],np.asarray(values)[order],color=colors)
    ax.axvline(development['candidates'][0]['selection']['prevalence'],color='#9d453e',linestyle='--',label='Selection prevalence (no-skill AP reference)')
    ax.set(xlabel='Selection average precision (higher is better)',title='BRFSS 2024: exploratory development search')
    ax.legend(loc='lower right',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(root/f'selection_comparison.{suffix}',dpi=200)
    plt.close(fig)
    if (root/'locked_test.json').exists():
        final=json.loads((root/'locked_test.json').read_text())
        bins=final['test']['calibration']
        fig,ax=plt.subplots(figsize=(5,5))
        ax.plot([0,1],[0,1],linestyle='--',color='gray',label='Ideal')
        ax.scatter([b['probability'] for b in bins],[b['observed'] for b in bins],
                   s=[15+np.sqrt(b['count']) for b in bins],color='#26836c',label='Equal-width bins')
        ax.set(xlabel='Mean predicted probability',ylabel='Observed stroke-history fraction',
               title='Selected model: internal test calibration',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ['png','pdf']:
            fig.savefig(root/f'calibration.{suffix}',dpi=200)
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    generate(parser.parse_args().run)
