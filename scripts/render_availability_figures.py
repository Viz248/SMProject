"""Render cached scientific aggregates with embedded TrueType plot fonts."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
root=Path(__file__).resolve().parent.parent
out=root/'docs/experiments/availability_paper'
r=json.loads((out/'transport2025.json').read_text());cfg=json.loads((out/'protocol.json').read_text())
short={'fixed_global_calibration':'Global calibration','fixed_scenario_calibration':'Scenario calibration','mask_trained':'Mask training','subset_retrained':'Subset refit'}
markers=['o','s','^','D']
fig,axes=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
for ax,family in zip(axes,cfg['families']):
 for condition,marker in zip(cfg['conditions'],markers):
  value=[r['common_metrics'][f'{family}|{s}|{condition}']['weighted']['brier'] for s in cfg['scenarios']]
  ax.plot(range(4),value,marker=marker,label=short[condition],linewidth=1.5,markersize=4)
 ax.set(xticks=range(4),xticklabels=['Full','Without health\nstate','Demographics\n+ histories','Demographics\nonly'],xlabel='Predefined questionnaire scenario',ylabel='Survey-weighted Brier loss',title='Boosted trees' if family=='hist_gradient_boosting' else 'MLP');ax.legend(fontsize=7)
fig.savefig(out/'availability.pdf');fig.savefig(out/'availability.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(4.5,3.5),layout='constrained');maximum=0
for condition,marker in zip(cfg['conditions'],markers):
 rows=json.loads((root/'docs/experiments/revision_audit/calibration.json').read_text())['metrics'][f'hist_gradient_boosting|demographics_history|{condition}']['curve']
 p=[v['probability'] for v in rows];q=[v['observed'] for v in rows]
 if p:maximum=max(maximum,max(p),max(q));ax.plot(p,q,marker=marker,label=short[condition],markersize=4)
lim=min(1,max(.2,maximum*1.1));ax.plot([0,lim],[0,lim],ls='--',color='grey',lw=1)
ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Survey-weighted mean prediction',ylabel='Survey-weighted stroke-history proportion',title='2025: demographics + history');ax.legend(fontsize=7)
fig.savefig(out/'calibration.pdf');fig.savefig(out/'calibration.png',dpi=200);plt.close(fig)
groups=r['subgroups'];fig,ax=plt.subplots(figsize=(5,3.5),layout='constrained')
for i,g in enumerate(groups):ax.plot(g['primary']['estimate'],i,'o',color='tab:blue')
labels={'sex_male':'Male','sex_female':'Female','age18_44':'Age 18-44','age45_64':'Age 45-64','age65plus':'Age 65+','income_under35k':'Income <35k','income35kplus':'Income >=35k'}
ax.axvline(0,color='grey',ls='--');ax.set(yticks=range(len(groups)),yticklabels=[labels[g['group']] for g in groups],xlabel='Brier difference (MT - SC)',title='Domain point contrasts');ax.invert_yaxis();ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
fig.savefig(out/'subgroups.pdf');fig.savefig(out/'subgroups.png',dpi=200);plt.close(fig)
