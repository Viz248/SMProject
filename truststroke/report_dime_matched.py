"""Create aggregate-only results report for the matched DIME follow-up."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'native_ce':'Native DIME CE','native_brier':'Native DIME Brier','insample_brier':'Refitted in-sample Brier head','cf_brier':'Cross-fitted Brier head','matched_size_insample':'Same-size in-sample teachers','cf_postloss':'Cross-fitted post-loss head','static':'Static ranking','random':'Random'}


def run(a):
    root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());shutil.copy2(root/'protocol.json',out/'protocol.json')
    data={name:json.loads((root/f'{name}.json').read_text()) for name in ['brfss2024','wisconsin']}
    for name in data:shutil.copy2(root/f'{name}.json',out/f'{name}.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,(name,d) in zip(axes,data.items()):
        for key in ['native_brier','insample_brier','cf_brier','static']:
            values=np.array([[r['brier'] for r in s['curves'][key]] for s in d['seeds']])
            color=ax.plot(range(4),values.mean(0),marker='o',label=LABELS[key])[0].get_color()
            ax.fill_between(range(4),values.min(0),values.max(0),color=color,alpha=.1)
        ax.set(title=name,xlabel='Additional features acquired',ylabel='Brier loss (lower is better)',xticks=range(4));ax.legend(fontsize=8)
    fig.suptitle('Exploratory matched follow-up: three-seed means; shading shows seed ranges, not CIs',fontsize=10)
    fig.savefig(out/'budget_curves.png',dpi=180);fig.savefig(out/'budget_curves.pdf');plt.close(fig)
    summary={name:{key:float(np.mean([s['curves'][key][3]['brier'] for s in d['seeds']])) for key in LABELS} for name,d in data.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Matched DIME follow-up — 19 October 2026','',
      '## Result and decision','',
      'This development follow-up does not support promoting the cross-fitted Brier value head as an improvement. At least one declared seed shows worse Brier with an interval excluding zero. All seed-level outcomes are retained below. This is evidence about this bounded implementation, not proof that cross-fitting cannot help in other settings. Do not spend the untouched 2025 evaluation to confirm this version or describe it as a new acquisition algorithm.','',
      '## What was matched','',
      'Official DIME classes were executed at reference commit `'+protocol['reference_commit']+'`. The original joint predictor/value optimization and training sampler were retained. Compatibility wrappers handle the removed torch scheduler `verbose` argument, NumPy `Inf` alias and scoped setuptools required by Lightning 1.9. The Brier variant uses binary Brier prediction loss and unconstrained value scaling; the CE reference retains entropy-bounded scaling. These changes are explicit adaptations, not published-score reproduction.','',
      'The primary comparison freezes the same full-data Brier-trained DIME predictor and refits either an in-sample or a three-fold out-of-fold value head. Both value heads have the same architecture, masks, episodes, optimizer, epochs, deployment predictor and evaluation records. Additional controls include fold-size-matched in-sample teachers, post-loss regression, native joint value heads, static and random policies. Cross-fitting uses more teacher-fitting compute. Fold teachers train on less data than the deployment model.','',
      'Each fold fits its own preprocessing, excludes every held-out respondent episode, and uses the same separate stopping partition. Reference pretraining and joint training each run '+str(protocol['pretrain_epochs'])+' epochs; checkpoints are selected by validation mean loss. Value heads use '+str(protocol['value_epochs'])+' fixed epochs. No probability recalibration or hyperparameter search was performed.','',
      '| Dataset | Fit | Stopping | Development evaluation | Evaluation positives |','|---|---:|---:|---:|---:|']
    for name,d in data.items():lines.append(f"| {name} | {d['fit_records']} | {d['stopping_records']} | {d['evaluation_records']} | {d['positives']['evaluation']} |")
    lines+=['','BRFSS is a complete-case sample from the existing fit/stop/select partitions. Sex and age are initially observed. Wisconsin starts with no features and uses a stopping subset drawn from its original fitting partition. Its outcome is diagnostic malignancy, not incident disease. Neither is a new confirmatory cohort. BRFSS 2024 final-test, 2023 transport and 2025 records were not evaluated.','',
      '## Brier loss after three acquisitions','',
      'Lower is better. Means across three initialization seeds; not independent-cohort averages. CE uses its own jointly trained predictor; all other rows share the full Brier predictor. Native and offline value heads have different state-training distributions, so their comparison concerns complete policy procedures rather than target estimation alone.','',
      '| Policy | BRFSS 2024 development | Wisconsin diagnostic |','|---|---:|---:|']
    for key,label in LABELS.items():lines.append(f"| {label} | {summary['brfss2024'][key]:.6f} | {summary['wisconsin'][key]:.6f} |")
    lines+=['','## Primary paired comparison','',
      'Benefit = refitted in-sample Brier minus cross-fitted Brier after exactly three acquisitions. Positive favors cross-fitting. Intervals use 500 paired IID record bootstraps, conditional on each fitted model; they do not account for training randomness or BRFSS survey design.','',
      '| Dataset | Seed | Benefit | 95% conditional record interval |','|---|---:|---:|---|']
    for name,d in data.items():
        for s in d['seeds']:
            p=s['primary'];lo,hi=p['conditional_record_bootstrap_ci'];lines.append(f"| {name} | {s['seed']} | {p['benefit_cf']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Identical-state diagnostics','',
      'All policies are queried on exactly the same independently seeded partial-information states. Candidate features already observed are excluded. This resolves the earlier pilot’s trajectory-comparison limitation. Value MSE compares predicted and realized one-step benefits from the same frozen full Brier predictor. It contains irreducible outcome noise, so it is not a pure calibration metric; signed bias, predicted-versus-realized quantile-bin means, and the realized benefit of selected actions are also saved.','',
      '| Dataset | Seed | Same-state CF delta/post-loss action agreement | In-sample value MSE | CF value MSE |','|---|---:|---:|---:|---:|']
    for name,d in data.items():
        for s in d['seeds']:
            c=s['common_state'];lines.append(f"| {name} | {s['seed']} | {c['same_state_delta_post_action_agreement']:.3f} | {c['insample_brier']['mse']:.6f} | {c['cf_brier']['mse']:.6f} |")
    lines+=['','![Budget curves](experiments/dime_matched/budget_curves.png)','','## Practical limits and next gate','',
      'The small diagnostic dataset and 62 stroke-history positives in BRFSS evaluation limit precision. There is no natural-missingness evaluation, survey-design inference, clinical utility validation or fresh-year confirmation. Masked-feature simulation is not evidence of a clinically useful questionnaire: stroke history is directly askable. Reference training begins with no observed features; BRFSS evaluation and offline value-head masks start with sex and age. The reference learns paths up to five acquisitions, whereas offline value heads use random partial states. These protocol differences are disclosed, and only the two refitted value heads form the primary matched comparison.','',
      'Do not infer an algorithmic contribution from a favorable single seed or metric. The cross-fitting improvement claim remains unsupported by these two development pilots. A defensible next research decision is whether an empirical study of action-value generalization has enough explanatory evidence to justify a broader benchmark. Before expanding, require a specific failure mechanism and matched learning/data-size/teacher-alignment controls. If the goal remains clinical stroke-risk prediction, a longitudinal or independently measured endpoint is required; BRFSS cannot supply that evidence.','',
      'Checkpoints, fitted preprocessors, respondent positions, prediction arrays and training logs are saved in the ignored run directory. Aggregate results and source/version protocol are tracked in `docs/experiments/dime_matched/`. The runner refuses output-directory reuse and writes the protocol before fitting.','',
      '## Reproduction','',
      'Install `requirements-dime-pilot.txt` and clone the pinned DIME reference. For Lightning 1.9, put a scoped setuptools 80.9 installation first on PYTHONPATH, as in the earlier execution-check instructions.','',
      '```bash',
      'PYTHONPATH=/path/to/dime-compat OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m truststroke.dime_matched --data /path/to/brfss2024_selected_raw.csv --reference /path/to/DIME --output results/new-matched',
      'python -m truststroke.report_dime_matched --run results/new-matched --output docs/experiments/new-matched',
      '```','',
      'Validation: full unit suite executed with new binary-Brier reduction, gradient and logit-shift checks; see validation record for counts. No new test-set tuning or conference-publication claim.']
    Path(a.report).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);run(p.parse_args())
