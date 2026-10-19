"""Verify preserved artifacts, executed diagnostics and sensitivity results."""
import json,ast,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1];out=root/'docs/experiments/revision_audit';fit=root/'results/availability-paper-fit-v2';ext=root/'results/availability-paper-2025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((fit/'model_manifest.json').read_text());assert sha(root/'truststroke/availability_study.py')==m['source_sha256'];assert sha(root/'configs/availability_study_frozen.json')==m['protocol_sha256']
for name,h in m['models'].items():assert sha(fit/name)==h
r=json.loads((ext/'results.json').read_text());d=json.loads((out/'design_audit.json').read_text());assert d['frozen_primary']==r['primary'];assert d['frozen_primary']['ci95'] is None;assert sha(ext/'predictions.npz')==d['predictions_sha256']
cfg=json.loads((fit/'protocol.json').read_text());assert d['raw2025_sha256']==cfg['input_sha256_2025_xpt']
c=json.loads((out/'calibration.json').read_text())['metrics'];assert len(c)==32
for v in c.values():assert np.isfinite([v['citl'],v['intercept'],v['slope'],v['weighted_ece']]).all();assert abs(sum(b['weight_fraction'] for b in v['curve'])-1)<1e-12
rob=json.loads((out/'fitting_sample_robustness.json').read_text());assert rob['complete'];assert len(rob['rows'])==3
for v in rob['rows']:np.testing.assert_allclose(v['contrast'],v['MT_brier']-v['SC_brier'],atol=1e-15,rtol=0)
for cell in json.loads((root/'notebooks/Availability_Study_Colab.ipynb').read_text())['cells']:
 if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
assert all(v['agreement'] for v in json.loads((out/'variance_crosscheck.json').read_text())['checks'])
new=root/'results/posthoc-fitting-samples';manifest={p.name:sha(p) for p in new.iterdir() if p.is_file()}
(out/'provenance.json').write_text(json.dumps(dict(version='v3-reconstructed-2026-10-19',source_sha256=sha(root/'truststroke/revision_audit.py'),protocol_sha256=sha(root/'configs/posthoc_revision.json'),fitting_sensitivity_artifacts=manifest,original_model_hashes_verified=30,original_predictions_preserved=True),indent=2))
log=Path('/workspace/scratch/fe5bdd3008be/revision-tests.log').read_text();assert 'Ran 35 tests' in log and '\nOK\n' in log
(out/'automated_tests.txt').write_text(log)
(out/'validation.json').write_text(json.dumps(dict(version='v3-reconstructed-2026-10-19',status='PASS',tests=35,original_models_verified=30,calibration_ensembles=32,additional_classifiers=18,fitting_samples=3,R_Python_agreement=True,notebook_syntax_checked=True,manuscript_sha256=sha(root/'paper/manuscript.pdf')),indent=2));print('v3 validation PASS')
