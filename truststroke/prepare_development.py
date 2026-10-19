"""Create reusable split protocol without fitting models or accessing test predictions."""
import argparse,json
from pathlib import Path
import numpy as np
from .baseline import clean
from .optimize import load_data,partitions

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    raw,digest=load_data(a.data);X,y,_=clean(raw)
    split=partitions(y,a.seed)
    np.savez_compressed(out/'splits.npz',**split)
    (out/'development.json').write_text(json.dumps(dict(status='SPLIT_PROTOCOL_ONLY',
        dataset_sha256=digest,seed=a.seed,labeled_records=len(y),final_test_accessed=False,
        split_counts={k:len(v) for k,v in split.items()}),indent=2))
