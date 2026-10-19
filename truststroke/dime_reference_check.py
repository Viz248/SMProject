"""Execute official DIME classes on a small real dataset; not published-score reproduction.

Only optimizer scheduler construction is adapted: torch 2.14 removed verbose=.
Reference code remains unmodified in a separate checkout, pinned below.
"""
import argparse, json, sys, subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss,average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits

REF='06ba70f7bcd54dbd8ccd64bed398dfef90d09da5'

def run(a):
    root=Path(a.reference).resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if commit!=REF:raise ValueError('Reference commit mismatch')
    # Lightning 1.9 references the alias removed by NumPy 2.
    if not hasattr(np,'Inf'):np.Inf=np.inf
    sys.path.insert(0,str(root))
    from dime import CMIEstimator,MaskingPretrainer,MaskLayer
    from pytorch_lightning import Trainer,seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    class ModernScheduler:
        def configure_optimizers(self):
            opt=torch.optim.Adam(self.parameters(),lr=self.lr)
            sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=self.factor,patience=self.patience,min_lr=self.min_lr)
            return {'optimizer':opt,'lr_scheduler':sch, **({'monitor':'Loss Val'} if self.automatic_optimization else {})}
    class CompatiblePretrainer(ModernScheduler,MaskingPretrainer):pass
    class CompatibleDIME(ModernScheduler,CMIEstimator):pass
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='OFFICIAL_CLASSES_REAL_DATA_EXECUTION_CHECK',published_scores_reproduced=False,
       reference_commit=commit,dataset='sklearn Wisconsin diagnostic breast cancer; 1=malignant',
       adaptation='CPU; 64/32 hidden layers; fixed 30-epoch pretraining and joint budgets; max_features=5; original bounded CE value scaling; scheduler verbose argument removed via subclass; NumPy Inf compatibility alias; scoped setuptools 80.9 for Lightning 1.9',
       final_brfss_test_accessed=False,brfss2025_accessed=False,seeds=[])
    (out/'protocol.json').write_text(json.dumps(result,indent=2))
    data=load_breast_cancer();X=data.data;y=1-data.target
    fitting,ev=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=2026)
    tr,va=train_test_split(fitting,test_size=.2,stratify=y[fitting],random_state=2026)
    scale=StandardScaler().fit(X[tr]);z=torch.tensor(scale.transform(X),dtype=torch.float32);target=torch.tensor(y,dtype=torch.long)
    np.savez_compressed(out/'positions.npz',fit=tr,validation=va,evaluation=ev)
    loader=lambda ix,shuffle:DataLoader(TensorDataset(z[ix],target[ix]),batch_size=64,shuffle=shuffle,num_workers=0)
    for seed in [42,43,44]:
        seed_everything(seed,workers=True)
        predictor=nn.Sequential(nn.Linear(60,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        value=nn.Sequential(nn.Linear(60,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,30))
        mask=MaskLayer(30,append=True)
        common=dict(accelerator='cpu',devices=1,max_epochs=30,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,log_every_n_steps=1)
        pre=CompatiblePretrainer(predictor,mask,lr=.001,loss_fn=nn.CrossEntropyLoss(),val_loss_fn=nn.CrossEntropyLoss(),early_stopping_epochs=100)
        Trainer(**common,logger=TensorBoardLogger(str(out),name=f'pre{seed}'),enable_checkpointing=False).fit(pre,loader(tr,True),loader(va,False))
        model=CompatibleDIME(value,predictor,mask,lr=.001,max_features=5,eps=.1,
            loss_fn=nn.CrossEntropyLoss(reduction='none'),val_loss_fn=nn.CrossEntropyLoss(),cmi_scaling='bounded',early_stopping_epochs=100)
        ck=ModelCheckpoint(dirpath=str(out/f'checkpoint{seed}'),monitor='Loss Val/Mean',mode='min',save_top_k=1)
        Trainer(**common,logger=TensorBoardLogger(str(out),name=f'joint{seed}'),callbacks=[ck]).fit(model,loader(tr,True),loader(va,False))
        model.load_state_dict(torch.load(ck.best_model_path,weights_only=False)['state_dict']);model.eval()
        with torch.no_grad():preds,_=model.validation_step((z[ev],target[ev]),0)
        curve=[];saved={}
        for budget,logits in enumerate(preds[:4]):
            p=logits.softmax(1)[:,1].numpy();saved[f'budget{budget}']=p
            curve.append(dict(budget=budget,brier=float(brier_score_loss(y[ev],p)),average_precision=float(average_precision_score(y[ev],p)),auroc=float(roc_auc_score(y[ev],p))))
        np.savez_compressed(out/f'seed{seed}_predictions.npz',labels=y[ev],**saved)
        result['seeds'].append(dict(seed=seed,curve=curve,selected_checkpoint=ck.best_model_path,validation_selection_loss=float(ck.best_model_score)))
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print('official DIME',seed,curve[-1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(p.parse_args())
