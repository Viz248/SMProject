"""Exploratory matched-predictor DIME target-estimation follow-up.

Executes original DIME classes, then freezes a shared Brier-trained predictor
for in-sample versus cross-fitted value-head comparisons. Not published-score
replication and not untouched confirmation. Results/checkpoints never overwrite.
"""
import argparse,hashlib,json,sys,time,subprocess,copy,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
import joblib
from .baseline import clean,FEATURES,CATEGORICAL
from .optimize import load_data,partitions
from .dime_pilot import Value,masks,targets,evaluate,paired_interval,state,network

REF='06ba70f7bcd54dbd8ccd64bed398dfef90d09da5'


class BinaryBrier(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__();self.reduction=reduction
    def forward(self,logits,y):
        loss=(logits.softmax(1)[:,1]-y.float())**2
        return loss if self.reduction=='none' else loss.mean()


class EncodedPredictor:
    def __init__(self,X,categorical):
        self.d=X.shape[1];blocks=[]
        for j,c in enumerate(X.columns):
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False) if c in categorical else StandardScaler()
            blocks.append((str(j),enc,[c]))
        self.prep=ColumnTransformer(blocks);z=self.prep.fit_transform(X).astype(np.float32)
        groups=[]
        for j,(_,enc,cols) in enumerate(self.prep.transformers_):groups.extend([j]*enc.transform(X[cols].iloc[:1]).shape[1])
        self.groups=torch.tensor(groups,dtype=torch.long)
        self.net=nn.Sequential(nn.Linear(z.shape[1]+self.d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def predict(self,X,m):
        z=torch.tensor(self.prep.transform(X).astype(np.float32))
        with torch.no_grad():return self.net(state(z,torch.tensor(m),self.groups)).softmax(1)[:,1].numpy()


def reference_classes(path):
    root=Path(path).resolve();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if commit!=REF:raise ValueError('Reference commit mismatch')
    if not hasattr(np,'Inf'):np.Inf=np.inf
    sys.path.insert(0,str(root))
    from dime import CMIEstimator,MaskingPretrainer,MaskLayerGrouped
    from pytorch_lightning import Trainer,seed_everything
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    class ModernScheduler:
        def configure_optimizers(self):
            opt=torch.optim.Adam(self.parameters(),lr=self.lr)
            sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=self.factor,patience=self.patience,min_lr=self.min_lr)
            return {'optimizer':opt,'lr_scheduler':sch,**({'monitor':'Loss Val'} if self.automatic_optimization else {})}
    class Pretrainer(ModernScheduler,MaskingPretrainer):pass
    class DIME(ModernScheduler,CMIEstimator):pass
    return Pretrainer,DIME,MaskLayerGrouped,Trainer,seed_everything,TensorBoardLogger,ModelCheckpoint


def train_reference(X,y,V,vy,categorical,seed,kind,out,a,classes):
    Pre,DIME,Grouped,Trainer,seed_everything,Logger,Checkpoint=classes
    seed_everything(seed,workers=True);wrapper=EncodedPredictor(X,categorical)
    value=network(wrapper.net[0].in_features,X.shape[1])
    group_matrix=nn.functional.one_hot(wrapper.groups,num_classes=wrapper.d).T.float()
    mask=Grouped(group_matrix,append=True)
    x=torch.tensor(wrapper.prep.transform(X).astype(np.float32));v=torch.tensor(wrapper.prep.transform(V).astype(np.float32))
    loader=lambda z,t,shuffle:DataLoader(TensorDataset(z,torch.tensor(t,dtype=torch.long)),batch_size=256,shuffle=shuffle,num_workers=0)
    train=loader(x,y,True);val=loader(v,vy,False)
    loss=nn.CrossEntropyLoss() if kind=='ce' else BinaryBrier()
    individual=nn.CrossEntropyLoss(reduction='none') if kind=='ce' else BinaryBrier('none')
    common=dict(accelerator='cpu',devices=1,max_epochs=a.reference_epochs,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,log_every_n_steps=1)
    pre=Pre(wrapper.net,mask,lr=.001,loss_fn=loss,val_loss_fn=loss,early_stopping_epochs=100)
    Trainer(**common,logger=Logger(str(out),name='pre'),enable_checkpointing=False).fit(pre,train,val)
    model=DIME(value,wrapper.net,mask,lr=.001,max_features=5,eps=.1,loss_fn=individual,val_loss_fn=loss,
         cmi_scaling='bounded' if kind=='ce' else 'none',early_stopping_epochs=100)
    checkpoint=Checkpoint(dirpath=str(out/'checkpoints'),monitor='Loss Val/Mean',mode='min',save_top_k=1)
    Trainer(**common,logger=Logger(str(out),name='joint'),callbacks=[checkpoint]).fit(model,train,val)
    model.load_state_dict(torch.load(checkpoint.best_model_path,weights_only=False)['state_dict']);model.eval();wrapper.net.eval()
    joblib.dump(wrapper.prep,out/'preprocessor.joblib')
    torch.save({'predictor':wrapper.net.state_dict(),'value':value.state_dict(),'groups':wrapper.groups},out/'selected_state.pt')
    return wrapper,model


class NativeValue:
    def __init__(self,predictor,model):self.predictor=predictor;self.model=model
    def predict(self,X,m):
        z=torch.tensor(self.predictor.prep.transform(X).astype(np.float32))
        with torch.no_grad():
            inp=self.model.mask_layer(z,torch.tensor(m));v=self.model.value_network(inp)
            if self.model.cmi_scaling=='bounded':
                logp=self.model.predictor(inp).log_softmax(1)
                entropy=-(logp.exp()*logp).sum(1,keepdim=True);v=v.sigmoid()*entropy
            return v.numpy()


def common_state_diagnostics(full,policies,X,y,initial):
    m=masks(len(X),X.shape[1],initial,np.random.default_rng(9081))
    realized,_=targets(full,X,y,m,'brier');est={k:p.predict(X,m) for k,p in policies.items()}
    result={}
    for key,v in est.items():
        scores=-v if key=='cf_postloss' else v
        scores=scores.copy();scores[m==1]=-np.inf;j=scores.argmax(1)
        if key=='cf_postloss':continue
        pred=v[m==0];truth=realized[m==0];bins=[]
        for ix in np.array_split(np.argsort(pred),5):bins.append(dict(n=len(ix),predicted=float(pred[ix].mean()),realized=float(truth[ix].mean())))
        result[key]=dict(mse=float(np.mean((pred-truth)**2)),signed_bias=float(np.mean(pred-truth)),bins=bins,
             selected_predicted_benefit=float(v[np.arange(len(X)),j].mean()),
             selected_realized_benefit=float(realized[np.arange(len(X)),j].mean()))
    v=est['cf_brier'].copy();q=-est['cf_postloss'].copy();v[m==1]=-np.inf;q[m==1]=-np.inf
    result['same_state_delta_post_action_agreement']=float(np.mean(v.argmax(1)==q.argmax(1)))
    return result


def run_dataset(name,X,y,cat,fit,stop,ev,initial,out,a,classes):
    train=X.iloc[fit].reset_index(drop=True);V=X.iloc[stop].reset_index(drop=True);E=X.iloc[ev].reset_index(drop=True)
    yt=y[fit];vy=y[stop];ye=y[ev]
    if set(fit)&set(stop) or set(fit)&set(ev) or set(stop)&set(ev):raise ValueError('Overlapping partitions')
    np.savez_compressed(out/f'{name}_positions.npz',fit=fit,stopping=stop,evaluation=ev)
    data=dict(dataset=name,fit_records=len(fit),stopping_records=len(stop),evaluation_records=len(ev),
        positives=dict(fit=int(yt.sum()),stopping=int(vy.sum()),evaluation=int(ye.sum())),features=list(X.columns),seeds=[])
    for seed in a.seeds:
        began=time.time();seedout=out/f'{name}_seed{seed}';seedout.mkdir()
        full,brier=train_reference(train,yt,V,vy,cat,seed,'brier',seedout/'native_brier',a,classes)
        ce_predictor,ce=train_reference(train,yt,V,vy,cat,seed,'ce',seedout/'native_ce',a,classes)
        owner=np.tile(np.arange(len(train)),3);episodes=train.iloc[owner].reset_index(drop=True);ey=yt[owner]
        m=masks(len(owner),X.shape[1],initial,np.random.default_rng(seed))
        inside,_=targets(full,episodes,ey,m,'brier');cf=np.zeros_like(inside);post=np.zeros_like(inside);matched=np.zeros_like(inside)
        folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(train,yt));teachers=[]
        for k,(tr,held) in enumerate(folds):
            teacher,_=train_reference(train.iloc[tr],yt[tr],V,vy,cat,seed+k+1,'brier',seedout/f'teacher{k}',a,classes)
            teachers.append(teacher);ix=np.flatnonzero(np.isin(owner,held))
            assert not set(owner[ix])&set(tr)
            cf[ix],post[ix]=targets(teacher,episodes.iloc[ix],ey[ix],m[ix],'brier')
        for k,(_,held) in enumerate(folds):
            ix=np.flatnonzero(np.isin(owner,held));assert set(owner[ix])<=set(folds[(k+1)%3][0])
            matched[ix],_=targets(teachers[(k+1)%3],episodes.iloc[ix],ey[ix],m[ix],'brier')
        policies={k:Value(full,episodes,m,t,seed,a.value_epochs) for k,t in {'insample_brier':inside,'cf_brier':cf,'cf_postloss':post,'matched_size_insample':matched}.items()}
        policies['native_brier']=NativeValue(full,brier)
        rows={};preds={};actions={}
        for key,policy in list(policies.items())+[('random','random'),('static','static')]:
            rows[key],preds[key],actions[key]=evaluate(full,policy,E,ye,initial,seed,static=inside.mean(0),post=key=='cf_postloss')
        rows['native_ce'],preds['native_ce'],actions['native_ce']=evaluate(ce_predictor,NativeValue(ce_predictor,ce),E,ye,initial,seed)
        for key,policy in policies.items():
            if isinstance(policy,Value):torch.save({'state_dict':policy.net.state_dict(),'scale':policy.scale},seedout/f'{key}.pt')
        diff=(ye-preds['insample_brier'][3])**2-(ye-preds['cf_brier'][3])**2
        diagnostics=common_state_diagnostics(full,policies,E,ye,initial)
        row=dict(seed=seed,curves=rows,primary=dict(benefit_cf=float(diff.mean()),conditional_record_bootstrap_ci=paired_interval(diff,seed)),
             common_state=diagnostics,seconds=time.time()-began)
        data['seeds'].append(row)
        np.savez_compressed(seedout/'predictions.npz',labels=ye,**{k:np.array(v) for k,v in preds.items()})
        (out/f'{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False))
        print('MATCHED',name,seed,row['primary'],round(row['seconds'],1),flush=True)


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='EXPLORATORY_MATCHED_REFERENCE_FOLLOWUP',reference_commit=REF,
       primary='Brier benefit cf_brier versus separately refitted insample_brier head after 3 acquisitions, same frozen full Brier DIME predictor',
       seeds=a.seeds,pretrain_epochs=a.reference_epochs,joint_epochs=a.reference_epochs,value_epochs=a.value_epochs,
       final_test_accessed=False,brfss2025_accessed=False,
       notes=['All evaluation records are development records, not fresh confirmation','CE native uses bounded scaling; Brier native uses unconstrained scaling because entropy bound is CE-specific','Native classes use original repeated-action training sampler','Each fold fits preprocessing and teacher only on its own training records; same separate stopping partition','CE and Brier reference checkpoints selected by their own validation losses','No external probability calibration; native CE uses separate predictor, other policy comparisons share full Brier predictor','CF requires three extra teacher fits; compute not equal','Fold teachers have less data than deployment predictor; same-size in-sample control included','Record bootstrap conditional on models; no design-based or training-uncertainty CI','Wisconsin is auxiliary tabular diagnostic data, not a published DIME benchmark'],
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','pytorch-lightning','torchmetrics']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    classes=reference_classes(a.reference)
    bc=load_breast_cancer(as_frame=True);X=bc.data;y=1-np.asarray(bc.target)
    fitting,ev=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=2026)
    fit,stop=train_test_split(fitting,test_size=.2,stratify=y[fitting],random_state=2026)
    run_dataset('wisconsin',X,y,[],fit,stop,ev,[],out,a,classes)
    raw,digest=load_data(a.data);X,y,_=clean(raw);y=np.asarray(y);s=partitions(y,42)
    complete=~X.isna().any(axis=1).to_numpy();rng=np.random.default_rng(2026)
    fit=rng.choice(s['fit'][complete[s['fit']]],6000,replace=False)
    ev=rng.choice(s['select'][complete[s['select']]],1500,replace=False)
    stop=rng.choice(s['stop'][complete[s['stop']]],1500,replace=False)
    assert not (set(fit)|set(stop)|set(ev))&set(s['test'])
    protocol['brfss_sha256']=digest;(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run_dataset('brfss2024',X,y,CATEGORICAL,fit,stop,ev,[0,1],out,a,classes)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--reference-epochs',type=int,default=15);p.add_argument('--value-epochs',type=int,default=25)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(p.parse_args())
