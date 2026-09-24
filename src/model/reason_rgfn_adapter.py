"""One-hour RGFN head on the same sensor-scoped features as reason HGB heads."""
import copy
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.model.hourly_rgfn import HourlyReliabilityAwareGatedFusionNetwork,HourlyRgfnConfig,set_hourly_rgfn_seed


class ReasonRgfnEstimator:
    def __init__(self, context, rules, seed=2026):
        self.context=np.asarray(context,int);self.rules=np.asarray(rules,int);self.seed=seed

    def tensors(self,x):
        z=torch.tensor(self.preprocessor.transform(x),dtype=torch.float32)
        n=len(z)
        return z[:,self.context,None].transpose(1,2),torch.ones(n,1,1),torch.zeros(n,1,1),torch.empty(n,0),z[:,self.rules]

    def fit(self,x,y,weight,valid=None,epochs=None):
        torch.set_num_threads(2);set_hourly_rgfn_seed(self.seed)
        self.preprocessor=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
        self.preprocessor.fit(x)
        self.model=HourlyReliabilityAwareGatedFusionNetwork(encoder='mlp',config=HourlyRgfnConfig(
            n_continuous=len(self.context),n_static=0,n_rule_evidence=len(self.rules),window_hours=1,
            sensor_hidden_size=48,evidence_hidden_size=16,evidence_embed_size=16,fusion_hidden_size=8,dropout=.3))
        tensors=self.tensors(x);target=torch.tensor(y,dtype=torch.float32);weights=torch.tensor(weight,dtype=torch.float32)
        if valid is not None:
            vx,vy,vw=valid;vt=self.tensors(vx);vy=torch.tensor(vy,dtype=torch.float32);vw=torch.tensor(vw,dtype=torch.float32)
        opt=torch.optim.Adam(self.model.parameters(),lr=.001,weight_decay=.0001)
        lossfn=torch.nn.BCEWithLogitsLoss(reduction='none')
        best=float('inf');state=None;stale=0;self.best_epoch=0
        for epoch in range(1,(epochs or 80)+1):
            self.model.train();order=torch.randperm(len(x))
            for ii in order.split(1024):
                opt.zero_grad();output=self.model(*(v[ii] for v in tensors))
                loss=(lossfn(output['final_fault_logit'],target[ii])*weights[ii]).mean()
                loss.backward();opt.step()
            if valid is None:self.best_epoch=epoch;continue
            self.model.eval();total=0.
            with torch.no_grad():
                for ii in torch.arange(len(vy)).split(4096):
                    total+=float((lossfn(self.model(*(v[ii] for v in vt))['final_fault_logit'],vy[ii])*vw[ii]).sum())
            value=total/float(vw.sum())
            if value<best-1e-4:
                best=value;state=copy.deepcopy(self.model.state_dict());self.best_epoch=epoch;stale=0
            else:stale+=1
            if epochs is None and stale>=8:break
        if state is not None:self.model.load_state_dict(state)
        self.model.eval();return self

    def predict_proba(self,x):
        self.model.eval();tensors=self.tensors(x);parts=[]
        with torch.no_grad():
            for ii in torch.arange(len(x)).split(4096):
                parts.append(self.model(*(v[ii] for v in tensors))['binary_prob'].numpy())
        p=np.concatenate(parts);return np.column_stack([1-p,p])
