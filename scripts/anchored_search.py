"""Small frequency corrections anchored to the original logistic x Gamma model.

The correction learns from cross-fitted logistic margins within each training
partition. Severity is always the original Gamma(alpha=.1), fitted on claims.
No leaderboard scores or historical labels enter candidate selection.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import importlib.metadata
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from pricing.reference import ReferenceFrequence,GammaMetier,preparer_metier,NUM_COMPACT,LOW_CATS
from pricing.experiment import load_data,metrics,write_json,provenance,bootstrap_gain
OUT=ROOT/'reports/anchored'
CACHE=ROOT/'.cache/anchored'
SEVERITY={'alpha':.1}


def configs():
    return [dict(depth=d,trees=n) for d in [1,2,3] for n in [80,240]]


def frame(X):
    x=preparer_metier(X)[NUM_COMPACT+LOW_CATS].copy()
    for c in LOW_CATS:x[c]=x[c].fillna('__MISSING__').astype(str)
    return x.replace([np.inf,-np.inf],np.nan)


class Anchored:
    def __init__(self,config,weight):self.config,self.weight=config,weight

    def fit(self,X):
        y=X.nombre_sinistres
        self.base_=ReferenceFrequence().fit(X,y,X.id_client)
        claims=X.loc[y.eq(1)]
        self.severity_=GammaMetier(SEVERITY).fit(claims,claims.montant_sinistre,claims.id_client)
        if not self.weight:return self
        # Each training margin comes from a model that excluded that client.
        margins=np.zeros(len(X))
        cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=18371)
        for a,b in cv.split(X,y,X.id_client):
            assert set(X.iloc[a].id_client).isdisjoint(X.iloc[b].id_client)
            m=ReferenceFrequence().fit(X.iloc[a],y.iloc[a],X.id_client.iloc[a])
            margins[b]=logit(np.clip(m.predict_proba(X.iloc[b])[:,1],1e-8,1-1e-8))
        self.fit_correction(X,margins)
        return self

    def encode(self,x):
        x=x.copy()
        for c,values in self.categories_.items():x[c]=pd.Categorical(x[c].where(x[c].isin(values)),categories=values)
        return x

    def fit_correction(self,X,margins):
        x=frame(X);self.categories_={c:sorted(x[c].unique()) for c in LOW_CATS}
        self.correction_=XGBClassifier(objective='binary:logistic',n_estimators=self.config['trees'],
            max_depth=self.config['depth'],learning_rate=.025,min_child_weight=100,
            reg_lambda=100,reg_alpha=1.,subsample=1.,colsample_bytree=1.,
            tree_method='hist',enable_categorical=True,n_jobs=4,random_state=42)
        self.correction_.fit(self.encode(x),X.nombre_sinistres,base_margin=margins)
        return self

    def predict_details(self,X):
        p=self.base_.predict_proba(X)[:,1]
        if self.weight:
            delta=self.correction_.predict(self.encode(frame(X)),output_margin=True,base_margin=np.zeros(len(X)))
            p=expit(logit(np.clip(p,1e-8,1-1e-8))+self.weight*delta)
        s=self.severity_.predict(X)
        assert np.isfinite(p*s).all() and ((p>=0)&(p<=1)).all() and (s>0).all()
        return pd.DataFrame({'index':X['index'].to_numpy(),'proba_pred':p,'montant_pred':s,'pred':p*s})


def fold_predictions(learn,evaluation):
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(json.dumps(provenance(),sort_keys=True).encode());h.update(importlib.metadata.version('xgboost').encode())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    path=CACHE/(h.hexdigest()+'.npz')
    if path.exists():
        z=np.load(path);return z['p'],z['s'],z['delta']
    base=Anchored(None,0.).fit(learn)
    # Share identical cross-fitted margins among the six prespecified corrections.
    margins=np.zeros(len(learn));y=learn.nombre_sinistres
    for a,b in StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=18371).split(learn,y,learn.id_client):
        m=ReferenceFrequence().fit(learn.iloc[a],y.iloc[a],learn.id_client.iloc[a])
        margins[b]=logit(np.clip(m.predict_proba(learn.iloc[b])[:,1],1e-8,1-1e-8))
    p=base.base_.predict_proba(evaluation)[:,1];s=base.severity_.predict(evaluation)
    delta=[]
    for c in configs():
        base.config=c;base.fit_correction(learn,margins)
        delta.append(base.correction_.predict(base.encode(frame(evaluation)),output_margin=True,base_margin=np.zeros(len(evaluation))))
    delta=np.column_stack(delta);CACHE.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=p,s=s,delta=delta)
    return p,s,delta


def choices():
    return [dict(config=None,weight=0.)]+[dict(config=c,weight=w) for c in configs() for w in [.1,.25,.5,1.]]


def predictions(p,s,delta):
    z=logit(np.clip(p,1e-8,1-1e-8))
    return np.column_stack([p*s]+[expit(z+w*delta[:,j])*s for j in range(len(configs())) for w in [.1,.25,.5,1.]])


def oof(dev,seed):
    pred=np.zeros((len(dev),len(choices())));folds=np.zeros(len(dev),dtype=int)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for k,(a,b) in enumerate(cv.split(dev,dev.nombre_sinistres,dev.id_client)):
        assert set(dev.iloc[a].id_client).isdisjoint(dev.iloc[b].id_client)
        p,s,delta=fold_predictions(dev.iloc[a],dev.iloc[b]);pred[b]=predictions(p,s,delta);folds[b]=k
        print(f'seed {seed} fold {k+1}/5 complete',flush=True)
    return pred,folds


def main():
    train,test,dev,hist=load_data();OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    first,folds=oof(dev,20261009)
    y=dev.montant_sinistre.to_numpy();rmse=np.sqrt(np.mean((first-y[:,None])**2,axis=0))
    ranking=pd.DataFrame([dict(candidate=j,**choices()[j],RMSE=float(rmse[j])) for j in range(len(rmse))]).sort_values('RMSE')
    ranking.to_csv(OUT/'ranking.csv',index=False)
    winner=int(np.argmin(rmse))
    # Winner fixed before the second split; do not switch to its best-scoring candidate.
    second,second_folds=oof(dev,74191)
    report=dict(selected_candidate=winner,choice=choices()[winner],search_RMSE=float(rmse[winner]),
        reference_search_RMSE=float(rmse[0]),confirmation=metrics(y,second[:,winner]),reference_confirmation=metrics(y,second[:,0]),
        gain=bootstrap_gain(dev,second[:,winner],second[:,0]),provenance=provenance(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat='Both splits reuse previously explored development rows; confirmation is a sensitivity check, not a fresh test.')
    np.savez_compressed(CACHE/'oof.npz',first=first,second=second,fold=folds,second_fold=second_folds)
    write_json(OUT/'results.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if winner==0 or report['confirmation']['RMSE']>=report['reference_confirmation']['RMSE']:
        print('No supported challenger; original reference retained.',flush=True);return
    ch=choices()[winner];m=Anchored(ch['config'],ch['weight']).fit(train)
    d=m.predict_details(test);sample=pd.read_csv(ROOT/'Data/prime_pred_sandbox.csv')
    assert np.array_equal(d['index'],sample['index'])
    path=ROOT/'deliverables/candidates/submission_anchored.csv';d[['index','pred']].to_csv(path,index=False)
    joblib.dump(m,ROOT/'models/anchored.joblib')
    np.testing.assert_array_equal(joblib.load(ROOT/'models/anchored.joblib').predict_details(test.iloc[:50])['pred'],d['pred'].iloc[:50])
    h=Anchored(ch['config'],ch['weight']).fit(dev)
    report['historical_diagnostic']=metrics(hist.montant_sinistre.to_numpy(),h.predict_details(hist)['pred'].to_numpy())
    report['export']=dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kaggle_score=None)
    write_json(OUT/'results.json',report)
    print('Exported',path,flush=True)


if __name__=='__main__':
    # Stable module name keeps saved joblib objects portable to another process.
    from anchored_search import main as run
    with threadpool_limits(limits=4):run()
