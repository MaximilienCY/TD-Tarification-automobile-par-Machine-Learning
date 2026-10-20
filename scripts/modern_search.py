"""XGBoost/Explainable Boosting comparison using the advanced grouped search engine."""
import sys
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import advanced_search as engine
from pricing.models import tree_frame
from pricing.reference import NUM_COMPACT, LOW_CATS


class ModernCandidate:
    def __init__(self,config,frequency):
        self.c,self.frequency=config,frequency

    def frame(self,X):
        x=tree_frame(X,self.c.get('compact',False))
        if self.c['kind']=='ebm' and self.frequency:
            x=x[NUM_COMPACT+LOW_CATS+['departement','marque_vehicule','modele_vehicule']]
        return x

    def fit(self,X,y):
        c=self.c;x=self.frame(X)
        self.scale_=1. if self.frequency else float(np.mean(y))
        if c['kind']=='ebm':
            cls=ExplainableBoostingClassifier if self.frequency else ExplainableBoostingRegressor
            # Fixed rounds with no internal random row split: grouping is controlled
            # entirely by the external client-separated folds.
            self.model_=cls(interactions=c['interactions'],max_bins=64,max_interaction_bins=16,
                outer_bags=1,validation_size=0,early_stopping_rounds=0,
                max_rounds=c['rounds'],learning_rate=.02,min_samples_leaf=50,
                min_cat_samples=100,cat_smooth=100,reg_lambda=20,
                objective='log_loss' if self.frequency else c['objective'],n_jobs=4,random_state=42)
        else:
            self.categories_={col:sorted(x[col].unique()) for col in x.select_dtypes(exclude='number')}
            x=self.encode(x)
            cls=XGBClassifier if self.frequency else XGBRegressor
            self.model_=cls(n_estimators=c['rounds'],max_depth=c['depth'],learning_rate=.025,
                min_child_weight=30 if self.frequency else 80,reg_lambda=30,
                subsample=.85,colsample_bytree=.85,enable_categorical=True,
                tree_method='hist',device='cpu',n_jobs=4,random_state=42,
                objective='binary:logistic' if self.frequency else c['objective'])
        self.model_.fit(x,np.asarray(y)/self.scale_)
        return self

    def encode(self,x):
        x=x.copy()
        for col,cats in self.categories_.items():
            x[col]=pd.Categorical(x[col].where(x[col].isin(cats)),categories=cats)
        return x

    def predict(self,X):
        x=self.frame(X)
        if self.c['kind']=='xgb':x=self.encode(x)
        if self.frequency:return self.model_.predict_proba(x)[:,1]
        return np.maximum(self.model_.predict(x)*self.scale_,1e-8)


def catalogue():
    fs=[];gs=[]
    for interactions,rounds in [(0,500),(0,1500),(10,700)]:
        fs.append(dict(name=f'ebm_i{interactions}_n{rounds}',kind='ebm',interactions=interactions,rounds=rounds))
    for depth in [2,3,5]:
        fs.append(dict(name=f'xgb_d{depth}',kind='xgb',depth=depth,rounds=650))
    for interactions in [0,5]:
        gs.append(dict(name=f'ebm_poisson_i{interactions}',kind='ebm',interactions=interactions,rounds=250,objective='poisson_deviance',compact=True))
    for depth in [1,2,3]:
        gs.append(dict(name=f'xgb_gamma_d{depth}',kind='xgb',depth=depth,rounds=200,objective='reg:gamma',compact=True))
    signature=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for c in fs+gs:
        c['implementation_sha256']=signature
        c['library_versions']={k:importlib.metadata.version(k) for k in ['interpret-core','xgboost']}
    return fs,gs


if __name__=='__main__':
    engine.Candidate=ModernCandidate
    engine.catalogue=catalogue
    engine.OUT=engine.ROOT/'reports'/'modern'
    engine.CACHE=engine.ROOT/'.cache'/'modern'
    # The experiment engine exports to separate names so earlier candidates survive.
    if sys.argv[1]=='export':
        raise SystemExit('Use scripts/export_modern.py after confirmation.')
    with engine.threadpool_limits(limits=4):
        {'search':engine.search,'confirm':engine.confirm}[sys.argv[1]]()
