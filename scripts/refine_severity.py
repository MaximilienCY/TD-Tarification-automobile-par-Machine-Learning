"""Keep original frequency fixed; evaluate premium-weighted and shifted severities."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor,PoissonRegressor
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
import restart_xgboost as r
from restart_cache import install
from pricing.reference import GammaMetier,LogistiqueMetier,preparer_metier,construire_preparation
OUT=ROOT/'reports/severity_refinement'
CACHE=ROOT/'.cache/severity_refinement'


def catalogue():
    result=[]
    for family,shifts in [('gamma',[0.,300.,600.]),('poisson',[0.,600.])]:
        for shift in shifts:
            for weighted in [False,True]:
                for spline in [False,True]:
                    result.append(dict(kind=family,shift=shift,weighted=weighted,spline=spline,alpha=.1))
    for alpha in [.01,1.]:
        result.append(dict(kind='gamma',shift=0.,weighted=True,spline=False,alpha=alpha))
    for panel in ['basic','compact']:
        for depth in [1,2,3]:
            for trees in [80,240]:
                result.append(dict(kind='xgb_ratio',panel=panel,depth=depth,trees=trees))
    return result


def key(frame,evaluation=None):
    import importlib.metadata
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode())
    h.update(importlib.metadata.version('xgboost').encode())
    for x in [frame,evaluation]:
        if x is not None:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    return h.hexdigest()


def moments(learn):
    """All auxiliary probabilities and baseline severities exclude the row's client."""
    path=CACHE/(key(learn)+'_moments.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['s']
    p,s=np.zeros(len(learn)),np.zeros(len(learn))
    for a,b in r.split_groups(learn,3,19431):
        x,z=learn.iloc[a],learn.iloc[b]
        f=LogistiqueMetier({'C':.1}).fit(x,x.nombre_sinistres,x.id_client)
        p[b]=f.predict_proba(z)[:,1]
        claims=x.loc[x.nombre_sinistres.eq(1)]
        m=GammaMetier({'alpha':.1}).fit(claims,claims.montant_sinistre,claims.id_client)
        s[b]=m.predict(z)
    assert np.isfinite(p*s).all() and ((p>0)&(p<1)).all() and (s>0).all()
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,s=s)
    return p,s


class Severity:
    def __init__(self,config):self.config=config

    def xframe(self,X,fit=False):
        if self.config['panel']=='basic':
            x=preparer_metier(X)[['anciennete_vehicule','type_contrat','utilisation']].copy()
            for c in ['type_contrat','utilisation']:x[c]=x[c].fillna('__MISSING__').astype(str)
        else:x=r.features(X,'compact',True)
        if fit:self.categories_={c:sorted(x[c].unique()) for c in x.select_dtypes(exclude='number')}
        for c,values in self.categories_.items():x[c]=pd.Categorical(x[c].where(x[c].isin(values)),categories=values)
        return x

    def fit(self,learn,p,s):
        c=self.config; mask=learn.nombre_sinistres.eq(1).to_numpy()
        claims=learn.loc[mask]; y=claims.montant_sinistre.to_numpy()
        if c['kind']=='xgb_ratio':
            # On claim rows, p*s_base^2*(y/s_base-r)^2 = p*(y-r*s_base)^2.
            weights=p[mask]*s[mask]**2;weights=weights/weights.mean()
            self.model_=XGBRegressor(objective='reg:squarederror',base_score=1.,max_depth=c['depth'],
                n_estimators=c['trees'],learning_rate=.025,min_child_weight=100,reg_lambda=100,
                reg_alpha=1.,subsample=1.,colsample_bytree=1.,tree_method='hist',enable_categorical=True,
                n_jobs=4,random_state=20261012)
            self.model_.fit(self.xframe(claims,True),y/s[mask],sample_weight=weights)
        else:
            assert (y>c['shift']).all(),'Shift must remain below every positive training cost'
            self.prep_=construire_preparation(['anciennete_vehicule'],['type_contrat','utilisation'],
                ['anciennete_vehicule'] if c['spline'] else [],20)
            x=self.prep_.fit_transform(preparer_metier(claims))
            target=y-c['shift']; self.scale_=float(target.mean()) if c['kind']=='poisson' else 1.
            weights=p[mask]/p[mask].mean() if c['weighted'] else np.ones(mask.sum())
            cls=GammaRegressor if c['kind']=='gamma' else PoissonRegressor
            self.model_=cls(alpha=c['alpha'],max_iter=5000,tol=1e-5)
            self.model_.fit(x,target/self.scale_,sample_weight=weights)
        return self

    def predict(self,X,base):
        if self.config['kind']=='xgb_ratio':
            out=base*np.maximum(self.model_.predict(self.xframe(X)),1e-8)
        else:out=self.config['shift']+self.scale_*self.model_.predict(self.prep_.transform(preparer_metier(X)))
        assert np.isfinite(out).all() and (out>0).all()
        return out


def reference(learn,evaluation):
    # Reuse exactly identical client partitions from the preceding restart, if present.
    old=r.CACHE/(r.cache_key(learn,evaluation,'reference')+'.npz')
    if old.exists():
        with np.load(old) as z:return z['p'],z['s']
    return r.reference_components(learn,evaluation)


def fold(learn,evaluation):
    path=CACHE/(key(learn,evaluation)+'_predictions.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['s']
    p,base=reference(learn,evaluation);ap,as_=moments(learn)
    predictions=[base]
    for c in catalogue():predictions.append(Severity(c).fit(learn,ap,as_).predict(evaluation,base))
    values=np.column_stack(predictions)
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,s=values)
    return p,values


def selection(frame,label):
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    p=np.zeros(len(frame));s=np.zeros((len(frame),len(catalogue())+1))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],s[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'fold',k,'/5 complete',flush=True)
    choices=[dict(candidate=0,weight=0.)]
    choices.extend(dict(candidate=j,weight=w) for j in range(1,s.shape[1]) for w in [.1,.25,.5,1.])
    cost=frame.montant_sinistre.to_numpy(); rows=[]
    for c in choices:
        pred=p*((1-c['weight'])*s[:,0]+c['weight']*s[:,c['candidate']])
        rows.append(dict(**c,RMSE=r.score(cost,pred)['RMSE']))
    ranking=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight'])
    chosen=ranking.iloc[0];j=int(chosen.candidate)
    choice=dict(candidate=j,weight=float(chosen.weight),config=catalogue()[j-1] if j else None,
        selection_RMSE=float(chosen.RMSE),reference_RMSE=r.score(cost,p*s[:,0])['RMSE'])
    ranking.to_csv(folder/'ranking.csv',index=False)
    r.write_json(folder/'selection.json',dict(**choice,catalogue=catalogue(),
        caveat='Inner selection is optimistic. Prior data/leaderboard exposure cannot be undone by nested evaluation.'))
    np.savez_compressed(folder/'oof.npz',ids=frame['index'],p=p,s=s)
    return choice


class Premium:
    def __init__(self,choice):self.choice=choice

    def fit(self,learn):
        self.frequency=r.ReferenceFrequence().fit(learn,learn.nombre_sinistres,learn.id_client)
        claims=learn.loc[learn.nombre_sinistres.eq(1)]
        self.base=GammaMetier({'alpha':.1}).fit(claims,claims.montant_sinistre,claims.id_client)
        if self.choice['candidate']:
            p,s=moments(learn);self.severity=Severity(self.choice['config']).fit(learn,p,s)
        return self

    def details(self,X):
        p=self.frequency.predict_proba(X)[:,1];base=self.base.predict(X);s=base.copy()
        if self.choice['candidate']:
            w=self.choice['weight'];s=(1-w)*base+w*self.severity.predict(X,base)
        return pd.DataFrame(dict(index=X['index'].to_numpy(),proba_pred=p,montant_pred=s,pred=p*s,reference=p*base))


def run():
    install();train,test,_,_=r.load_data();OUT.mkdir(parents=True,exist_ok=True)
    rows=[];parts=[]
    for k,(a,b) in enumerate(r.split_groups(train,3,r.SEED+23),1):
        choice=selection(train.iloc[a],f'outer_{k}')
        m=Premium(choice).fit(train.iloc[a]);v=m.details(train.iloc[b])
        v['observed']=train.iloc[b].montant_sinistre.to_numpy();v['fold']=k;parts.append(v)
        rows.append(dict(fold=k,choice=choice,selected=r.score(v.observed,v.pred),reference=r.score(v.observed,v.reference)))
        print('OUTER',rows[-1],flush=True)
    v=pd.concat(parts).set_index('index').loc[train['index']].reset_index()
    v.to_csv(OUT/'nested_predictions.csv',index=False)
    gain=r.bootstrap_gain(train.reset_index(drop=True),v.pred.to_numpy(),v.reference.to_numpy())
    result=dict(selected=r.score(v.observed,v.pred),reference=r.score(v.observed,v.reference),gain=gain,folds=rows,
        note='Frequency is unchanged. Covariates/targets have been explored previously; intervals are descriptive.')
    r.write_json(OUT/'results.json',result)
    choice=selection(train,'full_training');result['final_choice']=choice
    # Do not promote or export a candidate whose full selection procedure is worse.
    if result['selected']['RMSE']>=result['reference']['RMSE'] or not choice['candidate']:
        result['export']=None;result['recommendation']='original reference';r.write_json(OUT/'results.json',result)
        print('No supported new candidate. Reference retained.',flush=True);return
    m=Premium(choice).fit(train);path=ROOT/'models/severity_refinement.joblib';joblib.dump(m,path)
    v=joblib.load(path).details(test);destination=ROOT/'deliverables/candidates/submission_severity_refinement.csv'
    v[['index','pred']].to_csv(destination,index=False);r.valid_submission(destination,test)
    original=pd.read_csv(ROOT/'deliverables/candidates/submission_reference.csv')
    np.testing.assert_allclose(v.reference,original.pred,atol=1e-6,rtol=0)
    result['export']=dict(file=str(destination.relative_to(ROOT)),sha256=r.digest(destination),kaggle_score=None)
    result['recommendation']='experimental challenger' if gain['interval_95'][0]>0 else 'original reference; gain remains uncertain'
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();m=joblib.load(ROOT/'models/severity_refinement.joblib')
            m.details(test)[['index','pred']].to_csv(args.output,index=False);r.valid_submission(args.output,test)


if __name__=='__main__':
    from refine_severity import main
    main()
