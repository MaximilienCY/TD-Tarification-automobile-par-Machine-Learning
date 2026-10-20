"""Second research round: grouped OOF selection; original pipeline stays frozen.

Run with the project Python: scripts/advanced_search.py search|confirm|export.
All selection uses development rows; confirmation is a sensitivity check, not a
fresh unbiased holdout. The historical partition is reported only after selection.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from pricing.models import tree_frame, PremiumModel
from pricing.reference import preparer_metier, construire_preparation
from pricing.experiment import load_data, provenance, write_json, metrics, bootstrap_gain

OUT = ROOT/'reports'/'advanced'
CACHE = ROOT/'.cache'/'advanced'


def catalogue():
    f=[]
    for depth, trees, l2 in [(2,1200,10),(4,1200,20),(6,1400,40),(8,900,60)]:
        f.append(dict(name=f'cat_d{depth}_n{trees}',kind='cat',depth=depth,trees=trees,l2=l2))
    f.append(dict(name='cat_d6_seed2026',kind='cat',depth=6,trees=700,l2=30,seed=2026))
    for leaves,trees in [(4,700),(8,700),(16,500)]:
        f.append(dict(name=f'lgb_l{leaves}_n{trees}',kind='lgb',leaves=leaves,trees=trees))
    g=[]
    for form in ['linear','log_age','spline_age']:
        for interaction in [False,True]:
            g.append(dict(name=f'gamma_{form}_interaction{interaction}',kind='glm',form=form,interaction=interaction,alpha=.01))
    for form in ['linear','log_age','spline_age']:
        g.append(dict(name=f'poisson_{form}',kind='glm',form=form,objective='poisson',alpha=.05))
    for leaves in [2,4,8]:
        g.append(dict(name=f'lgb_gamma_l{leaves}',kind='lgb',leaves=leaves,trees=200,objective='gamma',compact=True))
    for objective in ['Poisson','Tweedie:variance_power=1.5']:
        g.append(dict(name=f'cat_severity_{objective.split(":")[0]}',kind='cat',depth=3,trees=400,l2=30,objective=objective,compact=True))
    return f,g


class Candidate:
    def __init__(self,c,frequency):
        self.c,self.frequency=c,frequency

    def frame(self,X):
        c=self.c
        if c['kind']=='glm':
            z=preparer_metier(X)
            if c['form']=='log_age':
                z['anciennete_vehicule']=np.log1p(z.anciennete_vehicule.clip(lower=0))
            if c.get('interaction'):
                for contract in ['Maxi','Median1','Median2','Mini']:
                    z['age_'+contract]=z.anciennete_vehicule*(z.type_contrat==contract)
            return z
        return tree_frame(X,c.get('compact',False))

    def fit(self,X,y):
        c=self.c;z=self.frame(X)
        self.scale_=1. if self.frequency else float(np.mean(y))
        target=np.asarray(y)/self.scale_
        if c['kind']=='glm':
            nums=['anciennete_vehicule']
            if c.get('interaction'):nums += ['age_'+s for s in ['Maxi','Median1','Median2','Mini']]
            self.prep_=construire_preparation(nums,['type_contrat','utilisation'],['anciennete_vehicule'] if c['form']=='spline_age' else [],20)
            z=self.prep_.fit_transform(z)
            cls=PoissonRegressor if c.get('objective')=='poisson' else GammaRegressor
            self.model_=cls(alpha=c['alpha'],max_iter=5000,tol=1e-7)
        else:
            self.cats_=list(z.select_dtypes(exclude='number').columns)
            if c['kind']=='cat':
                cls=CatBoostClassifier if self.frequency else CatBoostRegressor
                self.model_=cls(iterations=c['trees'],depth=c['depth'],learning_rate=.03,
                    l2_leaf_reg=c['l2'],loss_function='Logloss' if self.frequency else c['objective'],
                    cat_features=self.cats_,thread_count=4,random_seed=c.get('seed',42),
                    allow_writing_files=False,verbose=False)
            else:
                self.categories_={col:sorted(z[col].unique()) for col in self.cats_}
                z=self.lgb_frame(z)
                cls=LGBMClassifier if self.frequency else LGBMRegressor
                self.model_=cls(objective='binary' if self.frequency else c['objective'],
                    n_estimators=c['trees'],num_leaves=c['leaves'],learning_rate=.02,
                    min_child_samples=250 if self.frequency else 150,reg_lambda=20,
                    n_jobs=4,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)
        self.model_.fit(z,target)
        return self

    def lgb_frame(self,z):
        z=z.copy()
        for col,categories in self.categories_.items():z[col]=pd.Categorical(z[col],categories=categories)
        return z

    def predict(self,X):
        z=self.frame(X)
        if self.c['kind']=='glm':z=self.prep_.transform(z)
        if self.c['kind']=='lgb':z=self.lgb_frame(z)
        if self.frequency:return self.model_.predict_proba(z)[:,1]
        if self.c['kind']=='cat':
            values=self.model_.predict(z,prediction_type='Exponent')
        else:values=self.model_.predict(z)
        return np.maximum(values*self.scale_,1e-8)


def cached(learn,evaluation,c,frequency):
    h=hashlib.sha256(json.dumps(dict(config=c,frequency=frequency,provenance=provenance()),sort_keys=True).encode())
    h.update(Path(__file__).read_bytes())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    path=CACHE/(h.hexdigest()+'.npy')
    if path.exists():return np.load(path)
    subset=learn if frequency else learn.loc[learn.nombre_sinistres.eq(1)]
    y=subset.nombre_sinistres if frequency else subset.montant_sinistre
    m=Candidate(c,frequency).fit(subset,y)
    v=m.predict(evaluation)
    assert np.isfinite(v).all() and (v>=0).all()
    if frequency:assert (v<=1).all()
    CACHE.mkdir(parents=True,exist_ok=True);np.save(path,v)
    return v


def incumbent(learn,evaluation):
    old=json.loads((ROOT/'reports/selection.json').read_text())
    ch=old['choice']
    m=PremiumModel(old['frequency_configs'],old['severity_configs'],ch['frequency_weights'],ch['severity_weights']).fit(learn)
    d=m.predict_details(evaluation)
    return d.proba_pred.to_numpy(),d.montant_pred.to_numpy()


def blend_columns(base,values,configs):
    arrays=[base];desc=[dict(config=None,weight=0.,name='incumbent')]
    for i,c in enumerate(configs):
        for w in [.25,.5,.75,1.]:
            arrays.append((1-w)*base+w*values[:,i])
            desc.append(dict(config=c,weight=w,name=f'{w:g}*{c["name"]}+{1-w:g}*incumbent'))
    return np.column_stack(arrays),desc


def search():
    train,test,dev,hist=load_data();fc,gc=catalogue()
    old=np.load(ROOT/'reports/search_oof.npz')
    assert np.array_equal(old['ids'],dev['index'])
    old_meta=json.loads((ROOT/'reports/selection.json').read_text())
    assert old_meta['provenance']==provenance(), 'Original cache provenance changed'
    p0=old['p']@old_meta['choice']['frequency_weights']
    s0=old['s']@old_meta['choice']['severity_weights']
    p=np.zeros((len(dev),len(fc)));s=np.zeros((len(dev),len(gc)))
    for fold in range(5):
        a=np.flatnonzero(old['fold']!=fold);b=np.flatnonzero(old['fold']==fold)
        assert set(dev.iloc[a].id_client).isdisjoint(dev.iloc[b].id_client)
        for frequency,configs,out in [(True,fc,p),(False,gc,s)]:
            for j,c in enumerate(configs):
                t=time.perf_counter();out[b,j]=cached(dev.iloc[a],dev.iloc[b],c,frequency)
                print(f'fold {fold+1}/5 {c["name"]} {time.perf_counter()-t:.1f}s',flush=True)
    fp,fd=blend_columns(p0,p,fc);gs,gd=blend_columns(s0,s,gc)
    rows=[];y=dev.montant_sinistre.to_numpy()
    for i in range(fp.shape[1]):
        mse=((fp[:,i,None]*gs-y[:,None])**2).mean(axis=0)
        for j in range(gs.shape[1]):rows.append(dict(fi=i,gi=j,RMSE=float(np.sqrt(mse[j])),frequency=fd[i]['name'],severity=gd[j]['name']))
    ranking=pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    OUT.mkdir(parents=True,exist_ok=True);ranking.to_csv(OUT/'ranking.csv',index=False)
    # Keep a diverse runner-up only when it uses a different frequency or severity family.
    shortlist=[ranking.iloc[0]]
    for _,r in ranking.iterrows():
        if r.fi!=shortlist[0].fi and r.gi!=shortlist[0].gi:
            shortlist.append(r);break
    choices=[dict(frequency=fd[int(r.fi)],severity=gd[int(r.gi)],search_RMSE=float(r.RMSE)) for r in shortlist]
    write_json(OUT/'selection.json',dict(choices=choices,baseline_RMSE=metrics(y,p0*s0)['RMSE'],
        combinations=len(rows),provenance=provenance(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Selected OOF scores are optimistic; this is not an independent test.'))
    np.savez_compressed(CACHE/'search_oof.npz',p=p,s=s,p0=p0,s0=s0,fold=old['fold'])
    print(ranking.head(12).to_string(index=False),flush=True)


def choice_components(learn,evaluation,ch,p0,s0):
    components=[]
    for frequency,base,key in [(True,p0,'frequency'),(False,s0,'severity')]:
        c=ch[key];w=c['weight']
        components.append(base if not w else (1-w)*base+w*cached(learn,evaluation,c['config'],frequency))
    return components


def confirm():
    train,test,dev,hist=load_data();selection=json.loads((OUT/'selection.json').read_text());choices=selection['choices']
    pred=np.zeros((len(dev),len(choices)+1));fold_rows=[]
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=74191)
    for fold,(a,b) in enumerate(cv.split(dev,dev.nombre_sinistres,dev.id_client)):
        learn,evaluation=dev.iloc[a],dev.iloc[b]
        assert set(learn.id_client).isdisjoint(evaluation.id_client)
        p0,s0=incumbent(learn,evaluation);pred[b,0]=p0*s0
        for j,ch in enumerate(choices,1):
            p,s=choice_components(learn,evaluation,ch,p0,s0);pred[b,j]=p*s
        for j in range(pred.shape[1]):fold_rows.append(dict(fold=fold+1,candidate=j,**metrics(evaluation.montant_sinistre,pred[b,j])))
        print(f'confirmation fold {fold+1}/5 complete',flush=True)
    summary=[dict(candidate=j,**metrics(dev.montant_sinistre,pred[:,j])) for j in range(pred.shape[1])]
    # Confirmation tests stability; never describe it as unbiased after prior selection.
    gains={str(j):bootstrap_gain(dev,pred[:,j],pred[:,0]) for j in range(1,pred.shape[1])}
    pd.DataFrame(fold_rows).to_csv(OUT/'confirmation_folds.csv',index=False)
    np.savez_compressed(CACHE/'confirmation.npz',predictions=pred)
    write_json(OUT/'confirmation.json',dict(scores=summary,gains=gains,note='Different folds of reused development data; sensitivity check, not independent validation.'))
    print(json.dumps(summary,indent=2),flush=True)


def export():
    train,test,dev,hist=load_data();selection=json.loads((OUT/'selection.json').read_text());choices=selection['choices']
    confirmation=json.loads((OUT/'confirmation.json').read_text())
    p0h,s0h=incumbent(dev,hist);p0,s0=incumbent(train,test)
    sample=pd.read_csv(ROOT/'Data/prime_pred_sandbox.csv')
    assert np.array_equal(sample['index'],test['index'])
    folder=ROOT/'deliverables'/'candidates';folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for j,ch in enumerate(choices,1):
        p,s=choice_components(train,test,ch,p0,s0)
        ph,sh=choice_components(dev,hist,ch,p0h,s0h)
        assert np.isfinite(p*s).all() and (p*s>=0).all() and ((p>=0)&(p<=1)).all()
        filename=folder/f'submission_advanced_{j}.csv'
        pd.DataFrame({'index':test['index'],'pred':p*s}).to_csv(filename,index=False)
        pd.DataFrame({'index':test['index'],'proba_pred':p,'montant_pred':s,'pred':p*s}).to_csv(CACHE/f'components_{j}.csv',index=False)
        entries.append(dict(candidate=j,file=str(filename.relative_to(ROOT)),sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            choice=ch,historical_diagnostic=metrics(hist.montant_sinistre,ph*sh),kaggle_score=None))
    write_json(OUT/'exports.json',dict(entries=entries,confirmation=confirmation,incumbent_historical=metrics(hist.montant_sinistre,p0h*s0h),
        provenance=provenance(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        submission_status='Blocked: Kaggle browser permission denied; no upload attempted for these candidates.'))
    print(json.dumps(entries,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        {'search':search,'confirm':confirm,'export':export}[sys.argv[1]]()
