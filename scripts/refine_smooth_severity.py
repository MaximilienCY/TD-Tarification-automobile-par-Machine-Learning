"""Refine smooth severity pooling, benchmarked against the 1365.745 recipe."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer,OneHotEncoder
from sklearn.linear_model import Ridge,GammaRegressor,PoissonRegressor
from threadpoolctl import threadpool_limits
import variable_models as vm
import improve_severity as old
import refine_severity as previous
import restart_xgboost as r
ROOT=r.ROOT;OUT=ROOT/'reports/smooth_severity';CACHE=ROOT/'.cache/smooth_severity'

def catalogue():
    base=dict(kind='ridge',alpha=10.,knots=4,rare=15,category_scale=1.)
    c=[dict(base,alpha=a) for a in [1.,3.,10.,30.,100.]]
    for name,values in [('knots',[3,5,6]),('rare',[5,30,50]),('category_scale',[.5,2.])]:
        c += [dict(base,**{name:x}) for x in values]
    c += [dict(base,kind=k,alpha=a) for k in ['gamma','poisson'] for a in [.001,.01,.1]]
    return c

class Correction:
    def __init__(self,c):self.config=c
    def fit(self,learn,p,s):
        mask=learn.nombre_sinistres.eq(1).to_numpy();claims=learn.loc[mask]
        x,n,c=vm.features(claims,'all');config=self.config
        binary=[a for a in n if x[a].nunique()<=2];smooth=[a for a in n if a not in binary]
        parts=[('curves',Pipeline([('missing',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('splines',SplineTransformer(n_knots=config['knots'],degree=2,include_bias=False,extrapolation='constant'))]),smooth),
            ('categories',OneHotEncoder(min_frequency=config['rare'],handle_unknown='infrequent_if_exist',sparse_output=False),c)]
        if binary:parts.append(('flags',SimpleImputer(strategy='constant',fill_value=0),binary))
        self.prep_=ColumnTransformer(parts,transformer_weights={'categories':config['category_scale']})
        target=claims.montant_sinistre.to_numpy()/s[mask];w=p[mask]*s[mask]**2;w/=w.mean()
        if config['kind']=='ridge':self.model_=Ridge(alpha=config['alpha'],fit_intercept=True);target=target-1
        else:
            cls=GammaRegressor if config['kind']=='gamma' else PoissonRegressor
            self.model_=cls(alpha=config['alpha'],max_iter=3000,tol=1e-7)
        self.model_.fit(self.prep_.fit_transform(x),target,sample_weight=w)
        if config['kind']!='ridge':assert self.model_.n_iter_<3000
        return self
    def predict(self,X,base):
        x,_,_=vm.features(X,'all');ratio=self.model_.predict(self.prep_.transform(x))
        if self.config['kind']=='ridge':ratio=1+ratio
        out=base*np.maximum(ratio,1e-8);assert np.isfinite(out).all() and (out>0).all()
        return out

def fold(learn,evaluation):
    h=hashlib.sha256(Path(__file__).read_bytes());h.update(Path(vm.__file__).read_bytes())
    import importlib.metadata
    h.update(importlib.metadata.version('xgboost').encode())
    h.update(Path(old.__file__).read_bytes());h.update(Path(previous.__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    path=CACHE/(h.hexdigest()+'.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['anchor'],z['s']
    p,base=previous.reference(learn,evaluation);ap,as_=previous.moments(learn)
    anchor=.5*base+.5*old.Correction(old.catalogue()[0]).fit(learn,ap,as_).predict(evaluation,base)
    values=np.column_stack([Correction(c).fit(learn,ap,as_).predict(evaluation,base) for c in catalogue()])
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,anchor=anchor,s=values)
    return p,anchor,values

def select(frame,label):
    p,anchor=np.zeros(len(frame)),np.zeros(len(frame));s=np.zeros((len(frame),len(catalogue())))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],anchor[b],s[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'inner',k,flush=True)
    y=frame.montant_sinistre.to_numpy();rows=[]
    for j in range(len(catalogue())):
        for w in [.1,.25,.5,.75,1.]:rows.append(dict(candidate=j,weight=w,RMSE=r.score(y,p*((1-w)*anchor+w*s[:,j]))['RMSE']))
    rows.append(dict(candidate=-1,weight=0.,RMSE=r.score(y,p*anchor)['RMSE']))
    rank=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight']);best=rank.iloc[0]
    choice=dict(candidate=int(best.candidate),weight=float(best.weight),selection_RMSE=float(best.RMSE),
        incumbent_RMSE=r.score(y,p*(.75*anchor+.25*s[:,2]))['RMSE'])
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);rank.to_csv(folder/'ranking.csv',index=False)
    r.write_json(folder/'selection.json',dict(choice=choice,catalogue=catalogue()))
    return choice

class Premium:
    def __init__(self,choice):self.choice=choice
    def fit(self,learn):
        self.anchor=old.Premium(dict(candidate=0,weight=.5)).fit(learn)
        p,s=previous.moments(learn)
        self.benchmark=Correction(catalogue()[2]).fit(learn,p,s)
        if self.choice['candidate']>=0:self.correction=Correction(catalogue()[self.choice['candidate']]).fit(learn,p,s)
        return self
    def details(self,X):
        v=self.anchor.details(X);base=self.anchor.base.base.predict(X);anchor=v.montant_pred.to_numpy()
        v['incumbent']=v.proba_pred*(.75*anchor+.25*self.benchmark.predict(X,base))
        if self.choice['candidate']>=0:
            w=self.choice['weight'];v['montant_pred']=(1-w)*anchor+w*self.correction.predict(X,base)
            v['pred']=v.proba_pred*v.montant_pred
        return v

def run():
    old.install();train,test,_,_=r.load_data();OUT.mkdir(parents=True,exist_ok=True);parts=[];rows=[]
    for k,(a,b) in enumerate(r.split_groups(train,3,r.SEED+23),1):
        choice=select(train.iloc[a],f'outer_{k}');v=Premium(choice).fit(train.iloc[a]).details(train.iloc[b])
        v['observed']=train.iloc[b].montant_sinistre.to_numpy();v['fold']=k;parts.append(v)
        rows.append(dict(fold=k,choice=choice,selected=r.score(v.observed,v.pred),incumbent=r.score(v.observed,v.incumbent)))
        print('OUTER',rows[-1],flush=True)
    v=pd.concat(parts).set_index('index').loc[train['index']].reset_index();v.to_csv(OUT/'nested_predictions.csv',index=False)
    result=dict(selected=r.score(v.observed,v.pred),incumbent=r.score(v.observed,v.incumbent),folds=rows,
        gain=r.bootstrap_gain(train.reset_index(drop=True),v.pred.to_numpy(),v.incumbent.to_numpy()),
        note='Fixed incumbent recipe refitted per outer fold. Previously explored clients and public-feedback-informed search; diagnostic only.')
    r.write_json(OUT/'results.json',result);choice=select(train,'full_training');result['final_choice']=choice;result['export']=None
    same=choice['candidate']==2 and choice['weight']==.25
    if result['selected']['RMSE']<result['incumbent']['RMSE'] and not same and choice['candidate']>=0:
        m=Premium(choice).fit(train);path=ROOT/'models/smooth_severity.joblib';joblib.dump(m,path)
        dest=ROOT/'deliverables/candidates/submission_smooth_severity.csv';joblib.load(path).details(test)[['index','pred']].to_csv(dest,index=False)
        r.valid_submission(dest,test);result['export']=dict(file=str(dest.relative_to(ROOT)),sha256=r.digest(dest),model_sha256=r.digest(path),kaggle_score=None)
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();joblib.load(ROOT/'models/smooth_severity.joblib').details(test)[['index','pred']].to_csv(args.output,index=False);r.valid_submission(args.output,test)

if __name__=='__main__':
    from refine_smooth_severity import main
    main()
