"""Client-cross-fitted penalized logit offsets; successful severity stays fixed."""
from pathlib import Path
import argparse,json,hashlib,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from threadpoolctl import threadpool_limits
import variable_models as vm
import improve_severity as old
import refine_severity as previous
import restart_xgboost as r
ROOT=r.ROOT;OUT=ROOT/'reports/frequency_curves';CACHE=ROOT/'.cache/frequency_curves'

def catalogue():
    return [dict(panel=p,alpha=a) for p in ['vehicle','age_coverage','compact'] for a in [100.,1000.,10000.]]

def cache_key(*frames):
    h=hashlib.sha256(Path(__file__).read_bytes());h.update(importlib.metadata.version('xgboost').encode())
    for module in [vm,old,previous]:h.update(Path(module.__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode())
    for x in frames:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    return h.hexdigest()

def auxiliary_probability(learn):
    path=CACHE/(cache_key(learn)+'_probability.npz')
    if path.exists():
        with np.load(path) as z:return z['p']
    p=np.zeros(len(learn))
    for a,b in r.split_groups(learn,3,19431):
        x,z=learn.iloc[a],learn.iloc[b]
        assert set(x.id_client).isdisjoint(z.id_client)
        p[b]=r.ReferenceFrequence().fit(x,x.nombre_sinistres,x.id_client).predict_proba(z)[:,1]
    assert ((p>0)&(p<1)).all();CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p)
    return p

def loss_gradient(beta,z,y,offset,alpha):
    eta=offset+z@beta
    loss=float(np.sum(np.logaddexp(0,eta)-y*eta)+.5*alpha*np.dot(beta,beta))
    grad=z.T@(expit(eta)-y)+alpha*beta
    return loss,grad

class Correction:
    def __init__(self,c):self.config=c
    def frame(self,X):
        x=previous.preparer_metier(X)
        nums=['anciennete_vehicule'] if self.config['panel']!='compact' else ['anciennete_vehicule','age_conducteur1','log_bonus','log_prix','log_din']
        cats=[] if self.config['panel']=='vehicle' else ['type_contrat','utilisation']
        for c in cats:x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
        return x[nums+cats],nums,cats
    def matrix(self,X,fit=False):
        x,n,c=self.frame(X)
        if fit:
            parts=[('curves',Pipeline([('missing',SimpleImputer(strategy='median')),
                ('splines',SplineTransformer(n_knots=4,degree=2,include_bias=False,extrapolation='constant')),
                ('scale',StandardScaler())]),n)]
            if c:parts.append(('categories',OneHotEncoder(handle_unknown='ignore',min_frequency=50,sparse_output=False),c))
            self.prep_=ColumnTransformer(parts);z=self.prep_.fit_transform(x)
        else:z=self.prep_.transform(x)
        if self.config['panel']=='age_coverage':
            if fit:
                names=self.prep_.get_feature_names_out()
                self.numeric_=[i for i,a in enumerate(names) if a.startswith('curves__')]
                self.coverage_=[i for i,a in enumerate(names) if a.startswith('categories__type_contrat_')]
            cross=z[:,self.numeric_][:,:,None]*z[:,self.coverage_][:,None,:]
            z=np.column_stack([z,.5*cross.reshape(len(z),-1)])
        return np.column_stack([np.ones(len(z)),z])
    def fit(self,learn,p):
        z=self.matrix(learn,True);y=learn.nombre_sinistres.to_numpy();offset=logit(p)
        opt=minimize(loss_gradient,np.zeros(z.shape[1]),args=(z,y,offset,self.config['alpha']),jac=True,
            method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-6))
        assert opt.success,opt.message;self.coef_=opt.x;self.convergence_=dict(iterations=int(opt.nit),loss=float(opt.fun))
        return self
    def delta(self,X):return self.matrix(X)@self.coef_

def fold(learn,evaluation):
    path=CACHE/(cache_key(learn,evaluation)+'.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['s'],z['d']
    v=vm.Premium(dict(candidate=12,weight=.25)).fit(learn).details(evaluation)
    aux=auxiliary_probability(learn)
    d=np.column_stack([Correction(c).fit(learn,aux).delta(evaluation) for c in catalogue()])
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=v.proba_pred,s=v.montant_pred,d=d)
    return v.proba_pred.to_numpy(),v.montant_pred.to_numpy(),d

def select(frame,label):
    p,s=np.zeros(len(frame)),np.zeros(len(frame));d=np.zeros((len(frame),len(catalogue())))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],s[b],d[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'inner',k,flush=True)
    y=frame.montant_sinistre.to_numpy();rows=[dict(candidate=-1,weight=0.,RMSE=r.score(y,p*s)['RMSE'])]
    for j in range(len(catalogue())):
        for w in [.25,.5,1.]:rows.append(dict(candidate=j,weight=w,RMSE=r.score(y,expit(logit(p)+w*d[:,j])*s)['RMSE']))
    rank=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight']);best=rank.iloc[0]
    choice=dict(candidate=int(best.candidate),weight=float(best.weight),selection_RMSE=float(best.RMSE),incumbent_RMSE=rows[0]['RMSE'])
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);rank.to_csv(folder/'ranking.csv',index=False)
    r.write_json(folder/'selection.json',dict(choice=choice,catalogue=catalogue()));return choice

class Premium:
    def __init__(self,choice):self.choice=choice
    def fit(self,learn):
        self.base=vm.Premium(dict(candidate=12,weight=.25)).fit(learn)
        if self.choice['candidate']>=0:self.correction=Correction(catalogue()[self.choice['candidate']]).fit(learn,auxiliary_probability(learn))
        return self
    def details(self,X):
        v=self.base.details(X);v['incumbent']=v.pred.to_numpy();v['original_probability']=v.proba_pred.to_numpy()
        if self.choice['candidate']>=0:
            v['proba_pred']=expit(logit(v.proba_pred.to_numpy())+self.choice['weight']*self.correction.delta(X))
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
        note='Previously explored clients; nested validation is diagnostic. Severity is exactly the fixed 1365.745 recipe.')
    from sklearn.metrics import log_loss,brier_score_loss
    result['frequency']={name:dict(log_loss=log_loss(train.nombre_sinistres,v[col]),brier=brier_score_loss(train.nombre_sinistres,v[col]))
        for name,col in [('selected','proba_pred'),('incumbent','original_probability')]}
    r.write_json(OUT/'results.json',result);choice=select(train,'full_training');result['final_choice']=choice;result['export']=None
    if result['selected']['RMSE']<result['incumbent']['RMSE'] and choice['candidate']>=0:
        m=Premium(choice).fit(train);path=ROOT/'models/frequency_curves.joblib';joblib.dump(m,path)
        dest=ROOT/'deliverables/candidates/submission_frequency_curves.csv';joblib.load(path).details(test)[['index','pred']].to_csv(dest,index=False)
        r.valid_submission(dest,test);result['export']=dict(file=str(dest.relative_to(ROOT)),sha256=r.digest(dest),model_sha256=r.digest(path),kaggle_score=None)
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();joblib.load(ROOT/'models/frequency_curves.joblib').details(test)[['index','pred']].to_csv(args.output,index=False);r.valid_submission(args.output,test)

if __name__=='__main__':
    from frequency_curves import main
    main()
