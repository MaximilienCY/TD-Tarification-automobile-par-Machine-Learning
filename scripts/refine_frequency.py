"""Cost-sensitive proper probability losses; keep original Gamma severity fixed."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import refine_severity as sev
import restart_xgboost as r
from restart_cache import install
from pricing.reference import apprendre_frequence,preparer_frequence,LogistiqueMetier
OUT=ROOT/'reports/frequency_refinement';CACHE=ROOT/'.cache/frequency_refinement'


def catalogue():
    return [dict(C=c,power=p,loss=loss) for c in [.03,.1,.3,1.] for p in [0,2] for loss in ['logloss','brier']]


def brier_loss(theta,x,y,w,prior_variance,C):
    p=expit(x@theta[:-1]+theta[-1]);d=p-y;n=len(y)
    value=float(np.dot(w,d*d)/(2*n*prior_variance)+np.dot(theta[:-1],theta[:-1])/(2*C*n))
    v=w*d*p*(1-p)/(n*prior_variance)
    gradient=np.r_[x.T@v+theta[:-1]/(C*n),v.sum()]
    return value,gradient


class Frequency:
    def __init__(self,config):self.config=config

    def fit(self,learn,severity,prepared=None):
        if prepared is None:prepared=apprendre_frequence(learn,learn.nombre_sinistres,learn.id_client)
        self.params_,z,_=prepared
        x=z.to_numpy();y=learn.nombre_sinistres.to_numpy();c=self.config
        w=np.asarray(severity)**c['power'];w=w/w.mean()
        initial=LogisticRegression(C=c['C'],max_iter=3000).fit(x,y,sample_weight=w)
        if c['loss']=='logloss':
            self.coef_=initial.coef_.ravel();self.intercept_=float(initial.intercept_[0]);self.optimization_=dict(success=True)
        else:
            theta=np.r_[initial.coef_.ravel(),initial.intercept_[0]]
            result=minimize(brier_loss,theta,args=(x,y,w,float(y.mean()*(1-y.mean())),c['C']),
                method='L-BFGS-B',jac=True,options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8,'maxls':50})
            assert result.success, str(result.message)
            self.coef_=result.x[:-1];self.intercept_=float(result.x[-1])
            self.optimization_=dict(success=bool(result.success),iterations=int(result.nit),objective=float(result.fun))
        return self

    def predict(self,X):
        p=expit(preparer_frequence(X,self.params_).to_numpy()@self.coef_+self.intercept_)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        return p


def key(learn,evaluation):
    h=hashlib.sha256(Path(__file__).read_bytes());h.update(Path(sev.__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    return h.hexdigest()


def fold(learn,evaluation):
    path=CACHE/(key(learn,evaluation)+'.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['s']
    ref,s=sev.reference(learn,evaluation);_,auxs=sev.moments(learn)
    prepared=apprendre_frequence(learn,learn.nombre_sinistres,learn.id_client)
    p=np.column_stack([ref]+[Frequency(c).fit(learn,auxs,prepared).predict(evaluation) for c in catalogue()])
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,s=s)
    return p,s


def selection(frame,label):
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    p=np.zeros((len(frame),len(catalogue())+1));s=np.zeros(len(frame))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],s[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'fold',k,'/5 complete',flush=True)
    choices=[dict(candidate=0,weight=0.)]+[dict(candidate=j,weight=w) for j in range(1,p.shape[1]) for w in [.1,.25,.5,1.]]
    rows=[];y=frame.montant_sinistre.to_numpy()
    for c in choices:
        v=((1-c['weight'])*p[:,0]+c['weight']*p[:,c['candidate']])*s
        rows.append(dict(**c,RMSE=r.score(y,v)['RMSE']))
    rank=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight']);best=rank.iloc[0];j=int(best.candidate)
    choice=dict(candidate=j,weight=float(best.weight),config=catalogue()[j-1] if j else None,
        selection_RMSE=float(best.RMSE),reference_RMSE=r.score(y,p[:,0]*s)['RMSE'])
    rank.to_csv(folder/'ranking.csv',index=False);r.write_json(folder/'selection.json',dict(**choice,catalogue=catalogue()))
    np.savez_compressed(folder/'oof.npz',p=p,s=s,ids=frame['index'])
    return choice


class Premium:
    def __init__(self,choice):self.choice=choice

    def fit(self,learn):
        self.base=sev.Premium(dict(candidate=0,weight=0.,config=None)).fit(learn)
        if self.choice['candidate']:
            _,s=sev.moments(learn);self.frequency=Frequency(self.choice['config']).fit(learn,s)
        return self

    def details(self,X):
        v=self.base.details(X)
        if self.choice['candidate']:
            w=self.choice['weight'];v['proba_pred']=(1-w)*v.proba_pred.to_numpy()+w*self.frequency.predict(X)
            v['pred']=v.proba_pred.to_numpy()*v.montant_pred.to_numpy()
        return v


def run():
    install();train,test,_,_=r.load_data();OUT.mkdir(parents=True,exist_ok=True);parts=[];rows=[]
    for k,(a,b) in enumerate(r.split_groups(train,3,r.SEED+23),1):
        choice=selection(train.iloc[a],f'outer_{k}');m=Premium(choice).fit(train.iloc[a]);v=m.details(train.iloc[b])
        v['observed']=train.iloc[b].montant_sinistre.to_numpy();v['fold']=k;parts.append(v)
        rows.append(dict(fold=k,choice=choice,selected=r.score(v.observed,v.pred),reference=r.score(v.observed,v.reference)))
        print('OUTER',rows[-1],flush=True)
    v=pd.concat(parts).set_index('index').loc[train['index']].reset_index();v.to_csv(OUT/'nested_predictions.csv',index=False)
    gain=r.bootstrap_gain(train.reset_index(drop=True),v.pred.to_numpy(),v.reference.to_numpy())
    result=dict(selected=r.score(v.observed,v.pred),reference=r.score(v.observed,v.reference),gain=gain,folds=rows,
        note='Gamma severity unchanged; auxiliary loss weights exclude each training client. Previously explored data; descriptive intervals.')
    r.write_json(OUT/'results.json',result);choice=selection(train,'full_training');result['final_choice']=choice
    if result['selected']['RMSE']>=result['reference']['RMSE'] or not choice['candidate']:
        result['export']=None;result['recommendation']='original reference';r.write_json(OUT/'results.json',result)
        print('No supported new candidate. Reference retained.',flush=True);return
    m=Premium(choice).fit(train);path=ROOT/'models/frequency_refinement.joblib';joblib.dump(m,path)
    v=joblib.load(path).details(test);dest=ROOT/'deliverables/candidates/submission_frequency_refinement.csv'
    v[['index','pred']].to_csv(dest,index=False);r.valid_submission(dest,test)
    original=pd.read_csv(ROOT/'deliverables/candidates/submission_reference.csv')
    np.testing.assert_allclose(v.reference,original.pred,atol=1e-6,rtol=0)
    result['export']=dict(file=str(dest.relative_to(ROOT)),sha256=r.digest(dest),kaggle_score=None)
    result['recommendation']='experimental challenger' if gain['interval_95'][0]>0 else 'original reference; gain remains uncertain'
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();m=joblib.load(ROOT/'models/frequency_refinement.joblib')
            m.details(test)[['index','pred']].to_csv(args.output,index=False);r.valid_submission(args.output,test)


if __name__=='__main__':
    from refine_frequency import main
    main()
