"""Strongly pooled coverage/usage interactions around the expanded-variable incumbent."""
from pathlib import Path
import argparse,json,hashlib,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import variable_models as vm
import improve_severity as old
import refine_severity as previous
import restart_xgboost as r
ROOT=r.ROOT;OUT=ROOT/'reports/coverage_interactions';CACHE=ROOT/'.cache/coverage_interactions'

def catalogue():
    return [dict(block=b,alpha=a,scale=s) for b in ['vehicle_coverage','driver_usage','both'] for a in [10.,100.] for s in [.5,1.]]

class Correction:
    def __init__(self,c):self.config=c
    def matrix(self,X,fit=False,weights=None):
        x,_,_=vm.features(X,'all');z=self.base.prep_.transform(x)
        if fit:
            names=self.base.prep_.get_feature_names_out();self.groups_=[]
            blocks=[]
            if self.config['block'] in ['vehicle_coverage','both']:
                blocks.append((['anciennete_vehicule','log_prix','log_din','log_poids','log_cylindre','puissance_par_tonne'], 'type_contrat'))
            if self.config['block'] in ['driver_usage','both']:
                blocks.append((['age_conducteur1','anciennete_permis1','licence_start_age','log_bonus'], 'utilisation'))
            for variables,category in blocks:
                a=[i for i,n in enumerate(names) if any(n.startswith('curves__'+v+'_sp_') for v in variables)]
                b=[i for i,n in enumerate(names) if n.startswith('categories__'+category+'_')]
                assert a and b
                self.groups_.append((a,b,np.average(z[:,a],axis=0,weights=weights),np.average(z[:,b],axis=0,weights=weights)))
        interaction=[]
        for a,b,ma,mb in self.groups_:
            block=(z[:,a]-ma)[:,:,None]*(z[:,b]-mb)[:,None,:]
            interaction.append(block.reshape(len(z),-1)*self.config['scale'])
        return np.column_stack([z]+interaction)
    def fit(self,learn,p,s):
        self.base=vm.Correction(dict(kind='ridge',panel='all',alpha=10.)).fit(learn,p,s)
        mask=learn.nombre_sinistres.eq(1).to_numpy();claims=learn.loc[mask]
        w=p[mask]*s[mask]**2;w/=w.mean();y=claims.montant_sinistre.to_numpy()/s[mask]-1
        self.model_=Ridge(alpha=self.config['alpha']).fit(self.matrix(claims,True,w),y,sample_weight=w)
        return self
    def predict(self,X,base):
        s=base*np.maximum(1+self.model_.predict(self.matrix(X)),1e-8)
        assert np.isfinite(s).all() and (s>0).all();return s

def fold(learn,evaluation):
    h=hashlib.sha256(Path(__file__).read_bytes());h.update(importlib.metadata.version('xgboost').encode())
    for module in [vm,old,previous]:h.update(Path(module.__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    path=CACHE/(h.hexdigest()+'.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['incumbent'],z['s']
    p,base=previous.reference(learn,evaluation);ap,as_=previous.moments(learn)
    anchor=.5*base+.5*old.Correction(old.catalogue()[0]).fit(learn,ap,as_).predict(evaluation,base)
    incumbent=.75*anchor+.25*vm.Correction(dict(kind='ridge',panel='all',alpha=10.)).fit(learn,ap,as_).predict(evaluation,base)
    s=np.column_stack([Correction(c).fit(learn,ap,as_).predict(evaluation,base) for c in catalogue()])
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,incumbent=incumbent,s=s)
    return p,incumbent,s

def select(frame,label):
    p,inc=np.zeros(len(frame)),np.zeros(len(frame));s=np.zeros((len(frame),len(catalogue())))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],inc[b],s[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'inner',k,flush=True)
    y=frame.montant_sinistre.to_numpy();rows=[dict(candidate=-1,weight=0.,RMSE=r.score(y,p*inc)['RMSE'])]
    for j in range(len(catalogue())):
        for w in [.25,.5,1.]:rows.append(dict(candidate=j,weight=w,RMSE=r.score(y,p*((1-w)*inc+w*s[:,j]))['RMSE']))
    rank=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight']);best=rank.iloc[0]
    choice=dict(candidate=int(best.candidate),weight=float(best.weight),selection_RMSE=float(best.RMSE),incumbent_RMSE=rows[0]['RMSE'])
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);rank.to_csv(folder/'ranking.csv',index=False)
    r.write_json(folder/'selection.json',dict(choice=choice,catalogue=catalogue()));return choice

class Premium:
    def __init__(self,choice):self.choice=choice
    def fit(self,learn):
        self.base=vm.Premium(dict(candidate=12,weight=.25)).fit(learn)
        if self.choice['candidate']>=0:
            p,s=previous.moments(learn);self.correction=Correction(catalogue()[self.choice['candidate']]).fit(learn,p,s)
        return self
    def details(self,X):
        v=self.base.details(X);v['incumbent']=v.pred.to_numpy()
        if self.choice['candidate']>=0:
            gamma=self.base.incumbent.base.base.predict(X);w=self.choice['weight']
            v['montant_pred']=(1-w)*v.montant_pred+w*self.correction.predict(X,gamma);v['pred']=v.proba_pred*v.montant_pred
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
        note='Previously explored clients; nested validation is diagnostic. Benchmark is the fixed 1365.745 recipe refitted per fold.')
    r.write_json(OUT/'results.json',result);choice=select(train,'full_training');result['final_choice']=choice;result['export']=None
    if result['selected']['RMSE']<result['incumbent']['RMSE'] and choice['candidate']>=0:
        m=Premium(choice).fit(train);path=ROOT/'models/coverage_interactions.joblib';joblib.dump(m,path)
        dest=ROOT/'deliverables/candidates/submission_coverage_interactions.csv';joblib.load(path).details(test)[['index','pred']].to_csv(dest,index=False)
        r.valid_submission(dest,test);result['export']=dict(file=str(dest.relative_to(ROOT)),sha256=r.digest(dest),model_sha256=r.digest(path),kaggle_score=None)
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();joblib.load(ROOT/'models/coverage_interactions.joblib').details(test)[['index','pred']].to_csv(args.output,index=False);r.valid_submission(args.output,test)

if __name__=='__main__':
    from coverage_interactions import main
    main()
