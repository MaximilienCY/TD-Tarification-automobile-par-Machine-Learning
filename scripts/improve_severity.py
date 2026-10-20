"""Controlled successors to the user's 1365.913 severity-only public incumbent."""
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
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
import refine_severity as previous
import restart_xgboost as r
from restart_cache import install
OUT=ROOT/'reports/severity_successor';CACHE=ROOT/'.cache/severity_successor'


def catalogue():
    anchor=dict(name='incumbent_correction',panel='basic',mode='ratio',depth=2,trees=240,
        lr=.025,min_child_weight=100.,reg_lambda=100.,reg_alpha=1.,max_bin=256,seeds=1,subsample=1.)
    configs=[anchor]
    for field,values in [('trees',[120,360,480]),('depth',[1,3]),('min_child_weight',[50.,200.]),
                         ('reg_lambda',[30.,300.]),('lr',[.0125,.05]),('reg_alpha',[0.,10.]),('max_bin',[64,128])]:
        for value in values:configs.append(dict(anchor,**{field:value,'name':f'{field}_{value}'}))
    for n in [240,480]:configs.append(dict(anchor,name=f'bagged_{n}',trees=n,seeds=3,subsample=.8))
    for n in [240,480]:configs.append(dict(anchor,name=f'additive_{n}',trees=n,mode='additive'))
    configs.append(dict(anchor,name='constant_ratio',mode='constant'))
    return configs


class Correction(previous.Severity):
    def fit(self,learn,p,s):
        c=self.config;mask=learn.nombre_sinistres.eq(1).to_numpy();x=learn.loc[mask]
        y=x.montant_sinistre.to_numpy();base=s[mask]
        if c['mode']=='additive':
            self.scale_=float(y.mean());target=(y-base)/self.scale_;w=p[mask];start=0.
        else:
            self.scale_=1.;target=y/base;w=p[mask]*base**2;start=1.
        w=w/w.mean()
        if c['mode']=='constant':
            self.constant_=float(np.average(target,weights=w));return self
        self.models_=[];encoded=self.xframe(x,True)
        for seed in range(c['seeds']):
            m=XGBRegressor(objective='reg:squarederror',base_score=start,max_depth=c['depth'],
                n_estimators=c['trees'],learning_rate=c['lr'],min_child_weight=c['min_child_weight'],
                reg_lambda=c['reg_lambda'],reg_alpha=c['reg_alpha'],subsample=c['subsample'],
                colsample_bytree=1.,max_bin=c['max_bin'],tree_method='hist',enable_categorical=True,
                n_jobs=4,random_state=20261012+seed)
            m.fit(encoded,target,sample_weight=w);self.models_.append(m)
        return self

    def predict(self,X,base):
        if self.config['mode']=='constant':return base*self.constant_
        x=self.xframe(X);delta=np.mean([m.predict(x) for m in self.models_],axis=0)
        out=base+self.scale_*delta if self.config['mode']=='additive' else base*delta
        out=np.maximum(out,1e-8);assert np.isfinite(out).all()
        return out


def stable_members():
    return [j for j,c in enumerate(catalogue()) if c['name'] in ['incumbent_correction','min_child_weight_200.0','reg_lambda_300.0','bagged_240']]


def append_averages(values):
    return np.column_stack([values,np.mean(values[:,stable_members()],axis=1),np.mean(values,axis=1)])


def source_key(learn,evaluation):
    import importlib.metadata
    h=hashlib.sha256(Path(__file__).read_bytes());h.update(Path(previous.__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode());h.update(importlib.metadata.version('xgboost').encode())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    return h.hexdigest()


def fold(learn,evaluation):
    path=CACHE/(source_key(learn,evaluation)+'.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['base'],z['corrections']
    p,base=previous.reference(learn,evaluation);ap,as_=previous.moments(learn)
    values=np.column_stack([Correction(c).fit(learn,ap,as_).predict(evaluation,base) for c in catalogue()])
    values=append_averages(values);CACHE.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=p,base=base,corrections=values)
    return p,base,values


def choices():
    return [dict(candidate=-1,weight=0.)]+[dict(candidate=j,weight=w) for j in range(len(catalogue())+2) for w in [.25,.5,.75,1.]]


def select(frame,label):
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    p,base=np.zeros(len(frame)),np.zeros(len(frame));values=np.zeros((len(frame),len(catalogue())+2))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],base[b],values[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'inner',k,'/5',flush=True)
    incumbent=p*(.5*base+.5*values[:,0]);cost=frame.montant_sinistre.to_numpy()
    rows=[]
    for c in choices():
        s=base if c['candidate']<0 else (1-c['weight'])*base+c['weight']*values[:,c['candidate']]
        rows.append(dict(**c,RMSE=r.score(cost,p*s)['RMSE']))
    rank=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight']);best=rank.iloc[0]
    choice=dict(candidate=int(best.candidate),weight=float(best.weight),selection_RMSE=float(best.RMSE),
        incumbent_RMSE=r.score(cost,incumbent)['RMSE'],original_RMSE=r.score(cost,p*base)['RMSE'])
    rank.to_csv(folder/'ranking.csv',index=False);r.write_json(folder/'selection.json',dict(choice=choice,catalogue=catalogue(),
        ensemble_candidates={str(len(catalogue())):stable_members(),str(len(catalogue())+1):list(range(len(catalogue())))},
        note='Inner scores are selection scores. Current incumbent was identified using earlier data and public-score feedback.'))
    np.savez_compressed(folder/'oof.npz',p=p,base=base,corrections=values,ids=frame['index'])
    return choice


class Premium:
    def __init__(self,choice):self.choice=choice

    def fit(self,learn):
        self.base=previous.Premium(dict(candidate=0,weight=0.,config=None)).fit(learn)
        j=self.choice['candidate'];self.members_=[]
        if j>=0:
            members=[j] if j<len(catalogue()) else stable_members() if j==len(catalogue()) else list(range(len(catalogue())))
            p,s=previous.moments(learn)
            self.members_=[Correction(catalogue()[k]).fit(learn,p,s) for k in members]
        return self

    def details(self,X):
        v=self.base.details(X)
        if self.members_:
            s=v.montant_pred.to_numpy();correction=np.mean([m.predict(X,s) for m in self.members_],axis=0)
            w=self.choice['weight'];v['montant_pred']=(1-w)*s+w*correction
            v['pred']=v.proba_pred.to_numpy()*v.montant_pred.to_numpy()
        return v


def run():
    install();train,test,_,_=r.load_data();OUT.mkdir(parents=True,exist_ok=True);parts=[];rows=[]
    # Explicit comparison with a fixed refit of the new public incumbent.
    benchmark=dict(candidate=0,weight=.5)
    for k,(a,b) in enumerate(r.split_groups(train,3,r.SEED+23),1):
        learn,evaluation=train.iloc[a],train.iloc[b];choice=select(learn,f'outer_{k}')
        m=Premium(choice).fit(learn);v=m.details(evaluation)
        v['incumbent']=Premium(benchmark).fit(learn).details(evaluation).pred.to_numpy()
        v['observed']=evaluation.montant_sinistre.to_numpy();v['fold']=k;parts.append(v)
        rows.append(dict(fold=k,choice=choice,selected=r.score(v.observed,v.pred),
            incumbent=r.score(v.observed,v.incumbent),original=r.score(v.observed,v.reference)))
        print('OUTER',rows[-1],flush=True)
    v=pd.concat(parts).set_index('index').loc[train['index']].reset_index();v.to_csv(OUT/'nested_predictions.csv',index=False)
    gain=r.bootstrap_gain(train.reset_index(drop=True),v.pred.to_numpy(),v.incumbent.to_numpy())
    result=dict(selected=r.score(v.observed,v.pred),incumbent=r.score(v.observed,v.incumbent),
        original=r.score(v.observed,v.reference),gain_vs_incumbent=gain,folds=rows,
        incumbent_public_score=1365.913,incumbent_score_source='user report',target=1365.500,
        note='Outer clients excluded from fit/selection. All data have previously been explored; no fresh independent test.')
    r.write_json(OUT/'results.json',result);choice=select(train,'full_training');result['final_choice']=choice
    # Preserve the successful CSV and save each successor separately.
    if choice['candidate']<0 or (choice['candidate']==0 and choice['weight']==.5):
        result.update(export=None,recommendation='public incumbent unchanged; no distinct full-training winner')
    elif result['selected']['RMSE']>=result['incumbent']['RMSE']:
        result.update(export=None,recommendation='public incumbent retained; nested selection did not improve it')
    else:
        m=Premium(choice).fit(train);path=ROOT/'models/severity_successor.joblib';joblib.dump(m,path)
        pred=joblib.load(path).details(test);dest=ROOT/'deliverables/candidates/submission_severity_successor.csv'
        pred[['index','pred']].to_csv(dest,index=False);r.valid_submission(dest,test)
        result['export']=dict(file=str(dest.relative_to(ROOT)),sha256=r.digest(dest),kaggle_score=None,model_sha256=r.digest(path))
        result['recommendation']='experimental successor' if gain['interval_95'][0]>0 else 'exploratory successor; uncertain local gain'
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();m=joblib.load(ROOT/'models/severity_successor.joblib')
            m.details(test)[['index','pred']].to_csv(args.output,index=False);r.valid_submission(args.output,test)


if __name__=='__main__':
    from improve_severity import main
    main()
