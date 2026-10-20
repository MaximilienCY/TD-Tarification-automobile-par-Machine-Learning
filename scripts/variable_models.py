"""Expanded-variable, strongly pooled severity corrections, evaluated against public incumbent."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer,OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
import improve_severity as old
import refine_severity as previous
import restart_xgboost as r
ROOT=r.ROOT;OUT=ROOT/'reports/variable_models';CACHE=ROOT/'.cache/variable_models'

def catalogue():
    return [dict(kind='ridge',panel=p,alpha=a) for p in ['vehicle','driver','contract','all'] for a in [10.,100.,1000.,10000.]]+[
        dict(kind='xgb',panel='all',depth=d,trees=t) for d in [1,2] for t in [150,400]]

def features(X,panel):
    z=previous.preparer_metier(X)
    z['licence_start_age']=z.age_conducteur1-z.anciennete_permis1
    z['licence_anomaly1']=z.anciennete_permis1.gt(z.age_conducteur1).astype(float)
    z['licence_anomaly2']=z.anciennete_permis2.gt(z.age_conducteur2).astype(float)
    z['zero_weight']=X.poids_vehicule.eq(0).astype(float)
    z['zero_engine']=X.cylindre_vehicule.eq(0).astype(float)
    z['sale_age_gap']=z.debut_vente_vehicule-z.anciennete_vehicule
    absent=z.conducteur2.astype(str).str.lower().isin(['non','no','0','false'])|z.age_conducteur2.eq(0)
    z['second_absent']=absent.astype(float)
    for c in ['age_conducteur2','anciennete_permis2']:z[c]=z[c].mask(absent)
    nums=dict(vehicle=['log_prix','log_din','log_poids','log_cylindre','puissance_par_tonne','vitesse_vehicule','sale_age_gap','zero_weight','zero_engine'],
        driver=['age_conducteur1','age_conducteur2','anciennete_permis1','anciennete_permis2','licence_start_age','log_bonus','second_absent','licence_anomaly1','licence_anomaly2'],
        contract=['duree_contrat','anciennete_info'])
    cats=dict(vehicle=['marque_vehicule','modele_vehicule','essence_vehicule','type_vehicule'],driver=['conducteur2'],contract=['freq_paiement','paiement'])
    panels=list(nums) if panel=='all' else [panel]
    n=['anciennete_vehicule']+sum([nums[p] for p in panels],[])
    c=['type_contrat','utilisation']+sum([cats[p] for p in panels],[])
    if panel=='all':c+=['departement','contrat_usage']
    for col in c:z[col]=z[col].astype('string').fillna('__MISSING__').astype(str)
    return z[n+c],n,c

class Correction:
    def __init__(self,c):self.config=c
    def fit(self,learn,p,s):
        mask=learn.nombre_sinistres.eq(1).to_numpy();claims=learn.loc[mask]
        x,n,c=features(claims,self.config['panel']);self.categories_={}
        target=claims.montant_sinistre.to_numpy()/s[mask]-1
        w=p[mask]*s[mask]**2;w/=w.mean()
        if self.config['kind']=='ridge':
            # Continuous effects are smooth; category coefficients are pooled toward zero.
            binary=[a for a in n if x[a].nunique()<=2];smooth=[a for a in n if a not in binary]
            parts=[('curves',Pipeline([('missing',SimpleImputer(strategy='median',keep_empty_features=True)),
                ('splines',SplineTransformer(n_knots=4,degree=2,include_bias=False,extrapolation='constant'))]),smooth),
                ('categories',OneHotEncoder(min_frequency=15,handle_unknown='infrequent_if_exist',sparse_output=False),c)]
            if binary:parts.append(('flags',SimpleImputer(strategy='constant',fill_value=0),binary))
            self.prep_=ColumnTransformer(parts)
            self.model_=Ridge(alpha=self.config['alpha'],fit_intercept=True)
            self.model_.fit(self.prep_.fit_transform(x),target,sample_weight=w)
        else:
            self.categories_={a:sorted(x[a].unique()) for a in c}
            for a,categories in self.categories_.items():x[a]=pd.Categorical(x[a],categories=categories)
            self.model_=XGBRegressor(objective='reg:squarederror',base_score=0.,max_depth=self.config['depth'],
                n_estimators=self.config['trees'],learning_rate=.025,min_child_weight=100.,reg_lambda=300.,reg_alpha=10.,
                tree_method='hist',enable_categorical=True,n_jobs=4,random_state=r.SEED)
            self.model_.fit(x,target,sample_weight=w)
        return self
    def predict(self,X,base):
        x,_,_=features(X,self.config['panel'])
        if self.config['kind']=='ridge':x=self.prep_.transform(x)
        else:
            for a,categories in self.categories_.items():x[a]=pd.Categorical(x[a].where(x[a].isin(categories)),categories=categories)
        s=base*np.maximum(1+self.model_.predict(x),1e-8)
        assert np.isfinite(s).all() and (s>0).all()
        return s

def key(learn,evaluation):
    h=hashlib.sha256(Path(__file__).read_bytes());h.update(Path(previous.__file__).read_bytes());h.update(Path(old.__file__).read_bytes())
    h.update(json.dumps(r.provenance(),sort_keys=True).encode())
    import importlib.metadata
    h.update(importlib.metadata.version('xgboost').encode())
    for x in [learn,evaluation]:h.update(pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
    return h.hexdigest()

def fold(learn,evaluation):
    path=CACHE/(key(learn,evaluation)+'.npz')
    if path.exists():
        with np.load(path) as z:return z['p'],z['incumbent'],z['s']
    p,base=previous.reference(learn,evaluation);ap,as_=previous.moments(learn)
    incumbent=.5*base+.5*old.Correction(old.catalogue()[0]).fit(learn,ap,as_).predict(evaluation,base)
    s=np.column_stack([Correction(c).fit(learn,ap,as_).predict(evaluation,base) for c in catalogue()])
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,incumbent=incumbent,s=s)
    return p,incumbent,s

def select(frame,label):
    p,inc=np.zeros(len(frame)),np.zeros(len(frame));s=np.zeros((len(frame),len(catalogue())))
    for k,(a,b) in enumerate(r.split_groups(frame,5,r.SEED+11),1):
        p[b],inc[b],s[b]=fold(frame.iloc[a],frame.iloc[b]);print(label,'inner',k,flush=True)
    y=frame.montant_sinistre.to_numpy();rows=[dict(candidate=-1,weight=0.,RMSE=r.score(y,p*inc)['RMSE'])]
    for j,c in enumerate(catalogue()):
        for w in [.25,.5,1.]:rows.append(dict(candidate=j,weight=w,RMSE=r.score(y,p*((1-w)*inc+w*s[:,j]))['RMSE']))
    ranking=pd.DataFrame(rows).sort_values(['RMSE','candidate','weight']);best=ranking.iloc[0]
    choice=dict(candidate=int(best.candidate),weight=float(best.weight),selection_RMSE=float(best.RMSE),incumbent_RMSE=rows[0]['RMSE'])
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);ranking.to_csv(folder/'ranking.csv',index=False)
    r.write_json(folder/'selection.json',dict(choice=choice,catalogue=catalogue()))
    return choice

class Premium:
    def __init__(self,choice):self.choice=choice
    def fit(self,learn):
        self.incumbent=old.Premium(dict(candidate=0,weight=.5)).fit(learn)
        if self.choice['candidate']>=0:
            p,s=previous.moments(learn);self.correction=Correction(catalogue()[self.choice['candidate']]).fit(learn,p,s)
        return self
    def details(self,X):
        v=self.incumbent.details(X);v['incumbent']=v.pred.to_numpy()
        if self.choice['candidate']>=0:
            base=self.incumbent.base.base.predict(X);w=self.choice['weight']
            v['montant_pred']=(1-w)*v.montant_pred+w*self.correction.predict(X,base)
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
    gain=r.bootstrap_gain(train.reset_index(drop=True),v.pred.to_numpy(),v.incumbent.to_numpy())
    result=dict(selected=r.score(v.observed,v.pred),incumbent=r.score(v.observed,v.incumbent),gain=gain,folds=rows,
        note='Feature families informed by previously explored residuals. Nested fitting does not undo that exposure; diagnostic only.')
    r.write_json(OUT/'results.json',result);choice=select(train,'full_training');result['final_choice']=choice
    result['export']=None
    if choice['candidate']>=0 and result['selected']['RMSE']<result['incumbent']['RMSE']:
        model=Premium(choice).fit(train);path=ROOT/'models/variable_models.joblib';joblib.dump(model,path)
        dest=ROOT/'deliverables/candidates/submission_variable_models.csv'
        joblib.load(path).details(test)[['index','pred']].to_csv(dest,index=False);r.valid_submission(dest,test)
        result['export']=dict(file=str(dest.relative_to(ROOT)),sha256=r.digest(dest),model_sha256=r.digest(path),kaggle_score=None)
    r.write_json(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['train','predict']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='train':run()
        else:
            if args.output is None:parser.error('predict requires --output')
            _,test,_,_=r.load_data();joblib.load(ROOT/'models/variable_models.joblib').details(test)[['index','pred']].to_csv(args.output,index=False)
            r.valid_submission(args.output,test)

if __name__=='__main__':
    from variable_models import main
    main()
