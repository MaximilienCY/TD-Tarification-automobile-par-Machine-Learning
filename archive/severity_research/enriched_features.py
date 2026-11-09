"""Recherche élargie de gravité : variables du tableau et anciens blocs réintroduits.
Tous les apprentissages, y compris les encodages hiérarchiques, restent dans fit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from gravite_pipeline import SeverityFeatures, FORBIDDEN
from metier_features import BusinessFeatures, commune_codes

BLOCKS = ('recent','relative_price','technical','drivers','bonus','interactions',
          'geography','vehicle_groups','clusters','sales')
CORE = ['type_contrat','prix_vehicule','anciennete_vehicule']
ROOT = Path(__file__).resolve().parent


def category_keys(X):
    brand = X.marque_vehicule.fillna('__INCONNU__').astype(str).str.upper().str.strip()
    model = brand+'::'+X.modele_vehicule.fillna('__INCONNU__').astype(str).str.upper().str.strip()
    commune = commune_codes(X.code_postal)
    dept = commune.str[:2].where(~commune.str.startswith(('97','98')),commune.str[:3])
    return pd.DataFrame({'marque':brand,'modele':model,'departement':dept,'commune':commune},index=X.index)


class HierarchicalMeanEncoder(BaseEstimator):
    """Moyennes lissées modèle→marque et commune→département, sans données externes."""
    def __init__(self, smoothing=50.0):
        self.smoothing = smoothing

    def fit(self,X,y):
        keys=category_keys(X).reset_index(drop=True)
        y=np.asarray(y,dtype=float)
        if not np.isfinite(y).all():
            raise ValueError('Cible non finie dans un encodage.')
        self.global_mean_=float(y.mean())
        self.maps_={}
        for col,parent in [('marque',None),('departement',None),('modele','marque'),('commune','departement')]:
            f=keys.copy(); f['y']=y
            t=f.groupby(col,observed=True).y.agg(['sum','count'])
            if parent:
                parents=f.groupby(col,observed=True)[parent].first()
                prior=parents.map(self.maps_[parent]).fillna(self.global_mean_)
            else:
                prior=pd.Series(self.global_mean_,index=t.index)
            self.maps_[col]=(t['sum']+self.smoothing*prior)/(t['count']+self.smoothing)
        return self

    def transform(self,X,prefix='gravite'):
        keys=category_keys(X)
        out=pd.DataFrame(index=X.index)
        for col,parent in [('marque',None),('departement',None),('modele','marque'),('commune','departement')]:
            prior=out[prefix+'_'+parent] if parent else self.global_mean_
            out[prefix+'_'+col]=keys[col].map(self.maps_[col]).fillna(prior)
        return out


def grouped_encodings(X,y,groups,mode,portfolio=None,smoothing=50.0):
    """Encodages d'apprentissage hors client + dictionnaires pour les futurs clients.
    Le portefeuille auxiliaire doit appartenir au seul sous-train externe courant.
    """
    groups=np.asarray(groups)
    modes=['gravite','frequence'] if mode=='both' else [mode]
    output=pd.DataFrame(index=X.index)
    encoders={}; audit=[]
    for label in modes:
        out=pd.DataFrame(index=X.index,columns=[label+'_'+c for c in ['marque','departement','modele','commune']],dtype=float)
        splitter=GroupKFold(n_splits=3,shuffle=True,random_state=412)
        for tr,va in splitter.split(X,groups=groups):
            excluded=set(groups[va])
            assert not excluded.intersection(groups[tr])
            if label=='frequence':
                if portfolio is None:
                    raise ValueError('Le taux de sinistres nécessite le portefeuille du sous-train courant.')
                fit_portfolio=portfolio.loc[~portfolio.id_client.isin(excluded)]
                fit_X=fit_portfolio
                fit_y=fit_portfolio.nombre_sinistres.gt(0).to_numpy(dtype=float)
                assert not excluded.intersection(fit_portfolio.id_client)
            else:
                fit_X=X.iloc[tr]; fit_y=np.asarray(y)[tr]
            encoder=HierarchicalMeanEncoder(smoothing).fit(fit_X,fit_y)
            out.iloc[va]=encoder.transform(X.iloc[va],label).to_numpy()
            audit.append({'type':label,'n_train':len(fit_X),'n_validation':len(va),'client_overlap':0})
        if label=='frequence':
            encoders[label]=HierarchicalMeanEncoder(smoothing).fit(portfolio,portfolio.nombre_sinistres.gt(0).to_numpy(dtype=float))
        else:
            encoders[label]=HierarchicalMeanEncoder(smoothing).fit(X,y)
        assert np.isfinite(out.to_numpy()).all()
        output=pd.concat([output,out],axis=1)
    return output,encoders,audit


class EnrichedFeatures(TransformerMixin,BaseEstimator):
    """Variables brutes, créations explicites et anciens blocs, sans encodage cible."""
    def __init__(self,source='minimal',blocks=(),encoding='onehot',log_inputs=False):
        self.source=source
        self.blocks=blocks
        self.encoding=encoding
        self.log_inputs=log_inputs

    def _clean(self,X):
        return SeverityFeatures(profile='raw',clean=True,geography='none').fit_transform(X)

    def fit(self,X,y=None,price_reference=None):
        self.n_features_in_=X.shape[1]
        self.feature_names_in_=np.array(X.columns,dtype=object)
        self.price_reference_rows_=len(price_reference) if price_reference is not None else len(X)
        ref=price_reference if price_reference is not None else X
        age=pd.to_numeric(ref.anciennete_vehicule,errors='coerce')
        price=pd.to_numeric(ref.prix_vehicule,errors='coerce').where(ref.prix_vehicule.gt(0))
        self.global_price_=float(price.median())
        # Médiane par âge exact, ramenée vers la médiane globale pour les âges rares.
        t=pd.DataFrame({'age':age,'prix':price}).groupby('age',observed=True).prix.agg(['median','count'])
        weight=t['count']/(t['count']+30.)
        self.age_prices_=weight*t['median']+(1-weight)*self.global_price_
        clean=self._clean(X)
        cols=['age_conducteur1','din_vehicule','anciennete_vehicule']
        self.centers_=clean[cols].mean()
        self.scales_=clean[cols].std().replace(0,1).fillna(1)
        if 'geography' in self.blocks or 'vehicle_groups' in self.blocks:
            self.business_=BusinessFeatures(source='minimal',geo='all',vehicle='all').fit(X)
        if 'clusters' in self.blocks:
            self.clusterer_=SeverityFeatures(profile='technical',cluster_k=6).fit(X)
        return self

    def transform(self,X):
        full=self._clean(X)
        z=full.copy() if self.source=='complete' else full[CORE].copy()
        if self.source=='complete':
            z['departement']=category_keys(X).departement
        if self.encoding!='onehot':
            z=z.drop(columns=['marque_vehicule','modele_vehicule','departement'],errors='ignore')
        blocks=set(self.blocks)
        if 'recent' in blocks:
            a=full.anciennete_vehicule
            z['vehicule_recent']=a.le(2).astype(float).where(a.notna())
        if 'relative_price' in blocks:
            denominator=full.anciennete_vehicule.map(self.age_prices_).fillna(self.global_price_)
            z['prix_relatif']=full.prix_vehicule/denominator
        if 'technical' in blocks:
            w=full.poids_vehicule.where(full.poids_vehicule.gt(0))
            cyl=full.cylindre_vehicule.where(full.cylindre_vehicule.gt(0))
            z['puissance_poids']=1000*full.din_vehicule/w
            z['puissance_litre']=1000*full.din_vehicule/cyl
            z['prix_par_kg']=full.prix_vehicule/w
        if 'drivers' in blocks:
            present=full.conducteur2.eq('Yes')
            for i in [1,2]:
                age=full[f'age_conducteur{i}']; exp=full[f'anciennete_permis{i}']
                valid_age=age.gt(0)&age.notna()
                valid_exp=exp.ge(0)&exp.notna()
                z[f'jeune_conducteur{i}']=age.lt(25).astype(float).where(valid_age)
                z[f'novice{i}']=exp.lt(3).astype(float).where(valid_exp)
                z[f'age_permis{i}']=(age-exp).where(valid_age&valid_exp)
                if i==2:
                    z.loc[~present,[f'jeune_conducteur{i}',f'novice{i}']]=0.
                    z.loc[~present,f'age_permis{i}']=np.nan
            z['ecart_age_conducteurs']=abs(full.age_conducteur1-full.age_conducteur2)
        if 'bonus' in blocks:
            z['bonus']=full.bonus
            z['bonus_minimal']=full.bonus.eq(.5).astype(float).where(full.bonus.notna())
            z['coefficient_superieur_1']=full.bonus.gt(1).astype(float).where(full.bonus.notna())
        if 'sales' in blocks:
            z['debut_vente_vehicule']=full.debut_vente_vehicule
            z['fin_vente_vehicule']=full.fin_vente_vehicule
        if 'interactions' in blocks:
            standardized=(full[self.centers_.index]-self.centers_)/self.scales_
            young=full.age_conducteur1.lt(25).astype(float).where(full.age_conducteur1.notna())
            z['jeune_x_puissance']=young*full.din_vehicule
            z['age1_x_puissance']=standardized.age_conducteur1*standardized.din_vehicule
            z['age_vehicule_x_puissance']=standardized.anciennete_vehicule*standardized.din_vehicule
            z['age1_x_age_vehicule']=standardized.age_conducteur1*standardized.anciennete_vehicule
        if 'geography' in blocks or 'vehicle_groups' in blocks:
            business=self.business_.transform(X)
            if 'geography' in blocks:
                for col in ['geo_densite3','geo_densite7','geo_population_log','geo_taille_commune']:
                    z[col]=business[col]
            if 'vehicle_groups' in blocks:
                for col in ['proxy_sport','classe_anciennete_auto','vehicule_30plus','segment_valeur_declaree']:
                    z[col]=business[col]
                z['famille_modele']=full.modele_vehicule.str.split(r'[\s/\-]+',regex=True).str[0]
        if 'clusters' in blocks:
            z['groupe_auto_cluster']=self.clusterer_.transform(X).groupe_auto_cluster
        if self.log_inputs:
            for col in ['prix_vehicule','din_vehicule','poids_vehicule','cylindre_vehicule']:
                if col in z:
                    z[col]=np.log1p(z[col].where(z[col].ge(0)))
        for col in z.select_dtypes(exclude='number'):
            z[col]=z[col].astype(object).where(z[col].notna(),'__MANQUANT__')
        assert not set(z.columns)&FORBIDDEN
        return z.replace([np.inf,-np.inf],np.nan)


class EnrichedSeverityEstimator(RegressorMixin,BaseEstimator):
    """Pipeline complète avec encodage d'apprentissage hors client dans chaque fit."""
    def __init__(self,spec=None,params=None,objective='reg:squarederror',smoothing=50.):
        self.spec=spec
        self.params=params
        self.objective=objective
        self.smoothing=smoothing

    def fit(self,X,y,groups,portfolio=None):
        self.n_features_in_=X.shape[1]
        self.feature_names_in_=np.array(X.columns,dtype=object)
        y=np.asarray(y,dtype=float)
        assert np.isfinite(y).all() and (y>0).all()
        if portfolio is not None:
            assert set(groups).issubset(set(portfolio.id_client))
            prices=portfolio
            self.context_clients_=set(portfolio.id_client)
        else:
            prices=X
            self.context_clients_=set(groups)
        spec=dict(self.spec or {})
        self.excluded_=tuple(spec.pop('exclude',()))
        self.features_=EnrichedFeatures(**spec).fit(X,price_reference=prices)
        z=self.features_.transform(X)
        mode=spec.get('encoding','onehot')
        self.encoders_={}; self.encoding_audit_=[]
        if mode!='onehot':
            enc,self.encoders_,self.encoding_audit_=grouped_encodings(X,y,groups,mode,portfolio,self.smoothing)
            z=pd.concat([z,enc],axis=1)
        z=z.drop(columns=list(self.excluded_),errors='ignore')
        self.feature_columns_=z.columns.tolist()
        self.training_rows_=len(y)
        allowed=['n_estimators','max_depth','learning_rate','min_child_weight','reg_lambda','reg_alpha',
                 'subsample','colsample_bytree']
        params={k:v for k,v in dict(self.params or {}).items() if k in allowed}
        encoding=ColumnTransformer([
            ('num',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),make_column_selector(dtype_include=np.number)),
            ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,sparse_output=True,dtype=np.float32),make_column_selector(dtype_exclude=np.number))],sparse_threshold=1.)
        self.pipeline_=Pipeline([('encoding',encoding),('xgb',XGBRegressor(
            objective=self.objective,tree_method='hist',n_jobs=2,random_state=2026,verbosity=0,**params))])
        self.pipeline_.fit(z,y)
        self.training_predictions_=np.maximum(self.pipeline_.predict(z).astype(float),1.)
        return self

    def transformed_features(self,X):
        z=self.features_.transform(X)
        for label,encoder in self.encoders_.items():
            z=pd.concat([z,encoder.transform(X,label)],axis=1)
        return z.drop(columns=list(self.excluded_),errors='ignore')[self.feature_columns_]

    def predict_raw(self,X):
        return np.asarray(self.pipeline_.predict(self.transformed_features(X)),dtype=float)

    def predict(self,X):
        return np.maximum(self.predict_raw(X),1.)
