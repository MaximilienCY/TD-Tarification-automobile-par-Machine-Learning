"""Création et sélection de variables qualitatives pour la GRAVITÉ uniquement.
Les regroupements ne consultent aucun résultat de sinistre, ni taux de sinistralité.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from gravite_pipeline import SeverityFeatures,FORBIDDEN
from metier_features import BusinessFeatures

QUAL_BLOCKS=('age_auto','valeur_gabarit','performance','identite_auto','conducteurs','contrat','geographie','croisements')

def cut_category(s,bounds,labels):
    return pd.cut(s,bounds,labels=labels,right=False).astype(object).fillna('inconnu')

class QualitativeFeatures(TransformerMixin,BaseEstimator):
    # Valeur accessible aussi aux anciennes instances joblib, avant cet argument.
    drop=()

    def __init__(self,blocks=(),representation='mixte',rare_threshold=10,model_level='modele',
                 price_method='metier',geo_level=3,drop=()):
        self.blocks=blocks
        self.representation=representation
        self.rare_threshold=rare_threshold
        self.model_level=model_level
        self.price_method=price_method
        self.geo_level=geo_level
        self.drop=drop

    def _clean(self,X):
        return SeverityFeatures(profile='raw',clean=True,geography='none').fit_transform(X)

    def _categories(self,X):
        f=self._clean(X); out=pd.DataFrame(index=X.index)
        a=f.anciennete_vehicule
        out['classe_age_auto']=cut_category(a,[-np.inf,3,6,11,21,30,np.inf],
            ['0-2','3-5','6-10','11-20','21-29','30plus'])
        if self.price_method=='quantiles':
            edges=self.price_edges_
            labels=['quantile_'+str(i+1) for i in range(len(edges)-1)]
        else:
            edges=[-np.inf,10000,20000,35000,60000,np.inf]
            labels=['<10k','10-20k','20-35k','35-60k','60kplus']
        out['classe_prix_auto']=cut_category(f.prix_vehicule,edges,labels)
        out['classe_gabarit']=cut_category(f.poids_vehicule,[-np.inf,1000,1400,1800,np.inf],
                         ['<1000kg','1000-1400kg','1400-1800kg','1800kgplus'])
        out['classe_puissance']=cut_category(f.din_vehicule,[-np.inf,75,110,150,np.inf],
                         ['<75DIN','75-110DIN','110-150DIN','150DINplus'])
        ratio=1000*f.din_vehicule/f.poids_vehicule.where(f.poids_vehicule.gt(0))
        out['classe_puissance_poids']=cut_category(ratio,[-np.inf,70,100,130,np.inf],
                         ['<70DIN_t','70-100DIN_t','100-130DIN_t','130DIN_tplus'])
        out['classe_motorisation']=f.essence_vehicule.astype(object)
        out['classe_type_vehicule']=f.type_vehicule.astype(object)
        for i in [1,2]:
            age=f[f'age_conducteur{i}']; exp=f[f'anciennete_permis{i}']
            ac=cut_category(age,[-np.inf,25,40,60,75,np.inf],['<25','25-39','40-59','60-74','75plus'])
            ec=cut_category(exp,[-np.inf,3,10,30,np.inf],['permis<3','permis3-9','permis10-29','permis30plus'])
            profile=pd.Series('etabli_experimente',index=X.index,dtype=object)
            profile.loc[age.lt(25)&exp.ge(3)]='jeune_experimente'
            profile.loc[age.lt(25)&exp.lt(3)]='jeune_permis_recent'
            profile.loc[age.ge(25)&exp.lt(3)]='adulte_permis_recent'
            profile.loc[age.ge(75)&exp.ge(3)]='senior_experimente'
            profile.loc[age.isna()|exp.isna()]='inconnu'
            if i==2:
                absent=f.conducteur2.eq('No')
                ac.loc[absent]='absent'; ec.loc[absent]='absent'; profile.loc[absent]='absent'
            out[f'classe_age_conducteur{i}']=ac
            out[f'classe_experience{i}']=ec
            out[f'profil_conducteur{i}']=profile
        b=f.bonus
        out['classe_bonus']=pd.Series(np.select([b.eq(.5),b.gt(.5)&b.lt(1),b.eq(1),b.gt(1)],
                   ['minimum_0.5','entre_0.5_et_1','egal_1','superieur_1'],default='inconnu'),index=X.index)
        out['usage']=f.utilisation.astype(object)
        out['rythme_paiement']=f.freq_paiement.astype(object)
        out['indicateur_paiement']=f.paiement.astype(object)
        out['couverture']=f.type_contrat.astype(object)
        brand=f.marque_vehicule.fillna('inconnu').astype(str).str.upper().str.strip()
        model=f.modele_vehicule.astype(str)
        model_name=X.modele_vehicule.fillna('inconnu').astype(str).str.upper().str.strip()
        family=brand+'::'+model_name.str.split(r'[\s/\-]+',regex=True).str[0]
        brand_group=brand.where(brand.map(self.brand_counts_).fillna(0).ge(self.rare_threshold),'marques_rares')
        if self.model_level=='marque':
            grouped=brand_group
        elif self.model_level=='famille':
            grouped=family.where(family.map(self.family_counts_).fillna(0).ge(self.rare_threshold),brand_group+'::autres_familles')
        else:
            grouped=model.where(model.map(self.model_counts_).fillna(0).ge(self.rare_threshold),brand_group+'::autres_modeles')
        out['marque_regroupee']=brand_group
        out['identite_auto_regroupee']=grouped
        if 'geographie' in self.blocks or 'croisements' in self.blocks:
            geo=self.geo_.transform(X)
            out['classe_geographique']=geo.geo_densite7 if self.geo_level==7 else geo.geo_densite3
            out['classe_taille_commune']=geo.geo_taille_commune
        return out.astype(object).fillna('inconnu')

    def fit(self,X,y=None):
        self.n_features_in_=X.shape[1]
        f=self._clean(X)
        brand=f.marque_vehicule.fillna('inconnu').astype(str).str.upper().str.strip()
        self.brand_counts_=brand.value_counts()
        self.model_counts_=f.modele_vehicule.astype(str).value_counts()
        model_name=X.modele_vehicule.fillna('inconnu').astype(str).str.upper().str.strip()
        self.family_counts_=(brand+'::'+model_name.str.split(r'[\s/\-]+',regex=True).str[0]).value_counts()
        edges=np.unique(np.quantile(f.prix_vehicule.dropna(),[0,.25,.5,.75,1]))
        edges[0]=-np.inf; edges[-1]=np.inf
        self.price_edges_=edges
        if 'geographie' in self.blocks or 'croisements' in self.blocks:
            self.geo_=BusinessFeatures(geo='all').fit(X)
        q=self._categories(X)
        self.cross_counts_={}
        for col in ['classe_age_auto','classe_puissance_poids','usage','classe_geographique']:
            if col in q:
                self.cross_counts_[col]=(q.couverture.astype(str)+'::'+q[col].astype(str)).value_counts()
        return self

    def transform(self,X):
        f=self._clean(X); q=self._categories(X)
        if self.representation=='brutes_plus':
            z=f.drop(columns=['marque_vehicule','modele_vehicule'],errors='ignore').copy()
        elif self.representation=='qualitative':
            z=f[['type_contrat']].copy()
        else:
            z=f[['type_contrat','prix_vehicule','anciennete_vehicule']].copy()
        blocks=set(self.blocks)
        groups={
          'age_auto':['classe_age_auto'],
          'valeur_gabarit':['classe_prix_auto','classe_gabarit'],
          'performance':['classe_puissance','classe_puissance_poids','classe_motorisation','classe_type_vehicule'],
          'identite_auto':['marque_regroupee','identite_auto_regroupee'],
          'conducteurs':['classe_age_conducteur1','classe_experience1','profil_conducteur1',
                        'classe_age_conducteur2','classe_experience2','profil_conducteur2'],
          'contrat':['classe_bonus','usage','rythme_paiement','indicateur_paiement'],
          'geographie':['classe_geographique','classe_taille_commune'],
        }
        for group,columns in groups.items():
            if group in blocks:
                for col in columns: z[col]=q[col]
        if 'croisements' in blocks:
            for col,counts in self.cross_counts_.items():
                value=q.couverture.astype(str)+'::'+q[col].astype(str)
                z['couverture_x_'+col]=value.where(value.map(counts).fillna(0).ge(self.rare_threshold),q.couverture.astype(str)+'::combinaisons_rares')
        for col in z.select_dtypes(exclude='number'):
            z[col]=z[col].astype(object).where(z[col].notna(),'inconnu')
        # Retrait individuel après création des catégories; les anciens modèles
        # sérialisés sans cet attribut conservent leur comportement.
        z=z.drop(columns=list(getattr(self,'drop',())),errors='ignore')
        assert not set(z)&FORBIDDEN
        return z

class QualitativeSeverityEstimator(RegressorMixin,BaseEstimator):
    def __init__(self,spec=None,params=None,objective='reg:squarederror'):
        self.spec=spec
        self.params=params
        self.objective=objective

    def fit(self,X,y):
        self.n_features_in_=X.shape[1]
        y=np.asarray(y,dtype=float)
        assert (y>0).all() and np.isfinite(y).all()
        params={k:v for k,v in dict(self.params or {}).items() if k in
           ['n_estimators','max_depth','learning_rate','min_child_weight','reg_lambda','reg_alpha','subsample','colsample_bytree']}
        self.pipeline_=Pipeline([
          ('qualitative',QualitativeFeatures(**(self.spec or {}))),
          ('encoding',ColumnTransformer([
            ('num',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),make_column_selector(dtype_include=np.number)),
            ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=5,sparse_output=True,dtype=np.float32),make_column_selector(dtype_exclude=np.number))],sparse_threshold=1.)),
          ('xgb',XGBRegressor(objective=self.objective,tree_method='hist',n_jobs=2,random_state=2026,verbosity=0,**params))])
        self.pipeline_.fit(X,y)
        z=self.pipeline_.named_steps['qualitative'].transform(X)
        self.feature_columns_=z.columns.tolist()
        self.qualitative_columns_=z.select_dtypes(exclude='number').columns.tolist()
        return self

    def predict(self,X):
        return np.maximum(self.pipeline_.predict(X).astype(float),1.)
