"""Extension métier exploratoire : Insee, puissance et ancienneté.
Les règles n'utilisent aucun montant de sinistre. Le référentiel est embarqué au fit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from gravite_pipeline import SeverityFeatures

ROOT = Path(__file__).resolve().parent

def commune_codes(s):
    """Codes communaux présumés; rattachement documenté des arrondissements PLM."""
    out = s.astype(str).str.upper().str.strip().str.zfill(5)
    mapping = {str(n):'75056' for n in range(75101,75121)}
    mapping.update({str(n):'69123' for n in range(69381,69390)})
    mapping.update({str(n):'13055' for n in range(13201,13217)})
    return out.replace(mapping)

class BusinessFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, source='minimal', geo='none', vehicle='none', interactions=False,
                 reference_path=None):
        self.source = source
        self.geo = geo
        self.vehicle = vehicle
        self.interactions = interactions
        self.reference_path = reference_path

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        path = Path(self.reference_path) if self.reference_path else ROOT/'assets/referentiel_communes_insee_2021.csv'
        self.reference_ = pd.read_csv(path,dtype={'code_commune':str}).set_index('code_commune')
        return self

    def transform(self, X):
        full = SeverityFeatures(profile='raw',clean=True,geography='none').fit_transform(X)
        if self.source == 'minimal':
            z = full[['type_contrat','prix_vehicule','anciennete_vehicule']].copy()
        else:
            z = full.copy()
        if self.geo != 'none':
            codes = commune_codes(X['code_postal'])
            ref = self.reference_
            d3 = codes.map(ref['densite3']).fillna('inconnu')
            d7 = codes.map(ref['libelle_densite7']).fillna('inconnu')
            pop = codes.map(ref['population_2018'])
            if self.geo in ('g3','all'):
                z['geo_densite3'] = d3
            if self.geo in ('g7','all'):
                z['geo_densite7'] = d7
            if self.geo in ('population','all'):
                z['geo_population_log'] = np.log1p(pop)
                z['geo_taille_commune'] = pd.cut(pop,[-np.inf,2000,10000,50000,200000,np.inf],
                  labels=['<2k','2-10k','10-50k','50-200k','>=200k'],right=False).astype(object).fillna('inconnu')
            if self.interactions:
                z['couverture_densite'] = full.type_contrat.astype(str)+'::'+d3.astype(str)
        if self.vehicle != 'none':
            ratio = 1000*full.din_vehicule/full.poids_vehicule.where(full.poids_vehicule.gt(0))
            power = full.din_vehicule
            speed = full.vitesse_vehicule
            age = full.anciennete_vehicule
            mode = self.vehicle
            if mode in ('sport','sport_loose','sport_strict','all'):
                thresholds = {'sport_loose':(120,90,210),'sport':(150,110,230),'sport_strict':(180,130,250)}
                p,r,v = thresholds.get(mode,(150,110,230))
                flag = (power.ge(p)&ratio.ge(r)) | (power.ge(p)&speed.ge(v))
                known = (power.notna()&ratio.notna()) | (power.notna()&speed.notna())
                z['proxy_sport'] = pd.Series(np.where(flag,'profil_performant','autre'),index=X.index).where(known,'inconnu')
                z['puissance_par_tonne_metier'] = ratio
            if mode in ('age','all'):
                z['classe_anciennete_auto'] = pd.cut(age,[-np.inf,5,10,20,30,np.inf],
                       labels=['<5','5-9','10-19','20-29','>=30'],right=False).astype(object).fillna('inconnu')
                z['vehicule_30plus'] = np.where(age.isna(),'inconnu',np.where(age.ge(30),'>=30','<30'))
            if mode in ('price','all'):
                z['segment_valeur_declaree'] = pd.cut(full.prix_vehicule,[-np.inf,15000,30000,50000,np.inf],
                       labels=['<15k','15-30k','30-50k','>=50k'],right=False).astype(object).fillna('inconnu')
            if self.interactions:
                group = z['proxy_sport'] if 'proxy_sport' in z else z.get('classe_anciennete_auto',pd.Series('autre',index=X.index))
                z['couverture_profil_auto'] = full.type_contrat.astype(str)+'::'+group.astype(str)
        for c in z.select_dtypes(exclude='number'):
            z[c] = z[c].astype(object).where(z[c].notna(),'__MANQUANT__')
        return z.replace([np.inf,-np.inf],np.nan)

class BusinessSeverityEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, business=None, params=None):
        self.business = business
        self.params = params

    def fit(self,X,y):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns,dtype=object)
        params = dict(self.params or {})
        allowed = ['n_estimators','max_depth','learning_rate','min_child_weight','reg_lambda','reg_alpha',
                   'subsample','colsample_bytree']
        params = {k:params[k] for k in allowed if k in params}
        self.pipeline_ = Pipeline([
          ('business',BusinessFeatures(**(self.business or {}))),
          ('encoding',ColumnTransformer([
             ('num',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),make_column_selector(dtype_include=np.number)),
             ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,sparse_output=True,dtype=np.float32),make_column_selector(dtype_exclude=np.number))],sparse_threshold=1.0)),
          ('xgb',XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=2026,verbosity=0,**params))])
        self.pipeline_.fit(X,np.asarray(y))
        return self

    def predict(self,X):
        return np.maximum(self.pipeline_.predict(X).astype(float),1.)
