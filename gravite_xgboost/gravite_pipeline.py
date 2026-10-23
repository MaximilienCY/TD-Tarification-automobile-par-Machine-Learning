"""Transformations et estimateur de gravité, indépendants des anciens modèles.
Le notebook explique et affiche ce code. Tous les apprentissages restent dans fit.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

FORBIDDEN = {'index', 'id_client', 'id_vehicule', 'id_contrat',
             'nombre_sinistres', 'montant_sinistre', 'sex_conducteur1', 'sex_conducteur2'}
GROUPS = {
    'identite_auto': ['marque_vehicule', 'modele_vehicule'],
    'geographie': ['code_postal'],
    'gestion': ['duree_contrat', 'anciennete_info', 'freq_paiement', 'paiement'],
    'conducteurs': ['conducteur2', 'age_conducteur1', 'age_conducteur2',
                    'anciennete_permis1', 'anciennete_permis2'],
    'technique_auto': ['anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule',
                       'essence_vehicule', 'vitesse_vehicule', 'type_vehicule',
                       'prix_vehicule', 'poids_vehicule', 'debut_vente_vehicule', 'fin_vente_vehicule'],
    'couverture': ['type_contrat'],
    'bonus': ['bonus'],
    'usage': ['utilisation'],
    'redondantes_auto': ['vitesse_vehicule', 'cylindre_vehicule',
                         'debut_vente_vehicule', 'fin_vente_vehicule'],
}

class SeverityFeatures(TransformerMixin, BaseEstimator):
    """Nettoyage déterministe; clusters et médianes appris sur le pli courant."""
    def __init__(self, profile='raw', clean=True, geography='prefix', cluster_k=0, drop=()):
        self.profile = profile
        self.clean = clean
        self.geography = geography
        self.cluster_k = cluster_k
        self.drop = drop

    def _base(self, X):
        z = X.drop(columns=list(FORBIDDEN), errors='ignore').copy()
        if self.clean:
            for col in ['poids_vehicule', 'cylindre_vehicule']:
                z[col + '_invalide'] = z[col].le(0).astype(float)
                z.loc[z[col].le(0), col] = np.nan
            for i in [1, 2]:
                age, permit = f'age_conducteur{i}', f'anciennete_permis{i}'
                absent = z['conducteur2'].eq('No') if i == 2 else pd.Series(False, index=z.index)
                invalid = z[permit].eq(111) | z[permit].gt(z[age]) | z[permit].lt(0)
                z[permit + '_invalide'] = (invalid & ~absent).astype(float)
                z.loc[invalid | absent, permit] = np.nan
                z.loc[absent, age] = np.nan
        # La signification exacte du code n'est pas fournie : préfixe géographique,
        # jamais une densité de population ou un revenu inventé.
        if self.geography == 'prefix':
            c = z['code_postal'].astype(str).str.upper().str.zfill(5)
            z['zone_prefixe'] = np.where(c.str.startswith(('97', '98')), c.str[:3], c.str[:2])
            z = z.drop(columns='code_postal')
        elif self.geography == 'none':
            z = z.drop(columns='code_postal')
        z['modele_vehicule'] = (z['marque_vehicule'].astype(str).str.upper().str.strip()
                                + '::' + z['modele_vehicule'].astype(str).str.upper().str.strip())
        return z

    def _engineer(self, X):
        z = self._base(X)
        profile = self.profile
        if profile in ('family', 'all'):
            # Regroupement syntaxique conservateur : premier token, pas de coût cible.
            z['famille_modele'] = z['modele_vehicule'].str.split(r'[\s/\-]+', regex=True).str[0]
            z = z.drop(columns='modele_vehicule')
        if profile == 'no_model':
            z = z.drop(columns='modele_vehicule')
        if profile in ('technical', 'all', 'interactions'):
            weight = z['poids_vehicule'].where(z['poids_vehicule'].gt(0))
            cyl = z['cylindre_vehicule'].where(z['cylindre_vehicule'].gt(0))
            z['puissance_par_tonne'] = 1000 * z['din_vehicule'] / weight
            z['prix_par_kg'] = z['prix_vehicule'] / weight
            z['puissance_par_litre'] = 1000 * z['din_vehicule'] / cyl
            z['prix_par_age'] = z['prix_vehicule'] / (1 + z['anciennete_vehicule'])
            z['log_prix'] = np.log1p(z['prix_vehicule'])
            z['log_age_vehicule'] = np.log1p(z['anciennete_vehicule'])
            # Seuils métier préfixés, non appris sur les montants de validation.
            z['segment_prix'] = pd.cut(z['prix_vehicule'], [-np.inf,10000,20000,35000,np.inf],
                                        labels=['<10k','10-20k','20-35k','>=35k']).astype(object)
            z['segment_puissance'] = pd.cut(z['din_vehicule'], [-np.inf,75,110,150,np.inf],
                                             labels=['<=75','75-110','110-150','>150']).astype(object)
        if profile in ('driver', 'all'):
            z['age_obtention_permis1'] = z['age_conducteur1'] - z['anciennete_permis1']
            z['ecart_age_conducteurs'] = abs(z['age_conducteur1'] - z['age_conducteur2'])
            z['experience_min'] = z[['anciennete_permis1','anciennete_permis2']].min(axis=1)
            z['conducteur_novice'] = z['anciennete_permis1'].le(3).astype(float)
        if profile in ('interactions', 'all'):
            z['couverture_usage'] = z['type_contrat'].astype(str) + '::' + z['utilisation'].astype(str)
            z['couverture_segment'] = z['type_contrat'].astype(str) + '::' + z['segment_prix'].astype(str)
        if profile == 'core':
            z = z[['type_contrat', 'prix_vehicule', 'anciennete_vehicule', 'bonus', 'utilisation']]
        remove = set()
        for group in self.drop:
            remove.update(GROUPS.get(group, [group]))
        # Une ablation doit enlever aussi les variables dérivées de la source.
        if 'identite_auto' in self.drop:
            remove.add('famille_modele')
        if 'geographie' in self.drop:
            remove.add('zone_prefixe')
        if 'conducteurs' in self.drop:
            remove.update(c for c in z if 'permis' in c or c in
                          ['ecart_age_conducteurs','experience_min','conducteur_novice'])
        if 'redondantes_auto' in self.drop:
            remove.add('puissance_par_litre')
        if 'technique_auto' in self.drop:
            remove.update(['puissance_par_tonne','prix_par_kg','puissance_par_litre','prix_par_age',
                           'log_prix','log_age_vehicule','segment_prix','segment_puissance',
                           'couverture_segment','poids_vehicule_invalide','cylindre_vehicule_invalide'])
        if 'couverture' in self.drop or 'usage' in self.drop:
            remove.add('couverture_usage')
        if 'couverture' in self.drop:
            remove.add('couverture_segment')
        z = z.drop(columns=list(remove), errors='ignore')
        for c in z.select_dtypes(exclude='number'):
            z[c] = z[c].astype(object).where(z[c].notna(), '__MANQUANT__')
        return z.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.cluster_columns_ = ['prix_vehicule','din_vehicule','poids_vehicule','cylindre_vehicule','vitesse_vehicule']
        if self.cluster_k and not {'technique_auto','redondantes_auto'}.intersection(self.drop):
            self.cluster_preprocessor_ = Pipeline([
                ('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
            v = np.log1p(self._base(X)[self.cluster_columns_])
            values = self.cluster_preprocessor_.fit_transform(v)
            self.kmeans_ = KMeans(n_clusters=self.cluster_k, n_init=10, random_state=2026).fit(values)
        return self

    def transform(self, X):
        z = self._engineer(X)
        if self.cluster_k and not {'technique_auto','redondantes_auto'}.intersection(self.drop):
            v = np.log1p(self._base(X)[self.cluster_columns_])
            values = self.cluster_preprocessor_.transform(v)
            z['groupe_auto_cluster'] = self.kmeans_.predict(values).astype(str)
        return z


def make_pipeline(config):
    cfg = dict(config)
    feature_params = {k: cfg[k] for k in ['profile','clean','geography','cluster_k','drop'] if k in cfg}
    numeric = 'passthrough' if cfg.get('imputation') == 'native' else SimpleImputer(
        strategy=cfg.get('imputation', 'median'), add_indicator=True, keep_empty_features=True)
    preprocess = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist',
             min_frequency=cfg.get('min_frequency', 10), sparse_output=True, dtype=np.float32),
             make_column_selector(dtype_exclude=np.number))], sparse_threshold=1.0)
    objective = cfg.get('objective', 'reg:squarederror')
    params = dict(objective=objective, n_estimators=cfg.get('n_estimators', 180),
                  max_depth=cfg.get('max_depth', 2), learning_rate=cfg.get('learning_rate', .04),
                  min_child_weight=cfg.get('min_child_weight', 20),
                  reg_lambda=cfg.get('reg_lambda', 20), reg_alpha=cfg.get('reg_alpha', 0),
                  subsample=cfg.get('subsample', .8), colsample_bytree=cfg.get('colsample_bytree', .85),
                  tree_method='hist', n_jobs=2, random_state=2026, verbosity=0)
    if objective == 'reg:tweedie':
        params['tweedie_variance_power'] = 1.5
    return Pipeline([('features', SeverityFeatures(**feature_params)),
                     ('encoding', preprocess), ('xgb', XGBRegressor(**params))])


class SeverityEstimator(RegressorMixin, BaseEstimator):
    """Pipeline prédictive complète; correction logarithmique interne sans fuite."""
    def __init__(self, config=None):
        self.config = config

    def fit(self, X, y, groups=None):
        cfg = dict(self.config or {})
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        y = np.asarray(y, dtype=float)
        assert np.isfinite(y).all() and (y > 0).all()
        self.cap_ = float(np.quantile(y, cfg['cap'])) if cfg.get('cap') else None
        fit_y = np.minimum(y, self.cap_) if self.cap_ else y.copy()
        mode = cfg.get('target', 'identity')
        self.smear_ = 1.0
        if mode == 'log_smear':
            if groups is None:
                raise ValueError('La correction log nécessite les groupes clients.')
            pred_log = np.empty(len(y))
            inner = GroupKFold(n_splits=3, shuffle=True, random_state=17)
            for tr, va in inner.split(X, fit_y, groups):
                p = make_pipeline(cfg)
                p.fit(X.iloc[tr], np.log(fit_y[tr]))
                pred_log[va] = p.predict(X.iloc[va])
            self.smear_ = float(np.mean(np.exp(np.log(fit_y) - pred_log)))
        self.pipeline_ = make_pipeline(cfg)
        self.pipeline_.fit(X, np.log(fit_y) if mode.startswith('log') else fit_y)
        self.mode_ = mode
        self.fit_target_mean_ = float(y.mean())
        return self

    def predict(self, X):
        pred = self.pipeline_.predict(X)
        if self.mode_.startswith('log'):
            pred = np.exp(pred) * self.smear_
        # Contrainte de domaine fixée a priori; aucune borne apprise sur validation.
        return np.maximum(np.asarray(pred, dtype=float), 1.0)
