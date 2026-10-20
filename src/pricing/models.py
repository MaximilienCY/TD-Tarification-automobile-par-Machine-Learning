"""Independent frequency/severity estimators with deterministic preprocessing."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, PoissonRegressor
from .reference import (
    ReferenceFrequence, LogistiqueMetier, GammaMetier, BoostingMetier,
    preparer_metier, construire_preparation, RAW_FEATURES, LOW_CATS, NUM_COMPACT,
)


def tree_frame(frame, compact=False):
    """Targets, identifiers and sex are never model inputs."""
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing predictors: {missing}')
    x = preparer_metier(frame).drop(columns='code_postal')
    if compact:
        x = x[['type_contrat', 'utilisation', 'anciennete_vehicule', 'log_prix',
               'age_conducteur1', 'bonus', 'log_din', 'conducteur2', 'departement']]
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(-999).astype(float)
        else:
            x[col] = x[col].fillna('__MISSING__').astype(str)
    return x


class TreeModel:
    def __init__(self, config, frequency):
        self.config, self.frequency = config, frequency

    def fit(self, X, y, groups=None):
        c = self.config
        x = tree_frame(X, c.get('compact', False))
        self.categorical_ = list(x.select_dtypes(exclude='number').columns)
        if c['kind'] == 'cat':
            cls = CatBoostClassifier if self.frequency else CatBoostRegressor
            self.model_ = cls(
                iterations=c['iterations'], depth=c['depth'], learning_rate=c.get('lr', .04),
                l2_leaf_reg=c.get('l2', 10), loss_function='Logloss' if self.frequency else 'RMSE',
                cat_features=self.categorical_, thread_count=4, random_seed=42,
                verbose=False, allow_writing_files=False, boosting_type='Plain',
            )
        else:
            self.categories_ = {col: sorted(x[col].unique()) for col in self.categorical_}
            x = self._lgb_frame(x)
            cls = LGBMClassifier if self.frequency else LGBMRegressor
            self.model_ = cls(
                objective='binary' if self.frequency else c.get('objective', 'gamma'),
                n_estimators=c['iterations'], num_leaves=c.get('leaves', 7),
                learning_rate=c.get('lr', .025), min_child_samples=c.get('min_leaf', 200),
                reg_lambda=c.get('l2', 10), colsample_bytree=.9,
                n_jobs=4, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True,
            )
        self.model_.fit(x, y)
        return self

    def _lgb_frame(self, x):
        x = x.copy()
        for col, categories in self.categories_.items():
            x[col] = pd.Categorical(x[col], categories=categories)
        return x

    def _frame(self, X):
        x = tree_frame(X, self.config.get('compact', False))
        return self._lgb_frame(x) if self.config['kind'] == 'lgb' else x

    def predict(self, X):
        return np.maximum(self.model_.predict(self._frame(X)), 1e-6)

    def predict_proba(self, X):
        return self.model_.predict_proba(self._frame(X))


class SmoothSeverity:
    """Squared-error and Poisson alternatives to the Gamma severity GLM."""
    def __init__(self, config):
        self.config = config

    def fit(self, X, y, groups=None):
        c = self.config
        nums = ['anciennete_vehicule']
        cats = ['type_contrat', 'utilisation']
        if c.get('compact'):
            nums += ['log_prix', 'age_conducteur1', 'log_din']
            cats += ['conducteur2', 'type_vehicule']
        self.prep_ = construire_preparation(nums, cats, nums if c.get('splines') else [], 20)
        z = self.prep_.fit_transform(preparer_metier(X))
        if c['kind'] == 'ridge':
            self.model_ = Ridge(alpha=c['alpha'])
        else:
            self.model_ = PoissonRegressor(alpha=c['alpha'], max_iter=3000, tol=1e-6)
        # Normalization makes regularization comparable across training samples.
        self.scale_ = float(np.mean(y))
        self.model_.fit(z, np.asarray(y) / self.scale_)
        return self

    def predict(self, X):
        return np.maximum(self.model_.predict(self.prep_.transform(preparer_metier(X))) * self.scale_, 1e-6)


class Constant:
    def fit(self, X, y, groups=None):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)

    def predict_proba(self, X):
        p = self.predict(X)
        return np.column_stack([1-p, p])


def make_frequency(c):
    if c['kind'] == 'reference':
        return ReferenceFrequence()
    if c['kind'] == 'constant':
        return Constant()
    if c['kind'] == 'logistic':
        return LogistiqueMetier(c)
    if c['kind'] == 'hist':
        return BoostingMetier(c)
    return TreeModel(c, frequency=True)


def make_severity(c):
    if c['kind'] == 'gamma':
        return GammaMetier(c)
    if c['kind'] == 'constant':
        return Constant()
    if c['kind'] in ('ridge', 'poisson'):
        return SmoothSeverity(c)
    return TreeModel(c, frequency=False)


def candidates():
    f = [{'name': 'logistic_reference', 'kind': 'reference'},
         {'name': 'frequency_constant', 'kind': 'constant'},
         dict(name='hist_previous', kind='hist', leaves=7, iterations=250,
              min_leaf=200, l2=5, lr=.05, rare=100)]
    for C in [.03, .3]:
        f.append(dict(name=f'logistic_splines_C{C}', kind='logistic', C=C,
                      prix=True, splines=True, interaction=True))
    for depth, iterations, l2 in [(3, 350, 10), (4, 500, 10), (5, 500, 20), (6, 700, 30)]:
        f.append(dict(name=f'cat_frequency_d{depth}', kind='cat', depth=depth,
                      iterations=iterations, l2=l2))
    for leaves, iterations in [(3, 300), (7, 300), (15, 400)]:
        f.append(dict(name=f'lgb_frequency_l{leaves}', kind='lgb', leaves=leaves,
                      iterations=iterations, min_leaf=250))
    g = [{'name': 'gamma_reference', 'kind': 'gamma', 'alpha': .1},
         {'name': 'severity_constant', 'kind': 'constant'}]
    for alpha in [.001, .01, 1.]:
        g.append(dict(name=f'gamma_basic_a{alpha}', kind='gamma', alpha=alpha))
    for kind, alpha in [('gamma', .1), ('gamma', 1.), ('poisson', .1), ('ridge', 100.)]:
        g.append(dict(name=f'{kind}_smooth_a{alpha}', kind=kind, alpha=alpha, compact=True, splines=True))
    for depth, iterations in [(2, 250), (3, 350), (4, 400)]:
        g.append(dict(name=f'cat_severity_d{depth}', kind='cat', depth=depth,
                      iterations=iterations, l2=30, lr=.025, compact=True))
    for objective in ['gamma', 'regression']:
        g.append(dict(name=f'lgb_severity_{objective}', kind='lgb', objective=objective,
                      leaves=3, iterations=200, min_leaf=150, l2=20, compact=True))
    return f, g


class PremiumModel:
    """Two-part export; weighted means retain separate interpretable components."""
    def __init__(self, frequency_configs, severity_configs, frequency_weights, severity_weights):
        self.frequency_configs = frequency_configs
        self.severity_configs = severity_configs
        self.frequency_weights = np.asarray(frequency_weights, dtype=float)
        self.severity_weights = np.asarray(severity_weights, dtype=float)

    def fit(self, X):
        self.fitted_frequency_ = []
        self.fitted_severity_ = []
        for c, w in zip(self.frequency_configs, self.frequency_weights):
            self.fitted_frequency_.append(make_frequency(c).fit(X, X.nombre_sinistres, X.id_client) if w else None)
        claims = X.loc[X.nombre_sinistres.eq(1)]
        for c, w in zip(self.severity_configs, self.severity_weights):
            self.fitted_severity_.append(make_severity(c).fit(claims, claims.montant_sinistre, claims.id_client) if w else None)
        return self

    def predict_details(self, X):
        p = sum(w*m.predict_proba(X)[:, 1] for w,m in zip(self.frequency_weights, self.fitted_frequency_) if w)
        s = sum(w*m.predict(X) for w,m in zip(self.severity_weights, self.fitted_severity_) if w)
        if not np.isfinite(p).all() or not np.isfinite(s).all() or np.any((p<0)|(p>1)) or np.any(s<=0):
            raise ValueError('Invalid frequency or severity predictions')
        return pd.DataFrame({'proba_pred': p, 'montant_pred': s, 'pred': p*s}, index=X.index)

    def predict(self, X):
        return self.predict_details(X)['pred'].to_numpy()
