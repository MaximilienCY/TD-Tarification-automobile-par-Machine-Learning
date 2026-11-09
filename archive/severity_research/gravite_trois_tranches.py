"""Trois experts Gamma et orientation probabiliste entre trois coûts positifs."""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from qualitative_features import QualitativeFeatures, QualitativeSeverityEstimator


class ThreeBandSeverityEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, thresholds=(1500., 3000.), spec=None, params=None, gate_params=None):
        self.thresholds = thresholds
        self.spec = spec
        self.params = params
        self.gate_params = gate_params

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        cuts = np.asarray(self.thresholds, dtype=float)
        if (y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all()
                or not (y > 0).all()):
            raise ValueError('La gravité exige un montant strictement positif par ligne.')
        if (cuts.shape != (2,) or not np.isfinite(cuts).all()
                or cuts[0] <= 1 or cuts[1] <= cuts[0]):
            raise ValueError('Deux seuils finis, supérieurs à 1 et strictement croissants sont requis.')
        bands = np.searchsorted(cuts, y, side='right')
        if not np.array_equal(np.unique(bands), np.arange(3)):
            raise ValueError('Les trois tranches doivent être présentes dans le sous-train.')
        self.thresholds_ = cuts.copy()
        self.n_features_in_ = X.shape[1]
        self.training_counts_ = np.bincount(bands, minlength=3).tolist()
        self.band_priors_ = np.asarray(self.training_counts_, dtype=float)/len(y)
        allowed = ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight',
                   'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree']
        source = self.gate_params if self.gate_params is not None else self.params
        gate_params = {k: v for k, v in dict(source or {}).items() if k in allowed}
        self.gate_ = Pipeline([
            ('qualitative', QualitativeFeatures(**deepcopy(self.spec or {}))),
            ('encoding', ColumnTransformer([
                ('num', SimpleImputer(strategy='median', add_indicator=True,
                                      keep_empty_features=True),
                 make_column_selector(dtype_include=np.number)),
                ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                      min_frequency=5, sparse_output=True,
                                      dtype=np.float32),
                 make_column_selector(dtype_exclude=np.number))], sparse_threshold=1.)),
            # Pas de rééquilibrage; probabilités multiclasses apprises sur le train.
            ('xgb', XGBClassifier(objective='multi:softprob', num_class=3,
                                 eval_metric='mlogloss', base_score=0.,
                                 tree_method='hist', n_jobs=2, random_state=2026,
                                 verbosity=0, **gate_params)),
        ])
        self.gate_.fit(X, bands)
        self.experts_ = []
        for band in range(3):
            expert = QualitativeSeverityEstimator(
                deepcopy(self.spec), deepcopy(self.params), objective='reg:gamma')
            positions = np.flatnonzero(bands == band)
            expert.fit(X.iloc[positions], y[positions])
            self.experts_.append(expert)
        self.feature_columns_ = self.gate_.named_steps['qualitative'].transform(X).columns.tolist()
        if any(expert.feature_columns_ != self.feature_columns_ for expert in self.experts_):
            raise ValueError('Les experts et le modèle d’orientation doivent utiliser les mêmes variables.')
        return self

    def predict_components(self, X):
        check_is_fitted(self, ['gate_', 'experts_', 'thresholds_'])
        probabilities = self.gate_.predict_proba(X).astype(float)
        if probabilities.shape != (len(X), 3) or not np.isfinite(probabilities).all():
            raise ValueError('Les probabilités des trois tranches sont invalides.')
        probabilities = np.clip(probabilities, 0., 1.)
        row_sum = probabilities.sum(axis=1, keepdims=True)
        if not (row_sum > 0).all():
            raise ValueError('La somme des probabilités doit être strictement positive.')
        probabilities /= row_sum
        lower = [1., *self.thresholds_]
        upper = [*self.thresholds_, np.inf]
        out = {}
        prediction = np.zeros(len(X), dtype=float)
        for band, expert in enumerate(self.experts_):
            raw = expert.predict(X)
            bound = np.nextafter(upper[band], 0.) if np.isfinite(upper[band]) else np.inf
            cost = np.clip(raw, lower[band], bound)
            out[f'probabilite_tranche_{band+1}'] = probabilities[:, band]
            out[f'cout_tranche_{band+1}'] = cost
            out[f'expert_{band+1}_borne'] = raw != cost
            prediction += probabilities[:, band]*cost
        out['prediction_gravite'] = prediction
        return pd.DataFrame(out, index=X.index)

    def predict(self, X):
        return self.predict_components(X)['prediction_gravite'].to_numpy()
