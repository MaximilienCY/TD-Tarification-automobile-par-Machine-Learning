"""Gravité à deux tranches : orientation probabiliste et deux experts Gamma.

Les tranches sont connues seulement à l'entraînement. À la prédiction, leurs
probabilités dépendent exclusivement des caractéristiques du contrat.
"""
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


class TwoBandSeverityEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, threshold=2000., spec=None, params=None, gate_params=None):
        self.threshold = threshold
        self.spec = spec
        self.params = params
        self.gate_params = gate_params

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        threshold = float(self.threshold)
        if (y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all()
                or not (y > 0).all()):
            raise ValueError('La gravité exige un montant strictement positif par ligne.')
        if not np.isfinite(threshold) or threshold <= 1:
            raise ValueError('Le seuil des deux tranches doit être fini et supérieur à 1.')
        bands = (y >= threshold).astype(int)
        if np.unique(bands).size != 2:
            raise ValueError('Les deux tranches doivent être présentes dans le sous-train.')
        self.threshold_ = threshold
        self.n_features_in_ = X.shape[1]
        self.training_counts_ = np.bincount(bands, minlength=2).tolist()
        self.large_prior_ = float(bands.mean())
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
            # Pas de rééquilibrage des classes : on veut estimer leur probabilité.
            ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                 base_score=self.large_prior_, tree_method='hist',
                                 n_jobs=2, random_state=2026, verbosity=0,
                                 **gate_params)),
        ])
        self.gate_.fit(X, bands)
        self.experts_ = []
        for band in (0, 1):
            mask = bands == band
            expert = QualitativeSeverityEstimator(
                deepcopy(self.spec), deepcopy(self.params), objective='reg:gamma')
            expert.fit(X.iloc[np.flatnonzero(mask)], y[mask])
            self.experts_.append(expert)
        self.feature_columns_ = self.gate_.named_steps['qualitative'].transform(X).columns.tolist()
        if any(expert.feature_columns_ != self.feature_columns_ for expert in self.experts_):
            raise ValueError('Les experts et le modèle d’orientation doivent utiliser les mêmes variables.')
        return self

    def predict_components(self, X):
        check_is_fitted(self, ['gate_', 'experts_', 'threshold_'])
        probability = self.gate_.predict_proba(X)[:, 1].astype(float)
        raw_small = self.experts_[0].predict(X)
        raw_large = self.experts_[1].predict(X)
        # Une moyenne conditionnelle à une tranche doit appartenir à cette tranche.
        small = np.clip(raw_small, 1., np.nextafter(self.threshold_, 0.))
        large = np.maximum(raw_large, self.threshold_)
        prediction = (1. - probability) * small + probability * large
        return pd.DataFrame({
            'probabilite_grand': probability,
            'cout_petit': small,
            'cout_grand': large,
            'prediction_gravite': prediction,
            'expert_petit_borne': raw_small != small,
            'expert_grand_borne': raw_large != large,
        }, index=X.index)

    def predict(self, X):
        return self.predict_components(X)['prediction_gravite'].to_numpy()
