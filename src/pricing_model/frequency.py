"""Frequency recipe from projet-simplifie, commit 682b3c5.

50% logistic E + 50% calibrated 20-model XGBoost ensemble. All learned
preprocessing is fitted locally; target encodings use separate client groups.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

SEED = 42
NUMERIC = ['bonus', 'duree_contrat', 'anciennete_info', 'age_conducteur1',
           'age_conducteur2', 'anciennete_permis1', 'anciennete_permis2',
           'anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule',
           'debut_vente_vehicule', 'fin_vente_vehicule', 'vitesse_vehicule',
           'prix_vehicule', 'poids_vehicule']
LOW = ['type_contrat', 'freq_paiement', 'paiement', 'utilisation', 'conducteur2',
       'essence_vehicule', 'type_vehicule']
HIGH = ['marque_vehicule', 'modele_vehicule', 'departement', 'code_postal']
RAW_COLUMNS = NUMERIC + LOW + ['marque_vehicule', 'modele_vehicule', 'code_postal']
LOG_NUMERIC = ['bonus', 'age_conducteur1', 'anciennete_permis2',
               'anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule', 'poids_vehicule']
LOG_CATEGORIES = ['type_contrat', 'utilisation', 'conducteur2', 'essence_vehicule', 'type_vehicule']
IMPUTE = ['poids_vehicule', 'cylindre_vehicule', 'anciennete_vehicule']
DERIVED = ['puissance_poids', 'prix_par_cheval', 'age_permis1', 'jeune_conducteur2', 'novice2']
SETTINGS = [
    (250, dict(max_depth=4, min_child_weight=5., subsample=.7, colsample_bytree=.5, reg_lambda=1., reg_alpha=0., gamma=.5)),
    (300, dict(max_depth=3, min_child_weight=20., subsample=.7, colsample_bytree=.7, reg_lambda=1., reg_alpha=0., gamma=.5)),
    (200, dict(max_depth=4, min_child_weight=5., subsample=.7, colsample_bytree=.7, reg_lambda=1., reg_alpha=0., gamma=0.)),
    (200, dict(max_depth=4, min_child_weight=5., subsample=.8, colsample_bytree=.9, reg_lambda=1., reg_alpha=2., gamma=1.)),
]
SEEDS = [42, 1, 2, 3, 4]


def clean(frame, tree=False):
    x = frame.loc[:, RAW_COLUMNS].copy().reset_index(drop=True)
    x['code_postal'] = x['code_postal'].astype('string')
    x['departement'] = x['code_postal'].str[:2]
    for col in ['poids_vehicule', 'cylindre_vehicule']:
        x[col] = x[col].replace(0, np.nan)
    absent = x['conducteur2'].eq('No')
    if tree:
        x.loc[absent, ['age_conducteur2', 'anciennete_permis2']] = np.nan
    x['puissance_poids'] = x['din_vehicule'] / x['poids_vehicule']
    x['prix_par_cheval'] = x['prix_vehicule'] / x['din_vehicule']
    x['age_permis1'] = x['age_conducteur1'] - x['anciennete_permis1']
    x['jeune_conducteur2'] = ((~absent) & x['age_conducteur2'].lt(26)).astype(int)
    x['novice2'] = ((~absent) & x['anciennete_permis2'].lt(3)).astype(int)
    x['malus'] = x['bonus'].gt(.5).astype(int)
    return x


def target_mapping(keys, y, smoothing, prior):
    stats = y.groupby(keys).agg(['sum', 'count'])
    return (stats['sum'] + smoothing * prior) / (stats['count'] + smoothing)


def client_packets(groups):
    groups = pd.Series(np.asarray(groups))
    unique = groups.unique()
    return groups.map(pd.Series(np.random.RandomState(SEED).randint(0, 5, len(unique)), index=unique)).to_numpy()


def encoded_oof(keys, y, groups, smoothing, fixed_prior):
    packets = client_packets(groups)
    result = pd.Series(np.nan, index=keys.index)
    for packet in range(5):
        fit = packets != packet
        if not (~fit).any():
            continue
        prior = float(y.mean() if fixed_prior else y[fit].mean())
        mapping = target_mapping(keys[fit], y[fit], smoothing, prior)
        result.loc[~fit] = keys[~fit].map(mapping).astype(float).fillna(prior).to_numpy()
    return result


class LogisticFeatures:
    """The 17 conceptual variables of logistic E, with branch-identical scales."""

    def fit_transform(self, frame, y, groups):
        x = clean(frame)
        y = pd.Series(np.asarray(y), index=x.index)
        self.prior_ = float(y.mean())
        self.department_ = target_mapping(x['departement'], y, 1000, self.prior_)
        self.imputation_ = {c: (x.groupby('modele_vehicule')[c].median(),
                                x.groupby('marque_vehicule')[c].median(), float(x[c].median())) for c in IMPUTE}
        self.categories_ = {}
        for col in LOG_CATEGORIES:
            counts = x[col].value_counts()
            self.categories_[col] = (counts[counts.ge(50)].index, counts.index[0])
        self.ratio_median_ = float(x['puissance_poids'].median())
        result = self._table(x)
        result['departement'] = encoded_oof(x['departement'], y, groups, 1000, True)
        self.columns_ = result.columns.tolist()
        self.scaled_ = [c for c in LOG_NUMERIC + ['departement'] if result[c].nunique() > 2]
        self.means_ = result[self.scaled_].mean()
        self.scales_ = result[self.scaled_].std().replace(0, 1).fillna(1)
        self.ratio_mean_ = float(result['puissance_poids'].mean())
        self.ratio_std_ = float(result['puissance_poids'].std()) or 1.
        return self._scale(result)

    def _table(self, x):
        result = x[LOG_NUMERIC + LOG_CATEGORIES].copy()
        for col in IMPUTE:
            model, brand, overall = self.imputation_[col]
            result[col] = result[col].fillna(x['modele_vehicule'].map(model)).fillna(x['marque_vehicule'].map(brand)).fillna(overall)
        for col in ['din_vehicule', 'poids_vehicule', 'cylindre_vehicule']:
            result[col] = np.log1p(result[col])
        result['departement'] = x['departement'].map(self.department_).astype(float).fillna(self.prior_)
        for col, (frequent, reference) in self.categories_.items():
            result[col] = result[col].where(result[col].isin(frequent), 'Autre')
        result = pd.get_dummies(result, columns=LOG_CATEGORIES, dtype=int)
        result = result.drop(columns=[f'{c}_{ref}' for c, (_, ref) in self.categories_.items()], errors='ignore')
        result['puissance_poids'] = x['puissance_poids'].fillna(self.ratio_median_)
        for col in ['malus', 'jeune_conducteur2', 'novice2']:
            result[col] = x[col]
        return result

    def _scale(self, result):
        result = result.reindex(columns=self.columns_, fill_value=0).copy()
        result[self.scaled_] = (result[self.scaled_] - self.means_) / self.scales_
        result['puissance_poids'] = (result['puissance_poids'] - self.ratio_mean_) / self.ratio_std_
        return result.astype(float)

    def transform(self, frame):
        return self._scale(self._table(clean(frame)))


class TreeFeatures:
    def fit_transform(self, frame, y, groups):
        x = clean(frame, tree=True)
        y = pd.Series(np.asarray(y), index=x.index)
        self.prior_ = float(y.mean())
        self.categories_ = {c: sorted(x[c].dropna().unique()) for c in LOW}
        self.rates_ = {c: target_mapping(x[c], y, 300, self.prior_) for c in HIGH}
        self.counts_ = {c: x[c].value_counts() for c in HIGH}
        result = self._table(x)
        for col in HIGH:
            result[f'{col}_taux'] = encoded_oof(x[col], y, groups, 300, False)
        self.columns_ = result.columns.tolist()
        return result

    def _table(self, x):
        result = x[NUMERIC + DERIVED].copy()
        for col, levels in self.categories_.items():
            result[col] = pd.Categorical(x[col].where(x[col].isin(levels)), categories=levels)
        for col in HIGH:
            result[f'{col}_taux'] = x[col].map(self.rates_[col]).astype(float).fillna(self.prior_)
            result[f'{col}_effectif'] = x[col].map(self.counts_[col]).fillna(0).to_numpy()
        return result

    def transform(self, frame):
        return self._table(clean(frame, tree=True)).loc[:, self.columns_]


def fit_trees(frame, y, threads=2):
    models = []
    for trees, params in SETTINGS:
        for seed in SEEDS:
            model = XGBClassifier(objective='binary:logistic', tree_method='hist', enable_categorical=True,
                                  max_cat_to_onehot=8, random_state=seed, nthread=threads,
                                  n_estimators=trees, learning_rate=.02, **params)
            model.fit(frame, y)
            models.append(model)
    return models


def tree_average(models, frame):
    return np.mean([m.predict_proba(frame)[:, 1] for m in models], axis=0)


class FrequencyModel:
    def fit(self, frame, y, groups, threads=2):
        frame = frame.reset_index(drop=True)
        y, groups = np.asarray(y), np.asarray(groups)
        if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
            raise ValueError('Frequency requires two binary target classes.')
        oof_tree, oof_logistic = np.empty(len(y)), np.empty(len(y))
        fold_ids = np.zeros(len(y), dtype=int)
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        with threadpool_limits(limits=threads):
            for fold, (fit, score) in enumerate(folds.split(frame, y, groups), 1):
                assert not set(groups[fit]).intersection(groups[score])
                print(f'Frequency: calibration fold {fold}/5', flush=True)
                linear_prep, tree_prep = LogisticFeatures(), TreeFeatures()
                a = linear_prep.fit_transform(frame.iloc[fit], y[fit], groups[fit])
                linear = LogisticRegression(C=.1, max_iter=2000).fit(a, y[fit])
                oof_logistic[score] = linear.predict_proba(linear_prep.transform(frame.iloc[score]))[:, 1]
                a = tree_prep.fit_transform(frame.iloc[fit], y[fit], groups[fit])
                models = fit_trees(a, y[fit], threads)
                oof_tree[score] = tree_average(models, tree_prep.transform(frame.iloc[score]))
                fold_ids[score] = fold
            calibration = fold_ids != 1
            self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(
                logit(np.clip(oof_tree[calibration], 1e-9, 1-1e-9)).reshape(-1, 1), y[calibration])
            self.calibration_rows_ = int(calibration.sum())
            self.n_fit_ = len(frame)
            print('Frequency: fitting the final logistic E and 20 XGBoost models', flush=True)
            self.linear_prep_ = LogisticFeatures()
            a = self.linear_prep_.fit_transform(frame, y, groups)
            self.linear_ = LogisticRegression(C=.1, max_iter=2000).fit(a, y)
            self.tree_prep_ = TreeFeatures()
            a = self.tree_prep_.fit_transform(frame, y, groups)
            self.trees_ = fit_trees(a, y, threads)
        self.blend_weight_ = .5
        return self

    def predict_proba(self, frame):
        with threadpool_limits(limits=2):
            linear = self.linear_.predict_proba(self.linear_prep_.transform(frame))[:, 1]
            trees = tree_average(self.trees_, self.tree_prep_.transform(frame))
            calibrated = self.calibrator_.predict_proba(logit(np.clip(trees, 1e-9, 1-1e-9)).reshape(-1, 1))[:, 1]
        return self.blend_weight_ * calibrated + (1-self.blend_weight_) * linear
