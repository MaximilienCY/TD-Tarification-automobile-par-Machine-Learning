"""Variables dérivées et traitement des manquants du nouveau modèle de gravité."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


NUMERIC_FEATURES = (
    'bonus', 'duree_contrat', 'anciennete_info', 'age_conducteur1',
    'age_conducteur2', 'anciennete_permis1', 'anciennete_permis2',
    'anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule',
    'debut_vente_vehicule', 'fin_vente_vehicule', 'vitesse_vehicule',
    'prix_vehicule', 'poids_vehicule',
)
CATEGORICAL_FEATURES = (
    'type_contrat', 'freq_paiement', 'paiement', 'utilisation', 'conducteur2',
    'essence_vehicule', 'marque_vehicule', 'modele_vehicule', 'type_vehicule',
    'departement',
)
INTERACTION_FEATURE = 'age_puissance_vehicule'
INTERACTION_SOURCES = {
    INTERACTION_FEATURE: ('anciennete_vehicule', 'din_vehicule'),
    'puissance_anciennete_permis1': ('din_vehicule', 'anciennete_permis1'),
    'vitesse_age_conducteur1': ('vitesse_vehicule', 'age_conducteur1'),
}
DERIVED_NUMERIC_FEATURES = tuple(INTERACTION_SOURCES)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DERIVED_NUMERIC_FEATURES
VALID_DEPARTMENTS = frozenset(
    [f'{n:02d}' for n in range(1, 96) if n != 20]
    + ['2A', '2B', '971', '972', '973', '974', '976']
)


def extraire_departement(codes):
    """Extraire un département nominal, sans utiliser de montant de sinistre.

    Le champ du projet contient notamment des codes communaux 2Axxx / 2Bxxx.
    Un code 20xxx ambigu n'est pas affecté arbitrairement à l'un des deux.
    Les codes invalides et les territoires sans code département reconnu restent NA.
    """
    normalized = (codes.astype('string').str.strip().str.upper()
                  .str.replace(r'\.0$', '', regex=True).str.zfill(5))
    valid_format = normalized.str.fullmatch(r'(?:\d{5}|2[AB]\d{3})', na=False)
    department = normalized.str[:2]
    overseas = normalized.str.startswith(('97', '98'), na=False)
    department = department.mask(overseas, normalized.str[:3])
    return department.where(valid_format & department.isin(VALID_DEPARTMENTS)).rename('departement')


def _source_interaction(frame, col):
    values = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if col in {'din_vehicule', 'vitesse_vehicule', 'age_conducteur1'}:
        valid = values.gt(0)
    else:
        valid = values.ge(0)
    if col == 'anciennete_permis1':
        driver_age = pd.to_numeric(frame['age_conducteur1'], errors='coerce')
        valid &= values.ne(111) & values.le(driver_age) & np.isfinite(driver_age)
    return values.where(valid)


def creer_interactions(frame):
    """Produits sans cible, avec sources absentes ou invalides laissées manquantes."""
    sources = {col: _source_interaction(frame, col)
               for pair in INTERACTION_SOURCES.values() for col in pair}
    return pd.DataFrame({name: sources[left] * sources[right]
                         for name, (left, right) in INTERACTION_SOURCES.items()}, index=frame.index)


def interaction_age_puissance(frame):
    """Produit de l'ancienneté et de la puissance DIN ; NA si une source est invalide."""
    return (_source_interaction(frame, 'anciennete_vehicule')
            * _source_interaction(frame, 'din_vehicule')).rename(INTERACTION_FEATURE)


class PreparationGravite(TransformerMixin, BaseEstimator):
    """Médianes du seul apprentissage, catégories manquantes et indicateurs.

    Le résultat reste un DataFrame : les catégories, dont le département,
    ne sont pas transformées en nombres ordonnés. L'encodage du modèle viendra ensuite.
    """

    @staticmethod
    def _clean(X):
        frame = X.copy()
        # Recalculer la variable depuis sa source évite une colonne dérivée périmée.
        frame['departement'] = extraire_departement(frame['code_postal'])
        frame[list(DERIVED_NUMERIC_FEATURES)] = creer_interactions(frame)
        missing_columns = set(MODEL_FEATURES).difference(frame.columns)
        if missing_columns:
            raise ValueError(f'Colonnes absentes : {sorted(missing_columns)}')
        frame = frame.loc[:, list(MODEL_FEATURES)].copy()
        for col in NUMERIC_FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        for col in ['poids_vehicule', 'cylindre_vehicule']:
            frame.loc[frame[col].le(0), col] = np.nan
        frame.loc[frame['din_vehicule'].le(0), 'din_vehicule'] = np.nan
        frame.loc[frame['vitesse_vehicule'].le(0), 'vitesse_vehicule'] = np.nan
        frame.loc[frame['age_conducteur1'].le(0), 'age_conducteur1'] = np.nan
        for suffix in ['1', '2']:
            col = f'anciennete_permis{suffix}'
            invalid = frame[col].eq(111) | frame[col].lt(0) | frame[col].gt(frame[f'age_conducteur{suffix}'])
            frame.loc[invalid, col] = np.nan
        absent = frame['conducteur2'].eq('No')
        frame.loc[absent | frame['age_conducteur2'].le(0), 'age_conducteur2'] = np.nan
        frame.loc[absent, 'anciennete_permis2'] = np.nan
        frame.loc[frame['anciennete_vehicule'].lt(0), 'anciennete_vehicule'] = np.nan
        frame[list(DERIVED_NUMERIC_FEATURES)] = creer_interactions(frame)
        for col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype('string').replace(r'^\s*$', pd.NA, regex=True)
        return frame

    def fit(self, X, y=None):
        frame = self._clean(X)
        if frame.empty:
            raise ValueError('Aucun sinistre positif pour apprendre les médianes.')
        medians = frame[list(NUMERIC_FEATURES)].median()
        self.empty_numeric_features_ = medians.index[medians.isna()].tolist()
        # Repli explicite si une variable entière est manquante dans un sous-train.
        self.medians_ = medians.fillna(0.0)
        self.n_samples_fit_ = len(frame)
        self.feature_names_out_ = list(MODEL_FEATURES) + [
            f'{c}__manquant' for c in NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        frame = self._clean(X)
        flags = frame[list(NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES)].isna().astype('int8').add_suffix('__manquant')
        frame[list(NUMERIC_FEATURES)] = frame[list(NUMERIC_FEATURES)].fillna(self.medians_)
        frame[list(CATEGORICAL_FEATURES)] = frame[list(CATEGORICAL_FEATURES)].fillna('__MANQUANT__')
        # Chaque produit reste cohérent avec ses deux sources après leur imputation.
        for name, (left, right) in INTERACTION_SOURCES.items():
            frame[name] = frame[left] * frame[right]
        return pd.concat([frame, flags], axis=1).loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'medians_')
        return np.asarray(self.feature_names_out_, dtype=object)
