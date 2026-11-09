from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gravite_tranches import TwoBandSeverityEstimator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def fitted():
    frame = pd.read_csv(ROOT/'Data/train.csv', dtype={'code_postal': str})
    frame = frame.loc[frame.montant_sinistre.gt(0)].head(100).copy()
    model = TwoBandSeverityEstimator(
        threshold=2000.,
        spec={'blocks': ['geographie'], 'drop': ['classe_taille_commune']},
        params={'n_estimators': 5, 'max_depth': 1, 'min_child_weight': 1},
    ).fit(frame, frame.montant_sinistre)
    return frame, model


def test_predictions_combine_both_experts_without_observed_cost(fitted):
    frame, model = fitted
    probe = frame.head(8).copy()
    c = model.predict_components(probe)
    assert c.probabilite_grand.between(0, 1).all()
    assert c.cout_petit.lt(2000).all() and c.cout_grand.ge(2000).all()
    assert np.isfinite(c.prediction_gravite).all() and c.prediction_gravite.gt(0).all()
    expected = (1-c.probabilite_grand)*c.cout_petit+c.probabilite_grand*c.cout_grand
    np.testing.assert_array_equal(model.predict(probe), expected)
    # La cible et les identifiants ne peuvent pas servir à orienter le test.
    altered = probe.copy()
    altered['montant_sinistre'] = np.linspace(-1.e9, 1.e9, len(probe))
    altered['nombre_sinistres'] = 999
    altered['id_client'] = 'autre_client'
    pd.testing.assert_frame_equal(model.predict_components(altered), c)
    assert model.feature_columns_ == [
        'type_contrat', 'prix_vehicule', 'anciennete_vehicule', 'classe_geographique']


def test_serialization_unknown_categories_and_population_invariance(fitted, tmp_path):
    frame, model = fitted
    probe = frame.head(8).copy()
    probe['code_postal'] = '99999'
    probe['type_contrat'] = 'CONTRAT_INCONNU'
    probe['prix_vehicule'] = np.nan
    before = model.predict(probe)
    assert np.isfinite(before).all() and (before > 0).all()
    path = tmp_path/'tranches.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(probe), before)
    original = loaded.predict(frame.head(8))
    for features in [loaded.gate_.named_steps['qualitative'],
                     *[e.pipeline_.named_steps['qualitative'] for e in loaded.experts_]]:
        features.geo_.reference_['population_2018'] = 1.e9
    np.testing.assert_array_equal(loaded.predict(frame.head(8)), original)


def test_rejects_zero_targets_and_missing_training_band(fitted):
    frame, _ = fitted
    with pytest.raises(ValueError, match='strictement positif'):
        TwoBandSeverityEstimator().fit(frame, np.zeros(len(frame)))
    with pytest.raises(ValueError, match='deux tranches'):
        TwoBandSeverityEstimator().fit(frame, np.full(len(frame), 1000.))
