from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gravite_trois_tranches import ThreeBandSeverityEstimator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def fitted():
    frame = pd.read_csv(ROOT/'Data/train.csv', dtype={'code_postal': str})
    frame = frame.loc[frame.montant_sinistre.gt(0)].head(120).copy()
    model = ThreeBandSeverityEstimator(
        spec={'blocks': ['geographie'], 'drop': ['classe_taille_commune']},
        params={'n_estimators': 5, 'max_depth': 1, 'min_child_weight': 1},
    ).fit(frame, frame.montant_sinistre)
    return frame, model


def test_three_probabilities_and_costs_are_combined_without_target(fitted):
    frame, model = fitted
    probe = frame.head(10).copy()
    c = model.predict_components(probe)
    probabilities = c[[f'probabilite_tranche_{i}' for i in (1, 2, 3)]].to_numpy()
    costs = c[[f'cout_tranche_{i}' for i in (1, 2, 3)]].to_numpy()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-12)
    assert c.cout_tranche_1.lt(1500).all()
    assert c.cout_tranche_2.ge(1500).all() and c.cout_tranche_2.lt(3000).all()
    assert c.cout_tranche_3.ge(3000).all()
    np.testing.assert_allclose(model.predict(probe), (probabilities*costs).sum(axis=1))
    altered = probe.copy()
    altered['montant_sinistre'] = np.linspace(-1.e9, 1.e9, len(probe))
    altered['nombre_sinistres'] = 999
    altered['id_client'] = 'autre_client'
    pd.testing.assert_frame_equal(model.predict_components(altered), c)


def test_roundtrip_unknown_categories_and_population_invariance(fitted, tmp_path):
    frame, model = fitted
    probe = frame.head(10).copy()
    probe['code_postal'] = '99999'
    probe['type_contrat'] = 'CONTRAT_INCONNU'
    probe['prix_vehicule'] = np.nan
    before = model.predict(probe)
    assert np.isfinite(before).all() and (before > 0).all()
    path = tmp_path/'trois_tranches.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(probe), before)
    original = loaded.predict(frame.head(10))
    for features in [loaded.gate_.named_steps['qualitative'],
                     *[e.pipeline_.named_steps['qualitative'] for e in loaded.experts_]]:
        features.geo_.reference_['population_2018'] = 1.e9
    np.testing.assert_array_equal(loaded.predict(frame.head(10)), original)


def test_rejects_invalid_thresholds_and_absent_training_band(fitted):
    frame, _ = fitted
    with pytest.raises(ValueError, match='croissants'):
        ThreeBandSeverityEstimator(thresholds=(3000, 1500)).fit(frame, frame.montant_sinistre)
    with pytest.raises(ValueError, match='trois tranches'):
        ThreeBandSeverityEstimator().fit(frame, np.full(len(frame), 1000.))
    with pytest.raises(ValueError, match='strictement positif'):
        ThreeBandSeverityEstimator().fit(frame, np.zeros(len(frame)))
