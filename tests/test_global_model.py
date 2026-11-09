from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pricing_model.frequency import LogisticFeatures, TreeFeatures, encoded_oof
from pricing_model.global_model import GlobalModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def sample():
    return pd.read_csv(ROOT / 'Data/train.csv', dtype={'code_postal': 'string'}).head(2000)


def test_preparation_unseen_categories_and_input_copy(sample):
    original = sample.copy(deep=True)
    evaluation = sample.tail(10).copy()
    evaluation['marque_vehicule'] = 'unseen-brand'
    evaluation['modele_vehicule'] = 'unseen-model'
    evaluation['type_contrat'] = 'unseen-cover'
    evaluation['code_postal'] = '00000'
    for cls in [LogisticFeatures, TreeFeatures]:
        prep = cls()
        fit = prep.fit_transform(sample, sample.nombre_sinistres, sample.id_client)
        result = prep.transform(evaluation)
        assert result.columns.tolist() == fit.columns.tolist()
        assert len(result) == len(evaluation)
        if cls is LogisticFeatures:
            assert np.isfinite(result.to_numpy()).all()
    pd.testing.assert_frame_equal(sample, original)


def test_target_encoding_excludes_whole_client():
    keys = pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'])
    y = pd.Series([0, 1, 0, 0, 1, 1, 0, 1])
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
    initial = encoded_oof(keys, y, groups, 300, False)
    changed = y.copy(); changed.iloc[:2] = 1
    result = encoded_oof(keys, changed, groups, 300, False)
    np.testing.assert_allclose(initial.iloc[:2], result.iloc[:2])


def test_saved_model_targets_order_and_product(sample):
    model = joblib.load(ROOT / 'models/global_model.joblib')
    assert isinstance(model, GlobalModel)
    frame = sample.tail(25).copy()
    result = model.predict_components(frame)
    np.testing.assert_allclose(result.prime_predite, result.probabilite_sinistre * result.gravite_predite, rtol=0, atol=0)
    frame[['nombre_sinistres', 'montant_sinistre']] = -999
    pd.testing.assert_frame_equal(result, model.predict_components(frame))
    reverse = model.predict_components(frame.iloc[::-1])
    pd.testing.assert_frame_equal(result.iloc[::-1].reset_index(drop=True), reverse)
    manifest = json.loads((ROOT / 'models/global_model.json').read_text())
    assert model.frequency.n_fit_ == manifest['frequency']['fit_contracts'] == 50000
    assert manifest['severity']['refitted'] is True
    assert manifest['severity']['fit_positive_claims'] == 2917
    assert manifest['production_fit']['test_rows_used_for_fit'] == 0
    assert manifest['validation']['training_contracts'] == 39964


def test_duplicate_index_rejected(sample):
    model = joblib.load(ROOT / 'models/global_model.joblib')
    duplicate = sample.head(2).copy(); duplicate['index'] = 0
    with pytest.raises(ValueError, match='unique'):
        model.predict(duplicate)
