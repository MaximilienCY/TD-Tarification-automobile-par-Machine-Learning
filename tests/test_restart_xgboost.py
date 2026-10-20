"""Contracts for the independent restart, including unseen categories and grouping."""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from restart_xgboost import features, catalogue, Component, split_groups, valid_submission, calibration, calibrate, score
from pricing.experiment import load_data, digest, PROTECTED, PROTECTED_HASH


@pytest.fixture(scope='module')
def data():
    return load_data()[0].iloc[:1800].copy()


@pytest.mark.parametrize('panel', ['compact','raw','engineered'])
@pytest.mark.parametrize('cleaned', [False, True])
def test_no_target_id_or_sex_inputs(data, panel, cleaned):
    a = features(data, panel, cleaned)
    mutated = data.copy()
    for c in ['index','id_client','id_contrat','id_vehicule','sex_conducteur1','sex_conducteur2','nombre_sinistres','montant_sinistre']:
        mutated[c] = 'FORBIDDEN'
    pd.testing.assert_frame_equal(a, features(mutated, panel, cleaned))


def test_anomaly_semantics(data):
    z = data.iloc[:2].copy()
    z['poids_vehicule'] = 0; z['cylindre_vehicule'] = 0
    z['anciennete_permis1'] = 111; z['age_conducteur1'] = 25
    z['conducteur2'] = 'No'; z['age_conducteur2'] = 0; z['anciennete_permis2'] = 0
    raw, clean = features(z, 'raw', False), features(z, 'raw', True)
    assert raw.poids_vehicule.eq(0).all() and clean.poids_vehicule.isna().all()
    assert clean.anciennete_permis1.isna().all() and clean.anciennete_permis1_invalid.eq(1).all()
    assert clean.age_conducteur2.isna().all() and clean.second_driver_present.eq(0).all()


def test_folds_and_catalogue(data):
    for a,b in split_groups(data, 3, 10):
        assert set(data.iloc[a].id_client).isdisjoint(data.iloc[b].id_client)
    for frequency in [True,False]:
        configs = catalogue(frequency)
        assert len(configs) == 60 and configs == catalogue(frequency)
        assert {(c['feature_set'],c['cleaned']) for c in configs} == {(p,k) for p in ['compact','raw','engineered'] for k in [True,False]}
        assert all(1 <= c['max_depth'] <= 5 and .01 <= c['learning_rate'] <= .08 for c in configs)
    assert {c['objective'] for c in catalogue(False)} == {'reg:gamma','reg:squarederror'}


@pytest.mark.parametrize('frequency', [True,False])
def test_unseen_and_serialization(data, tmp_path, frequency):
    config = catalogue(frequency)[0]
    model = Component(config, frequency, rounds=8).fit(data)
    x = data.iloc[:20].copy(); x['type_contrat'] = 'UNSEEN'; x['modele_vehicule'] = 'UNSEEN'
    p = model.predict(x)
    assert np.isfinite(p).all() and (p>0).all()
    if frequency: assert (p<1).all()
    path = tmp_path/'component.joblib'; joblib.dump(model,path)
    np.testing.assert_array_equal(p,joblib.load(path).predict(x))


def test_submission_alignment_and_protection(data, tmp_path):
    path = tmp_path/'submission.csv'
    pd.DataFrame({'index':data['index'], 'pred':np.ones(len(data))}).to_csv(path,index=False)
    valid_submission(path,data)
    with pytest.raises(AssertionError): valid_submission(path,data.iloc[::-1])
    assert digest(PROTECTED) == PROTECTED_HASH


@pytest.mark.parametrize('frequency', [True,False])
def test_early_stopping_split_and_severity_population(data, monkeypatch, frequency):
    original = Component._fit
    calls = []
    def inspected(self, learn, valid, rounds):
        if valid is not None:
            assert set(learn.id_client).isdisjoint(valid.id_client)
        if not self.frequency:
            assert learn.nombre_sinistres.eq(1).all() and learn.montant_sinistre.gt(0).all()
            if valid is not None: assert valid.nombre_sinistres.eq(1).all()
        calls.append((len(learn),valid is not None))
        return original(self,learn,valid,rounds)
    monkeypatch.setattr(Component,'_fit',inspected)
    m = Component(catalogue(frequency)[0],frequency).fit(data)
    assert 1 <= m.rounds_ <= 3000 and len(calls) == 2 and calls[0][1] and not calls[1][1]
    assert calls[1][0] == (len(data) if frequency else int(data.nombre_sinistres.sum()))


def test_calibration_and_metric_positional_semantics():
    p = np.linspace(.01,.4,100); y = (np.arange(100)%5 == 0).astype(int)
    calibrated = calibrate(calibration(p,y),p)
    assert np.isfinite(calibrated).all() and ((calibrated>0)&(calibrated<1)).all()
    result = score(pd.Series([0.,2.],index=[40,50]),pd.Series([1.,3.],index=[0,1]))
    assert result['MAE'] == 1. and result['bias'] == 1. and result['MSE'] == 1.
