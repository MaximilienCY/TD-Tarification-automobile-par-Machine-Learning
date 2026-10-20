"""Behavioural checks for leakage, robustness, persistence and submission contracts."""
import joblib
import numpy as np
import pandas as pd
import pytest
from pricing.experiment import load_data, validate_submission, PROTECTED, PROTECTED_HASH, digest
from pricing.models import PremiumModel, TreeModel, tree_frame


@pytest.fixture(scope='module')
def data():
    train,test,_,_=load_data()
    return train.iloc[:3000],test.iloc[:20]


@pytest.fixture(scope='module')
def fitted(data):
    model=PremiumModel([{'name':'reference','kind':'reference'}],
                       [{'name':'gamma','kind':'gamma','alpha':.1}],[1.],[1.])
    return model.fit(data[0])


def test_protected_notebook():
    assert digest(PROTECTED)==PROTECTED_HASH


def test_group_partition_and_targets():
    train,test,dev,holdout=load_data()
    assert set(dev.id_client).isdisjoint(holdout.id_client)
    assert len(dev)+len(holdout)==len(train)
    assert len(test)==50000


def test_targets_identifiers_and_sex_cannot_change_predictions(data,fitted):
    x=data[0].iloc[:12].copy()
    expected=fitted.predict(x)
    for c in ['nombre_sinistres','montant_sinistre','index','id_client','id_vehicule','id_contrat',
              'sex_conducteur1','sex_conducteur2']:
        x[c]='arbitrary irrelevant value'
    np.testing.assert_allclose(fitted.predict(x),expected)
    assert not set(['nombre_sinistres','montant_sinistre','index','id_client']) & set(tree_frame(x).columns)


def test_unknown_categories_missing_values_and_component_identity(data,fitted):
    x=data[1].copy()
    x.loc[x.index[0],'type_contrat']='unseen_contract'
    x.loc[x.index[0],'modele_vehicule']='unseen_model'
    x.loc[x.index[1],['poids_vehicule','cylindre_vehicule','anciennete_vehicule']]=np.nan
    detail=fitted.predict_details(x)
    assert np.isfinite(detail.to_numpy()).all()
    assert detail.proba_pred.between(0,1).all() and detail.montant_pred.gt(0).all()
    np.testing.assert_allclose(detail.pred,detail.proba_pred*detail.montant_pred)


def test_serialization(data,fitted,tmp_path):
    path=tmp_path/'model.joblib'
    joblib.dump(fitted,path)
    np.testing.assert_allclose(joblib.load(path).predict(data[1]),fitted.predict(data[1]))


@pytest.mark.parametrize('kind',['cat','lgb'])
def test_new_tree_models_ignore_targets(data,kind):
    train,test=data
    c=dict(kind=kind,depth=2,iterations=10,leaves=3,min_leaf=30)
    model=TreeModel(c,True).fit(train,train.nombre_sinistres,train.id_client)
    p=model.predict_proba(test)
    dirty=test.assign(nombre_sinistres=1,montant_sinistre=1000000,id_client='new')
    np.testing.assert_allclose(p,model.predict_proba(dirty))
    dirty.loc[dirty.index[0],'modele_vehicule']='unseen'
    assert np.isfinite(model.predict_proba(dirty)).all()


def test_submission_rejects_misalignment_and_invalid_values(data):
    test=data[1]
    good=pd.DataFrame({'index':test['index'],'pred':100.})
    validate_submission(good,test)
    with pytest.raises(AssertionError):validate_submission(good.iloc[::-1],test)
    for invalid in [-1.,np.nan,np.inf]:
        bad=good.copy();bad.loc[bad.index[0],'pred']=invalid
        with pytest.raises(AssertionError):validate_submission(bad,test)
