"""Check new model families preserve the no-target-input prediction contract."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from advanced_search import Candidate


@pytest.mark.parametrize('kind',['glm','cat','lgb','ebm','xgb'])
def test_research_prediction_ignores_targets_ids_and_sex(kind):
    train=pd.read_csv(ROOT/'Data/train.csv',dtype={'code_postal':str})
    learn=train.loc[train.nombre_sinistres.eq(1)].iloc[:200]
    evaluation=train.iloc[200:210].copy()
    if kind=='glm':c=dict(kind=kind,form='log_age',interaction=True,alpha=.01)
    elif kind=='cat':c=dict(kind=kind,depth=2,trees=5,l2=10,objective='Poisson',compact=True)
    elif kind=='lgb':c=dict(kind=kind,leaves=2,trees=5,objective='gamma',compact=True)
    elif kind=='ebm':c=dict(kind=kind,interactions=0,rounds=5,objective='poisson_deviance',compact=True)
    else:c=dict(kind=kind,depth=2,rounds=5,objective='reg:gamma',compact=True)
    cls=Candidate
    if kind in ['ebm','xgb']:
        pytest.importorskip('interpret');pytest.importorskip('xgboost')
        from modern_search import ModernCandidate
        cls=ModernCandidate
    m=cls(c,False).fit(learn,learn.montant_sinistre)
    expected=m.predict(evaluation)
    changed=evaluation.copy()
    for col in ['index','id_client','id_vehicule','id_contrat','sex_conducteur1','sex_conducteur2','nombre_sinistres','montant_sinistre']:
        changed[col]=999999 if pd.api.types.is_numeric_dtype(changed[col]) else 'CHANGED'
    np.testing.assert_array_equal(m.predict(changed),expected)
    assert np.isfinite(expected).all() and (expected>0).all()


def test_modern_export_blends_components_with_read_only_pandas_arrays():
    pytest.importorskip('interpret');pytest.importorskip('xgboost')
    from export_modern import predict
    class Baseline:
        def predict_details(self,X):
            return pd.DataFrame({'proba_pred':[.1,.2],'montant_pred':[100.,200.]})
    class Values:
        def __init__(self,values):self.values=np.asarray(values)
        def predict(self,X):return self.values
    bundle=dict(incumbent=Baseline(),frequency=Values([.2,.4]),severity=Values([200.,400.]),
        choice={'frequency':{'weight':.25},'severity':{'weight':.5}})
    result=predict(bundle,pd.DataFrame({'index':[9,8]}))
    np.testing.assert_allclose(result['pred'],[18.75,75.])
    assert result['index'].tolist()==[9,8]
