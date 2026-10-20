import sys
from pathlib import Path
import joblib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import refine_severity as m


@pytest.fixture(scope='module')
def data():return m.r.load_data()[0].iloc[:1800].copy()


@pytest.mark.parametrize('kind',['gamma','poisson','xgb_ratio'])
def test_positive_invariant_predictions(data,tmp_path,kind):
    config=next(c for c in m.catalogue() if c['kind']==kind)
    p=np.full(len(data),.06);s=np.full(len(data),1700.)
    model=m.Severity(config).fit(data,p,s)
    x=data.iloc[:20].copy();x['type_contrat']='UNSEEN'
    a=model.predict(x,np.full(len(x),1700.))
    for c in ['index','id_client','id_contrat','id_vehicule','sex_conducteur1','sex_conducteur2','nombre_sinistres','montant_sinistre']:
        x[c]='FORBIDDEN'
    np.testing.assert_array_equal(a,model.predict(x,np.full(len(x),1700.)))
    assert np.isfinite(a).all() and (a>0).all()
    path=tmp_path/'severity.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(a,joblib.load(path).predict(x,np.full(len(x),1700.)))


def test_zero_shift_unweighted_reproduces_gamma(data):
    p=np.full(len(data),.06);s=np.full(len(data),1700.)
    model=m.Severity(m.catalogue()[0]).fit(data,p,s)
    claims=data[data.nombre_sinistres.eq(1)]
    original=m.GammaMetier({'alpha':.1}).fit(claims,claims.montant_sinistre,claims.id_client)
    np.testing.assert_allclose(model.predict(data,s),original.predict(data),rtol=0,atol=1e-7)


def test_weighted_ratio_loss_identity_and_shift_guard(data):
    p=np.array([.03,.1]);s=np.array([1200.,2400.]);y=np.array([600.,3000.]);ratio=np.array([.9,1.1])
    np.testing.assert_allclose(p*s*s*(y/s-ratio)**2,p*(y-s*ratio)**2)
    c=dict(kind='gamma',shift=1e9,weighted=True,spline=False,alpha=.1)
    with pytest.raises(AssertionError):m.Severity(c).fit(data,np.full(len(data),.1),np.full(len(data),1700.))
