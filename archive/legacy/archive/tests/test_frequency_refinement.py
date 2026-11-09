import sys
from pathlib import Path
import joblib
import numpy as np
import pytest
from scipy.optimize import check_grad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import refine_frequency as f


def test_weighted_brier_gradient():
    rng=np.random.default_rng(41);x=rng.normal(size=(25,4));y=(np.arange(25)%4==0).astype(float)
    w=rng.uniform(.5,2,size=25);theta=rng.normal(size=5)
    fun=lambda z:f.brier_loss(z,x,y,w,.1,.3)[0]
    jac=lambda z:f.brier_loss(z,x,y,w,.1,.3)[1]
    assert check_grad(fun,jac,theta)<1e-6


@pytest.mark.parametrize('loss',['logloss','brier'])
def test_probability_bounds_invariance_serialization(tmp_path,loss):
    data=f.r.load_data()[0].iloc[:1800].copy();s=np.linspace(1200,2000,len(data))
    m=f.Frequency(dict(C=.1,power=2,loss=loss)).fit(data,s)
    x=data.iloc[:25].copy();p=m.predict(x)
    for c in ['index','id_client','id_contrat','id_vehicule','nombre_sinistres','montant_sinistre','sex_conducteur1','sex_conducteur2']:
        x[c]='FORBIDDEN'
    np.testing.assert_array_equal(p,m.predict(x));assert ((p>0)&(p<1)).all()
    path=tmp_path/'frequency.joblib';joblib.dump(m,path)
    np.testing.assert_array_equal(p,joblib.load(path).predict(x))
    assert m.optimization_['success']


def test_reference_reproduction():
    data=f.r.load_data()[0].iloc[:1800].copy();s=np.full(len(data),1500.)
    new=f.Frequency(dict(C=.1,power=0,loss='logloss')).fit(data,s)
    old=f.r.ReferenceFrequence().fit(data,data.nombre_sinistres,data.id_client)
    np.testing.assert_allclose(new.predict(data),old.predict_proba(data)[:,1],rtol=0,atol=1e-10)
