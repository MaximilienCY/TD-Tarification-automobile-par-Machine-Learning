import sys
from pathlib import Path
import numpy as np
import joblib
import pytest
from scipy.optimize import check_grad
from scipy.special import expit,logit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import frequency_curves as m

def test_offset_gradient():
    rng=np.random.default_rng(42);z=rng.normal(size=(40,5));y=rng.binomial(1,.1,40);o=np.full(40,-2.)
    b=rng.normal(size=5)
    assert check_grad(lambda a:m.loss_gradient(a,z,y,o,100.)[0],lambda a:m.loss_gradient(a,z,y,o,100.)[1],b)<1e-3

@pytest.mark.parametrize('panel',['vehicle','age_coverage','compact'])
def test_frequency_offset_serialization_and_invariance(panel,tmp_path):
    data=m.r.load_data()[0].iloc[:2500].copy();p=np.full(len(data),.06)
    model=m.Correction(dict(panel=panel,alpha=100.)).fit(data,p)
    x=data.iloc[:30].copy();x['type_contrat']='__UNSEEN__';x['utilisation']='__UNSEEN__'
    expected=model.delta(x)
    for a in ['id_client','index','nombre_sinistres','montant_sinistre','sex_conducteur1','sex_conducteur2']:x[a]='FORBIDDEN'
    np.testing.assert_array_equal(expected,model.delta(x))
    prob=expit(logit(np.full(len(x),.06))+expected);assert ((prob>0)&(prob<1)).all()
    path=tmp_path/'model.joblib';joblib.dump(model,path);np.testing.assert_array_equal(expected,joblib.load(path).delta(x))
