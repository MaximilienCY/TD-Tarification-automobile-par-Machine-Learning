import sys
from pathlib import Path
import numpy as np
import joblib
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import refine_smooth_severity as m

@pytest.fixture(scope='module')
def data():return m.r.load_data()[0].iloc[:2500].copy()

def test_exact_incumbent_correction(data):
    p=np.full(len(data),.06);s=np.full(len(data),1700.)
    old=m.vm.Correction(dict(kind='ridge',panel='all',alpha=10.)).fit(data,p,s)
    new=m.Correction(m.catalogue()[2]).fit(data,p,s)
    np.testing.assert_array_equal(old.predict(data,s),new.predict(data,s))

@pytest.mark.parametrize('kind',['ridge','gamma','poisson'])
def test_new_links_invariance_serialization(data,tmp_path,kind):
    c=next(c for c in m.catalogue() if c['kind']==kind)
    model=m.Correction(c).fit(data,np.full(len(data),.06),np.full(len(data),1700.))
    x=data.iloc[:30].copy();s=np.full(len(x),1700.)
    x['marque_vehicule']='__UNSEEN__';x['modele_vehicule']='__UNSEEN__'
    expected=model.predict(x,s)
    for a in ['id_client','index','montant_sinistre','nombre_sinistres','sex_conducteur1','sex_conducteur2']:x[a]='FORBIDDEN'
    np.testing.assert_array_equal(expected,model.predict(x,s))
    assert np.isfinite(expected).all() and (expected>0).all()
    path=tmp_path/'model.joblib';joblib.dump(model,path);np.testing.assert_array_equal(expected,joblib.load(path).predict(x,s))
