import sys
from pathlib import Path
import joblib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import improve_severity as m


@pytest.fixture(scope='module')
def data():return m.r.load_data()[0].iloc[:1800].copy()


def test_incumbent_correction_reproduced_exactly(data):
    p=np.linspace(.02,.1,len(data));s=np.full(len(data),1700.)
    old=m.previous.Severity(dict(kind='xgb_ratio',panel='basic',depth=2,trees=240)).fit(data,p,s)
    new=m.Correction(m.catalogue()[0]).fit(data,p,s)
    np.testing.assert_array_equal(old.predict(data,s),new.predict(data,s))
    assert len(m.stable_members())==4 and len(m.choices())==1+4*(len(m.catalogue())+2)


@pytest.mark.parametrize('name',['bagged_240','additive_240','constant_ratio'])
def test_successor_bounds_invariance_serialization(data,tmp_path,name):
    c=next(c for c in m.catalogue() if c['name']==name)
    p=np.linspace(.02,.1,len(data));s=np.full(len(data),1700.)
    model=m.Correction(c).fit(data,p,s);x=data.iloc[:30].copy();base=np.full(len(x),1700.)
    x['type_contrat']='UNSEEN';values=model.predict(x,base)
    for col in ['index','id_client','id_contrat','id_vehicule','nombre_sinistres','montant_sinistre','sex_conducteur1','sex_conducteur2']:x[col]='FORBIDDEN'
    np.testing.assert_array_equal(values,model.predict(x,base));assert (values>0).all() and np.isfinite(values).all()
    path=tmp_path/'correction.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(values,joblib.load(path).predict(x,base))
