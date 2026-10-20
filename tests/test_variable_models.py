import sys
from pathlib import Path
import joblib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import variable_models as m

@pytest.fixture(scope='module')
def data():return m.r.load_data()[0].iloc[:2500].copy()

@pytest.mark.parametrize('kind',['ridge','xgb'])
def test_correction_features_serialization(data,tmp_path,kind):
    c=next(c for c in m.catalogue() if c['kind']==kind and c['panel']=='all')
    model=m.Correction(c).fit(data,np.full(len(data),.06),np.full(len(data),1700.))
    x=data.iloc[:30].copy();base=np.full(len(x),1700.)
    for col in ['marque_vehicule','modele_vehicule','type_contrat']:x[col]='__UNSEEN__'
    expected=model.predict(x,base)
    for col in ['index','id_client','id_contrat','id_vehicule','nombre_sinistres','montant_sinistre','sex_conducteur1','sex_conducteur2']:x[col]='FORBIDDEN'
    np.testing.assert_array_equal(expected,model.predict(x,base))
    assert np.isfinite(expected).all() and (expected>0).all()
    dest=tmp_path/'model.joblib';joblib.dump(model,dest)
    np.testing.assert_array_equal(expected,joblib.load(dest).predict(x,base))

def test_feature_anomalies_and_structural_absence(data):
    x=data.iloc[:5].copy();x['poids_vehicule']=0;x['cylindre_vehicule']=0
    x['age_conducteur2']=0;x['anciennete_permis2']=0;x['anciennete_permis1']=111
    z,n,c=m.features(x,'all')
    assert z.zero_weight.eq(1).all() and z.zero_engine.eq(1).all()
    assert z.log_poids.isna().all() and z.log_cylindre.isna().all()
    assert z.second_absent.eq(1).all() and z.age_conducteur2.isna().all()
    assert z.licence_anomaly1.eq(1).all()
    assert not set(z)&{'id_client','index','montant_sinistre','nombre_sinistres','sex_conducteur1','sex_conducteur2'}
