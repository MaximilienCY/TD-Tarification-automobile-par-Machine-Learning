import sys
from pathlib import Path
import numpy as np
import joblib
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import coverage_interactions as m

@pytest.mark.parametrize('block',['vehicle_coverage','driver_usage','both'])
def test_interactions_invariance_and_serialization(block,tmp_path):
    data=m.r.load_data()[0].iloc[:2500].copy();p=np.full(len(data),.06);s=np.full(len(data),1700.)
    model=m.Correction(dict(block=block,alpha=100.,scale=.5)).fit(data,p,s)
    x=data.iloc[:30].copy();base=np.full(len(x),1700.)
    for c in ['type_contrat','utilisation','marque_vehicule']:x[c]='__UNSEEN__'
    expected=model.predict(x,base)
    for c in ['index','id_client','montant_sinistre','nombre_sinistres','sex_conducteur1','sex_conducteur2']:x[c]='FORBIDDEN'
    np.testing.assert_array_equal(expected,model.predict(x,base));assert np.isfinite(expected).all() and (expected>0).all()
    path=tmp_path/'model.joblib';joblib.dump(model,path);np.testing.assert_array_equal(expected,joblib.load(path).predict(x,base))
