"""Contrôles des fuites et des cas métier sensibles de la nouvelle recherche."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enriched_features import (EnrichedFeatures,EnrichedSeverityEstimator,grouped_encodings,
                               category_keys,FORBIDDEN,BLOCKS)

ROOT=Path(__file__).resolve().parents[2]

def sample_data():
    portfolio=pd.read_csv(ROOT/'Data/train.csv',dtype={'code_postal':str}).head(12000)
    pos=portfolio.loc[portfolio.nombre_sinistres.gt(0)].copy()
    return portfolio,pos


def test_cost_encoding_excludes_all_contracts_of_held_client():
    _,x=sample_data()
    x=x.head(90).copy()
    # Plusieurs contrats du même client doivent sortir ensemble de l'encodage.
    groups=np.array(['client_'+str(i//3) for i in range(len(x))])
    y=x.montant_sinistre.to_numpy()
    before,_,audit=grouped_encodings(x,y,groups,'gravite')
    changed=y.copy(); changed[groups==groups[0]]=1e8
    after,_,_=grouped_encodings(x,changed,groups,'gravite')
    np.testing.assert_allclose(before.loc[groups==groups[0]],after.loc[groups==groups[0]])
    assert all(a['client_overlap']==0 for a in audit)
    assert np.abs(before.to_numpy()-after.to_numpy()).sum()>0


def test_frequency_encoding_excludes_all_held_client_contracts():
    portfolio,x=sample_data(); x=x.head(90).copy()
    g=x.id_client.to_numpy(); y=x.montant_sinistre.to_numpy()
    # Modifier tous les résultats du client ne doit pas modifier son propre encodage hors pli.
    before,_,_=grouped_encodings(x,y,g,'frequence',portfolio)
    altered=portfolio.copy()
    altered.loc[altered.id_client.eq(g[0]),'nombre_sinistres']=0
    after,_,_=grouped_encodings(x,y,g,'frequence',altered)
    np.testing.assert_allclose(before.loc[g==g[0]],after.loc[g==g[0]])


def test_absent_secondary_driver_and_invalid_licence_are_not_young_novice():
    _,x=sample_data(); x=x.head(3).copy()
    x['conducteur2']='No'; x['age_conducteur2']=0; x['anciennete_permis2']=0
    x['anciennete_permis1']=111
    z=EnrichedFeatures(blocks=('drivers',)).fit_transform(x)
    assert z.jeune_conducteur2.eq(0).all() and z.novice2.eq(0).all()
    assert z.age_permis1.isna().all() and z.age_permis2.isna().all()
    assert z.novice1.isna().all()


def test_relative_price_statistics_do_not_learn_from_transform():
    portfolio,x=sample_data()
    f=EnrichedFeatures(blocks=('relative_price',)).fit(x,price_reference=portfolio)
    before=f.age_prices_.copy(); global_before=f.global_price_
    probe=x.head(3).copy(); probe['prix_vehicule']=1e9; probe['anciennete_vehicule']=999
    z=f.transform(probe)
    pd.testing.assert_series_equal(f.age_prices_,before)
    assert f.global_price_==global_before
    np.testing.assert_allclose(z.prix_relatif,1e9/global_before)


def test_complete_model_roundtrip_unknowns_and_forbidden_invariance(tmp_path):
    portfolio,x=sample_data(); x=x.head(150).copy()
    for objective in ['reg:squarederror','reg:gamma']:
        m=EnrichedSeverityEstimator({'source':'complete','blocks':BLOCKS,'encoding':'both'},
             {'n_estimators':8,'max_depth':2,'min_child_weight':10},objective=objective).fit(
                 x,x.montant_sinistre,x.id_client,portfolio)
        probe=x.head(5).copy(); expected=m.predict(probe)
        probe['montant_sinistre']=1e12; probe['nombre_sinistres']=99; probe['id_client']='FAUX'
        np.testing.assert_array_equal(m.predict(probe),expected)
        assert not set(m.feature_columns_)&FORBIDDEN
        probe['code_postal']='99999'; probe['marque_vehicule']='INCONNUE'; probe['modele_vehicule']='INCONNU'
        probe['poids_vehicule']=0.; probe['cylindre_vehicule']=0.; probe['anciennete_vehicule']=np.nan
        assert np.isfinite(m.predict(probe)).all() and (m.predict(probe)>0).all()
        path=tmp_path/(objective.split(':')[1]+'.joblib'); joblib.dump(m,path)
        np.testing.assert_array_equal(joblib.load(path).predict(probe),m.predict(probe))
