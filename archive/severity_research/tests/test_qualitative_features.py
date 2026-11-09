from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qualitative_features import QualitativeFeatures,QualitativeSeverityEstimator,QUAL_BLOCKS

ROOT=Path(__file__).resolve().parents[2]
def positive():
    x=pd.read_csv(ROOT/'Data/train.csv',dtype={'code_postal':str})
    return x.loc[x.nombre_sinistres.gt(0)].head(180).copy()

def test_groupings_do_not_use_cost_targets_or_validation_prices():
    x=positive()
    a=QualitativeFeatures(blocks=QUAL_BLOCKS,price_method='quantiles').fit(x,x.montant_sinistre)
    b=QualitativeFeatures(blocks=QUAL_BLOCKS,price_method='quantiles').fit(x,np.arange(len(x))*1e6)
    pd.testing.assert_frame_equal(a.transform(x),b.transform(x))
    before=a.price_edges_.copy(); counts=a.model_counts_.copy()
    changed=x.head().copy(); changed.prix_vehicule=1e12; changed.modele_vehicule='INCONNU'
    a.transform(changed)
    np.testing.assert_array_equal(a.price_edges_,before)
    pd.testing.assert_series_equal(a.model_counts_,counts)

def test_absent_driver_is_a_category_not_young_or_novice():
    x=positive().head(4).copy()
    x.conducteur2='No'; x.age_conducteur2=0; x.anciennete_permis2=0
    x.poids_vehicule=0
    f=QualitativeFeatures(blocks=QUAL_BLOCKS).fit(x)
    z=f.transform(x)
    assert z.profil_conducteur2.eq('absent').all()
    assert z.classe_age_conducteur2.eq('absent').all()
    assert z.classe_experience2.eq('absent').all()
    assert z.classe_puissance_poids.eq('inconnu').all()

def test_gravity_only_roundtrip_and_unknown_categories(tmp_path):
    x=positive()
    m=QualitativeSeverityEstimator({'blocks':QUAL_BLOCKS,'representation':'qualitative'},
       {'n_estimators':6,'max_depth':2},objective='reg:gamma').fit(x,x.montant_sinistre)
    assert not any(c.startswith('frequence_') for c in m.feature_columns_)
    probe=x.head(5).copy(); probe.marque_vehicule='INCONNUE'; probe.modele_vehicule='INCONNU'; probe.code_postal='99999'
    probe.prix_vehicule=np.nan; probe.poids_vehicule=0
    p=m.predict(probe)
    assert np.isfinite(p).all() and (p>0).all()
    path=tmp_path/'quali.joblib'; joblib.dump(m,path)
    np.testing.assert_array_equal(joblib.load(path).predict(probe),p)


def test_model_family_preserves_brands_with_spaces():
    x=positive().head(4).copy()
    x['marque_vehicule']='MERCEDES BENZ'
    x['modele_vehicule']=['C220 CDI','C220 TD','E220 CDI','E220']
    f=QualitativeFeatures(blocks=('identite_auto',),model_level='famille',rare_threshold=1).fit(x)
    z=f.transform(x)
    assert z.identite_auto_regroupee.str.startswith('MERCEDES BENZ::').all()
    assert z.identite_auto_regroupee.iloc[0]==z.identite_auto_regroupee.iloc[1]
    assert z.identite_auto_regroupee.iloc[0]!=z.identite_auto_regroupee.iloc[2]
