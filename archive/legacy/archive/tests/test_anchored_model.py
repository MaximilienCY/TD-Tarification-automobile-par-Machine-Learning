"""The correction must preserve the original two-part pricing contract."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from anchored_search import Anchored


def test_anchor_zero_recovers_reference_and_correction_never_changes_severity(tmp_path):
    train=pd.read_csv(ROOT/'Data/train.csv',dtype={'code_postal':str})
    learn=train.iloc[:1200];evaluation=train.iloc[1500:1520].copy()
    m=Anchored({'depth':1,'trees':5},.25).fit(learn)
    corrected=m.predict_details(evaluation)
    expected_s=m.severity_.predict(evaluation)
    np.testing.assert_array_equal(corrected.montant_pred,expected_s)
    m.weight=0.
    baseline=m.predict_details(evaluation)
    np.testing.assert_array_equal(baseline.proba_pred,m.base_.predict_proba(evaluation)[:,1])
    np.testing.assert_array_equal(baseline.montant_pred,expected_s)
    m.weight=.25
    for col in ['nombre_sinistres','montant_sinistre','index','id_client','id_vehicule','id_contrat','sex_conducteur1','sex_conducteur2']:
        evaluation[col]=999999 if pd.api.types.is_numeric_dtype(evaluation[col]) else 'CHANGED'
    np.testing.assert_array_equal(m.predict_details(evaluation)['pred'],corrected['pred'])
    path=tmp_path/'anchored.joblib';joblib.dump(m,path)
    np.testing.assert_array_equal(joblib.load(path).predict_details(evaluation)['pred'],corrected['pred'])
