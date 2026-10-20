"""Descriptive out-of-fold residual audit; never an independent feature-selection test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import restart_xgboost as r
ROOT=r.ROOT;OUT=ROOT/'reports/variable_reanalysis'
OUT.mkdir(parents=True,exist_ok=True)
train,test,_,_=r.load_data()
v=pd.read_csv(ROOT/'reports/severity_successor/nested_predictions.csv')
assert np.array_equal(train['index'],v['index'])
frame=train.copy();frame['p']=v.proba_pred;frame['s']=v.incumbent/v.proba_pred
frame['premium']=v.incumbent;frame['fold']=v.fold
frame['department']=frame.code_postal.astype(str).str[:2]
frame['zero_weight']=frame.poids_vehicule.eq(0)
frame['zero_engine']=frame.cylindre_vehicule.eq(0)
frame['licence_anomaly1']=frame.anciennete_permis1.gt(frame.age_conducteur1)
frame['licence_anomaly2']=frame.anciennete_permis2.gt(frame.age_conducteur2)
frame['power_weight']=frame.din_vehicule/frame.poids_vehicule.replace(0,np.nan)
frame['sale_age_gap']=frame.debut_vente_vehicule-frame.anciennete_vehicule
frame['licence_start_age']=frame.age_conducteur1-frame.anciennete_permis1
largest=set(frame.nlargest(25,'montant_sinistre')['index'])
rows=[];foldrows=[]
variables=r.RAW_FEATURES+['department','zero_weight','zero_engine','licence_anomaly1','licence_anomaly2','power_weight','sale_age_gap','licence_start_age']
for c in variables:
    s=frame[c]
    if pd.api.types.is_numeric_dtype(s) and s.nunique()>10:level=pd.qcut(s,6,duplicates='drop').astype('string')
    else:level=s.astype('string').where(s.isin(s.value_counts().head(20).index),'__OTHER__')
    level=level.fillna('__MISSING__')
    for label,z in frame.assign(level=level).groupby('level',observed=True):
        claims=z[z.nombre_sinistres.eq(1)]
        signs=[]
        for k,a in z.groupby('fold'):
            q=a[a.nombre_sinistres.eq(1)]
            delta=float(np.sum(q.p*(q.montant_sinistre-q.s))/np.sum(q.p*q.s)) if len(q) else np.nan
            signs.append(delta)
            foldrows.append(dict(variable=c,level=str(label),fold=int(k),rows=len(a),claims=len(q),weighted_severity_relative_residual=delta))
        weighted=float(np.sum(claims.p*(claims.montant_sinistre-claims.s))/np.sum(claims.p*claims.s)) if len(claims) else np.nan
        diagnostic=claims.loc[~claims['index'].isin(largest)]
        no_tail=float(np.sum(diagnostic.p*(diagnostic.montant_sinistre-diagnostic.s))/np.sum(diagnostic.p*diagnostic.s)) if len(diagnostic) else np.nan
        rows.append(dict(variable=c,meaning=r.MEANINGS.get(c,'Derived diagnostic'),level=str(label),rows=len(z),clients=z.id_client.nunique(),
            claims=len(claims),claim_rate=z.nombre_sinistres.mean(),predicted_rate=z.p.mean(),
            conditional_severity=claims.montant_sinistre.mean(),predicted_claim_severity=claims.s.mean(),
            weighted_severity_relative_residual=weighted,mean_cost=z.montant_sinistre.mean(),mean_premium=z.premium.mean(),
            top25_claims=int(claims['index'].isin(largest).sum()),severity_residual_without_top25_diagnostic=no_tail,
            frequency_relative_residual=float(z.nombre_sinistres.sum()/z.p.sum()-1),
            severity_direction_consistent=len(signs)==3 and (all(x>0 for x in signs) or all(x<0 for x in signs)),
            cost_squared_share=float(np.square(z.montant_sinistre).sum()/np.square(frame.montant_sinistre).sum())))
pd.DataFrame(rows).to_csv(OUT/'residual_groups.csv',index=False)
pd.DataFrame(foldrows).to_csv(OUT/'residual_groups_by_fold.csv',index=False)
rank=pd.DataFrame(rows);rank=rank[rank.claims.ge(100)&rank.severity_direction_consistent].copy()
rank['absolute_residual']=rank.weighted_severity_relative_residual.abs()
rank.sort_values('absolute_residual',ascending=False).to_csv(OUT/'stable_descriptive_severity_groups.csv',index=False)
# Associate the exact reported score with the already verified file.
path=ROOT/'deliverables/candidates/submission_severity_successor.csv'
receipt=dict(file=str(path.relative_to(ROOT)),sha256=r.digest(path),public_score=1365.914,source='User supplied Kaggle Complete result',
    independently_browser_verified=False,local_date='2026-10-12',target=1365.5,target_reached=False,
    decision='Not an improvement over severity_refinement at 1365.913; retain previous incumbent')
r.write_json(ROOT/'reports/severity_successor/kaggle_result.json',receipt)
result_path=ROOT/'reports/severity_successor/results.json';result=json.loads(result_path.read_text())
result['export']['kaggle_score']=1365.914;result['recommendation']=receipt['decision'];r.write_json(result_path,result)
for name in ['reports/kaggle_submissions.json','reports/submission_queue.json']:
    path=ROOT/name;data=json.loads(path.read_text())
    if 'submissions' in data:
        for z in data['submissions']:
            if z.get('sha256')==receipt['sha256']:z.update(public_score=1365.914,status='complete_user_reported',decision=receipt['decision'])
    else:
        if data.get('next_submission',{}).get('sha256')==receipt['sha256']:
            data['next_submission'].update(kaggle_score=1365.914,status='complete_user_reported',recommendation=receipt['decision'])
            data['state']='variable_reanalysis; severity_refinement_remains_incumbent'
    r.write_json(path,data)
print(rank.sort_values('absolute_residual',ascending=False)[['variable','level','claims','weighted_severity_relative_residual']].head(25).to_string(index=False))
