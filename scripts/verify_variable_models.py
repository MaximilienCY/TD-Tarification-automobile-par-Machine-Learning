"""Verify and document the expanded-variable model and the user-reported score."""
import json
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import variable_models as m
ROOT=m.ROOT;OUT=m.OUT
result=json.loads((OUT/'results.json').read_text());path=ROOT/result['export']['file']
assert m.r.digest(path)==result['export']['sha256']
assert path.read_bytes()==(OUT/'reproduced.csv').read_bytes()
with threadpool_limits(limits=4):
    _,test,_,_=m.r.load_data();model=joblib.load(ROOT/'models/variable_models.joblib')
    old=joblib.load(ROOT/'models/severity_refinement.joblib');v=model.details(test);anchor=old.details(test)
    np.testing.assert_array_equal(v.proba_pred,anchor.proba_pred)
    np.testing.assert_array_equal(v.incumbent,anchor.pred)
    m.r.valid_submission(path,test)
    assert v.proba_pred.between(0,1).all() and (v.montant_pred>0).all()
assert m.r.digest(ROOT/'models/variable_models.joblib')==result['export']['model_sha256']
assert m.r.digest(ROOT/'deliverables/candidates/submission_severity_refinement.csv')=='2874eeb918f4fd40ec773baf000a3cf743da4a036e230c0fb204a0447a57c1ee'
assert '46 passed' in (ROOT/'reports/variable_models_tests.log').read_text()
config=m.catalogue()[result['final_choice']['candidate']]
manifest=dict(status='verified',rows=len(v),tests_passed=46,fresh_process_byte_identical=True,frequency_unchanged=True,
    original_notebook_sha256=m.r.digest(ROOT/"Calcul de la prime d'assurance.ipynb"),candidate_sha256=m.r.digest(path),
    model_sha256=result['export']['model_sha256'],script_sha256=m.r.digest(Path(m.__file__)),configuration=config,
    provenance=m.r.provenance(),xgboost_version=importlib.metadata.version('xgboost'))
m.r.write_json(OUT/'verification.json',manifest)
receipt=dict(file=str(path.relative_to(ROOT)),sha256=m.r.digest(path),public_score=1365.745,
    source='User supplied Kaggle Complete public score',independently_browser_verified=False,local_date='2026-10-12',
    previous_best_public_score=1365.913,improvement=.168,target=1365.5,target_reached=False)
m.r.write_json(OUT/'kaggle_result.json',receipt)
result['export']['kaggle_score']=1365.745;result['recommendation']='new user-reported public incumbent';m.r.write_json(OUT/'results.json',result)
registry_path=ROOT/'reports/kaggle_submissions.json';registry=json.loads(registry_path.read_text())
entry=dict(receipt,label='Expanded-variable smooth severity correction at 25 percent',status='complete_user_reported',validation_gain_uncertain=True)
if not any(z.get('sha256')==receipt['sha256'] for z in registry['submissions']):registry['submissions'].append(entry)
is_best=receipt['public_score']<=registry.get('best_user_reported_public_score',float('inf'))
if is_best:registry.update(best_user_reported_public_score=1365.745,best_user_reported_file=receipt['file'])
m.r.write_json(registry_path,registry)
queue_path=ROOT/'reports/submission_queue.json';queue=json.loads(queue_path.read_text())
if is_best:
    queue.update(public_incumbent=receipt,best_user_reported_public_score=1365.745)
    if queue.get('next_submission',{}).get('kaggle_score') is not None:
        queue.update(state='improving_expanded_variable_incumbent',next_submission=dict(entry,kaggle_score=1365.745))
m.r.write_json(queue_path,queue)
groups=pd.read_csv(ROOT/'reports/variable_reanalysis/stable_descriptive_severity_groups.csv')
summary=groups.head(8)[['variable','level','claims','weighted_severity_relative_residual','top25_claims','severity_residual_without_top25_diagnostic']]
gain=result['gain']
text=f"""# Réanalyse des variables et correction de gravité

Le score public **1365.745**, transmis par l'utilisateur le 12 octobre 2026, améliore 1365.913 de 0.168. Le seuil strictement inférieur à 1365.500 reste non atteint. Le successeur précédent, à 1365.914, est écarté. Reçus et empreintes associés : `reports/variable_models/kaggle_result.json` et `reports/severity_successor/kaggle_result.json`.

## Réanalyse

Chaque variable brute éligible et neuf variables dérivées sont étudiées sur les prédictions hors client du précédent meilleur modèle : fréquence observée/prédite, gravité observée/prédite, coût moyen, erreur de gravité pondérée par la probabilité auxiliaire, cohérence de direction dans trois plis, et part des coûts au carré. Les variables identifiantes et les sexes restent exclus. Les zéros de poids/cylindrée deviennent des mesures manquantes avec indicateurs. L'absence du second conducteur est explicite ; ses âges et permis nuls deviennent manquants pour les courbes, sans effacer l'indicateur d'absence. Les anomalies de permis sont signalées. Aucun sinistre n'est supprimé à l'entraînement.

Les groupes ci-dessous sont descriptifs, choisis après consultation des données. La dernière colonne retire les 25 coûts les plus élevés uniquement pour mesurer leur influence ; ce retrait n'est jamais utilisé pour ajuster le modèle. Pour Mercedes, le résidu relatif pondéré passe d'environ +22 % à +6 % ; pour les permis obtenus après 26 ans, de +11 % à +1 %. Une apparente association ne garantit donc pas un signal stable ni causal.

```text
{summary.to_string(index=False)}
```

Les détails sont dans `reports/variable_reanalysis/residual_groups.csv`, `residual_groups_by_fold.csv` et `stable_descriptive_severity_groups.csv`. Le dictionnaire initial et les différences train/test restent dans `reports/restart_xgboost/variable_dictionary.csv`.

## Modèles et sélection

Quatre familles de variables : véhicule, conducteur, contrat et ensemble complet. Pour chaque famille : correction linéaire Ridge avec courbes splines quadratiques et catégories regroupées si moins de 15 sinistres d'apprentissage ; régularisations 10/100/1000/10000. Quatre XGBoost supplémentaires testent l'ensemble des variables (profondeurs 1/2, 150/400 arbres, L2 300, L1 10). Chaque candidat est mélangé au meilleur modèle précédent avec poids 0,25/0,5/1 ; le précédent modèle seul reste candidat. 61 choix, cinq plis internes et trois plis externes séparés par client.

Les corrections apprennent la cible coût/Gamma hors client − 1 avec poids probabilité auxiliaire × Gamma², normalisés à moyenne 1. Les catégories, imputations, courbes, paramètres et poids de mélange sont ajustés dans l'apprentissage de chaque pli. Les données ont déjà été explorées : cette validation ne devient pas un test indépendant.

| Procédure sur les mêmes plis | RMSE |
| --- | ---: |
| Sélection des variables étendues | {result['selected']['RMSE']:.9f} |
| Recette fixe du précédent meilleur modèle | {result['incumbent']['RMSE']:.9f} |

Gain descriptif : {gain['gain_RMSE']:.9f}, intervalle bootstrap clients 95 % {gain['interval_95']}. Les trois plis s'améliorent, mais l'intervalle inclut zéro.

Configuration finale : `{config}`, poids **{result['final_choice']['weight']}**. La fréquence reste exactement celle d'origine. Le fichier est `deliverables/candidates/submission_variable_models.csv`, 50 000 lignes index,pred ; il est reproduit octet pour octet dans un processus distinct. Les 46 tests passent. Le notebook protégé est inchangé.

```powershell
.\\.venv\\Scripts\\python.exe scripts/reanalyse_variables.py
.\\.venv\\Scripts\\python.exe scripts/variable_models.py train
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts/variable_models.py predict --output reports/variable_models/reproduced.csv
.\\.venv\\Scripts\\python.exe scripts/verify_variable_models.py
```

Les commandes de vérification consignent les scores exactement transmis dans cette conversation ; elles ne les récupèrent pas sur Kaggle. L'accès automatisé au navigateur reste bloqué par le réglage enregistré.
"""
(ROOT/'docs/variable_reanalysis.md').write_text(text,encoding='utf-8')
print(json.dumps(manifest,indent=2))
