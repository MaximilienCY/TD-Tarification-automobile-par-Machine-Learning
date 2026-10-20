"""Verify completed artifacts and register the challenger without claiming an upload."""
from pathlib import Path
import importlib.metadata
import json
import re
import numpy as np
import pandas as pd
import nbformat
import joblib
import restart_xgboost as pipeline
ROOT, OUT = pipeline.ROOT, pipeline.OUT
_, test, _, _ = pipeline.load_data()
final = json.loads((OUT/'final_model.json').read_text())
results = json.loads((OUT/'nested_results.json').read_text())
receipt_path=OUT/'kaggle_result.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
if receipt and receipt.get('sha256')!=final['challenger_sha256']:
    receipt=None  # A score belongs to exact predictions, not just a reused filename.
kaggle_score=receipt['public_score'] if receipt else None
if receipt:
    final.update(kaggle_score=kaggle_score,submission_state='complete_user_reported',score_source=receipt['source'])
    pipeline.write_json(OUT/'final_model.json',final)
candidate = ROOT/final['challenger']
pipeline.valid_submission(candidate,test)
replica = OUT/'reproduced.csv'
assert candidate.read_bytes() == replica.read_bytes(), 'Fresh-process predictions differ'
notebook = ROOT/'notebooks/04_reprise_xgboost.ipynb'
nb = nbformat.read(notebook,as_version=4)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
for name in ['outer_1','outer_2','outer_3','full_training']:
    for component in ['frequency','severity']:
        assert len(pd.read_csv(OUT/name/(component+'_trials.csv'))) == 60
v = pd.read_csv(OUT/'nested_predictions.csv')
assert len(v)==50000 and v['index'].is_unique and np.isfinite(v.pred).all()
assert pipeline.digest(candidate)==final['challenger_sha256']
assert pipeline.digest(ROOT/'models/restart_xgboost.joblib') == final['model_sha256']
test_log = ROOT/'reports/restart_xgboost_tests.log'
test_match = re.search(r'(\d+) passed',test_log.read_text())
assert test_match and not re.search(r'\d+ failed|\d+ error',test_log.read_text())
tests_passed = int(test_match.group(1))
model = joblib.load(ROOT/'models/restart_xgboost.joblib')
deployed = {}
importance = []
for name in ['frequency','severity']:
    spec = final['choice'][name]
    if spec['kind'] == 'xgb':
        component = getattr(model,name+'_model')
        deployed[name] = dict(config=component.config, trees=component.rounds_, weight=spec['weight'], calibrated=spec.get('calibrated',False))
        for feature,value in zip(component.model_.feature_names_in_,component.model_.feature_importances_):
            importance.append(dict(component=name,variable=feature,gain_importance=float(value)))
    else:
        deployed[name] = dict(kind='original reference',weight=1.)
pipeline.write_json(OUT/'deployed_components.json',deployed)
pd.DataFrame(importance,columns=['component','variable','gain_importance']).sort_values('gain_importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
gain = results['gain']
table = '\n'.join(f"| {name} | {results['reference'][name]:.6f} | {results['selected'][name]:.6f} |" for name in ['RMSE','MSE','MAE','bias'])
summary = f"""# Résultats de la reprise XGBoost

Trois plis externes évaluent la sélection complète sur les 50 000 contrats, avec cinq plis internes et 60 essais par composante. Les données ont déjà été explorées : cette évaluation ne constitue pas un test neuf.

| Mesure externe | Référence initiale | Procédure de sélection |
| --- | ---: | ---: |
{table}

Gain RMSE : **{gain['gain_RMSE']:.6f}** ; intervalle descriptif bootstrap clients à 95 % : **[{gain['interval_95'][0]:.6f} ; {gain['interval_95'][1]:.6f}]**. Recommandation : **{final['production_recommendation']}**.

Le choix final sur tous les clients est :

```json
{json.dumps(deployed,indent=2,ensure_ascii=False)}
```

Les catalogues détaillés des configurations retenues sont dans `reports/restart_xgboost/full_training/selection.json`. Les scores de cette sélection interne sont optimistes et ne remplacent pas les résultats externes ci-dessus.

Dans les trois recherches internes externes, les quinze configurations du top cinq de fréquence utilisent le panneau compact. La gravité sélectionnée et les poids varient davantage entre plis. Les configurations de panneaux utilisent des tirages de paramètres différents : cette observation n'est pas une ablation contrôlée ni une preuve causale de l'effet d'une variable.

Le challenger est `deliverables/candidates/submission_restart_xgboost.csv`. Le fichier original reste disponible et les anciennes expériences sont conservées. Le modèle sauvegardé reproduit les prédictions dans un processus séparé, octet pour octet. Le notebook `notebooks/04_reprise_xgboost.ipynb` est exécuté sans erreur. Les {tests_passed} tests passent.

Les importances de gain dans `reports/restart_xgboost/feature_importance.csv` sont relatives à chaque composante et ne mesurent pas des effets causaux.

**Kaggle : {str(kaggle_score)+' (résultat transmis par l’utilisateur, non vérifié indépendamment dans le navigateur)' if receipt else 'aucun nouveau score confirmé'}.** Le seuil de 1365.500 n'est pas atteint. La référence initiale reste à 1365.943. Le modèle XGBoost n'est pas recommandé pour remplacer cette référence.
"""
(ROOT/'docs/restart_results.md').write_text(summary,encoding='utf-8')
paths = [candidate,ROOT/'models/restart_xgboost.joblib',notebook,
    ROOT/'scripts/restart_xgboost.py',ROOT/'scripts/run_restart_parallel.py',
    ROOT/'scripts/restart_cache.py',
    ROOT/'scripts/build_restart_notebook.py',ROOT/'scripts/verify_restart.py']
manifest = dict(status='verified', rows=len(test), all_60_trials_in_all_four_searches=True,
    tests_passed=tests_passed, test_log_sha256=pipeline.digest(test_log),
    fresh_process_byte_identical=True, notebook_executed_without_errors=True,
    protected_notebook_sha256=pipeline.digest(ROOT/"Calcul de la prime d'assurance.ipynb"),
    hashes={str(p.relative_to(ROOT)):pipeline.digest(p) for p in paths},
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','nbformat','nbclient']},
    target_reached=bool(kaggle_score is not None and kaggle_score<1365.5), kaggle_score=kaggle_score,
    note=receipt['source'] if receipt else 'No new Kaggle result: saved browser site permission remains unresolved.')
pipeline.write_json(OUT/'verification.json',manifest)
queue_path = ROOT/'reports/submission_queue.json'
queue = json.loads(queue_path.read_text())
if not receipt:
    if queue['next_submission']['file'] != str(candidate.relative_to(ROOT)):
        queue['previous_next_submission'] = queue['next_submission']
    queue['next_submission'] = dict(file=str(candidate.relative_to(ROOT)),sha256=pipeline.digest(candidate),
        kaggle_score=None,role='challenger',recommendation=final['production_recommendation'])
pipeline.write_json(queue_path,queue)
registry_path = ROOT/'reports/kaggle_submissions.json'
registry = json.loads(registry_path.read_text())
entry = dict(label='XGBoost restart: nested frequency x severity selection',
    file=str(candidate.relative_to(ROOT)),sha256=pipeline.digest(candidate),status='complete_user_reported' if receipt else 'permission_blocked',public_score=kaggle_score)
entries = registry['submissions']
if not any(z.get('sha256')==entry['sha256'] for z in entries): entries.append(entry)
pipeline.write_json(registry_path,registry)
print(json.dumps(manifest,indent=2))
