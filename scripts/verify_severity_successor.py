"""Verify the successor and publish its evidence without changing previous models."""
import json
from pathlib import Path
import sys
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import improve_severity as m

ROOT=m.ROOT
out=m.OUT
result=json.loads((out/'results.json').read_text())
receipt_path=out/'kaggle_result.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
candidate=ROOT/result['export']['file']
assert m.r.digest(candidate)==result['export']['sha256']
assert candidate.read_bytes()==(out/'reproduced.csv').read_bytes()
incumbent=ROOT/'deliverables/candidates/submission_severity_refinement.csv'
assert m.r.digest(incumbent)=='2874eeb918f4fd40ec773baf000a3cf743da4a036e230c0fb204a0447a57c1ee'
with threadpool_limits(limits=4):
    m.install()
    train,test,_,_=m.r.load_data()
    model=joblib.load(ROOT/'models/severity_successor.joblib')
    old=joblib.load(ROOT/'models/severity_refinement.joblib')
    v=model.details(test);old_v=old.details(test)
    np.testing.assert_array_equal(v.proba_pred,old_v.proba_pred)
    anchor=m.Premium(dict(candidate=0,weight=.5)).fit(train).details(test)
    np.testing.assert_array_equal(anchor.pred,old_v.pred)
    np.testing.assert_allclose(old_v.pred,pd.read_csv(incumbent).pred,rtol=1e-14,atol=1e-12)
    m.r.valid_submission(candidate,test)
    assert v.proba_pred.between(0,1).all() and (v.montant_pred>0).all()
assert m.r.digest(ROOT/'models/severity_successor.joblib')==result['export']['model_sha256']
assert '43 passed' in (ROOT/'reports/severity_successor_tests.log').read_text(encoding='utf-16' if (ROOT/'reports/severity_successor_tests.log').read_bytes().startswith(b'\xff\xfe') else 'utf-8')
config=m.catalogue()[result['final_choice']['candidate']]
manifest=dict(status='verified',tests_passed=43,rows=len(v),fresh_process_byte_identical=True,
    frequency_unchanged=True,full_refit_incumbent_exact=True,incumbent_csv_unchanged=True,
    original_notebook_sha256=m.r.digest(ROOT/"Calcul de la prime d'assurance.ipynb"),
    candidate_sha256=m.r.digest(candidate),model_sha256=result['export']['model_sha256'],
    script_sha256=m.r.digest(Path(m.__file__)),provenance=m.r.provenance(),configuration=config,
    xgboost_version=importlib.metadata.version('xgboost'),kaggle_score=receipt['public_score'] if receipt else None)
m.r.write_json(out/'verification.json',manifest)
delta=v.pred.to_numpy()-old_v.pred.to_numpy()
m.r.write_json(out/'prediction_change.json',dict(mean=float(delta.mean()),mean_absolute=float(np.abs(delta).mean()),
    maximum_absolute=float(np.abs(delta).max()),percentiles=np.quantile(delta,[0,.01,.5,.99,1]).tolist()))
entry=dict(file=str(candidate.relative_to(ROOT)),sha256=m.r.digest(candidate),public_score=None,
    status='verified_not_submitted',label='Three-seed severity correction; original frequency unchanged',
    validation_gain_uncertain=True)
registry_path=ROOT/'reports/kaggle_submissions.json';registry=json.loads(registry_path.read_text())
if not any(z.get('sha256')==entry['sha256'] for z in registry['submissions']):registry['submissions'].append(entry)
m.r.write_json(registry_path,registry)
queue_path=ROOT/'reports/submission_queue.json';queue=json.loads(queue_path.read_text())
if not receipt:
    queue['next_submission']=dict(entry,kaggle_score=None,recommendation=result['recommendation'])
    queue['state']='verified_severity_successor_ready; browser_restriction_unresolved'
queue['exploratory_candidate']['recommendation']='public incumbent; local gain uncertain'
m.r.write_json(queue_path,queue)
gain=result['gain_vs_incumbent']
text=f"""# Successeur de gravité — 12 octobre 2026

Le meilleur score public transmis par l'utilisateur est **1365.913** pour `submission_severity_refinement.csv`. L'objectif strictement inférieur à **1365.500** reste non atteint. {'Ce successeur a obtenu 1365.914 selon le résultat transmis par l’utilisateur ; il ne remplace pas le précédent meilleur modèle.' if receipt else 'Ce nouveau CSV n’a pas de score Kaggle connu.'}

21 corrections prédéfinies, deux moyennes et quatre poids de mélange sont comparés, avec Gamma originale comme option supplémentaire : 93 choix. Trois plis externes séparés par client évaluent la sélection dans cinq plis internes. Les corrections et leurs pondérations auxiliaires sont ajustées uniquement dans l'apprentissage. Les sinistres extrêmes sont conservés. Les données ont déjà été explorées ; cette évaluation reste diagnostique.

| Modèle/procédure sur les mêmes plis | RMSE de prime |
| --- | ---: |
| Sélection du successeur | {result['selected']['RMSE']:.9f} |
| Recette fixe du modèle à 1365.913, réapprise par pli | {result['incumbent']['RMSE']:.9f} |
| Référence originale | {result['original']['RMSE']:.9f} |

Gain de RMSE : **{gain['gain_RMSE']:.9f}** ; intervalle descriptif bootstrap apparié par client à 95 % : **{gain['interval_95']}** (2 000 répétitions). Deux plis sur trois s'améliorent. L'intervalle inclut zéro : le gain reste incertain. La comparaison à la recette fixe diffère de l'ancienne évaluation, qui sélectionnait elle-même le modèle de gravité à chaque pli.

La fréquence logistique reste exactement identique. La gravité finale mélange à 50 % Gamma originale et à 50 % une correction multiplicative moyenne de trois XGBoost de {config['trees']} arbres, profondeur 2, sous-échantillonnage 0,8. Les graines sont 20261012, 20261013 et 20261014. Les variables sont âge du véhicule, type de contrat et usage. La cible est coût/Gamma hors client ; les poids sont probabilité auxiliaire × Gamma², normalisés à moyenne 1. Le modèle à 1365.913 utilisait une seule correction sans sous-échantillonnage.

```json
{json.dumps(config,indent=2)}
```

Le CSV `deliverables/candidates/submission_severity_successor.csv` contient 50 000 lignes `index,pred`, alignées sur le test. Il est reproduit octet pour octet dans un processus distinct. Le modèle est rechargé, les probabilités sont inchangées, la recette du précédent meilleur modèle est reproduite exactement et son CSV est préservé. Les 43 tests passent. Le notebook original reste identique. Empreintes et contrôles : `reports/severity_successor/verification.json`.

```powershell
.\\.venv\\Scripts\\python.exe scripts/improve_severity.py train
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts/improve_severity.py predict --output reports/severity_successor/reproduced.csv
.\\.venv\\Scripts\\python.exe scripts/verify_severity_successor.py
```

L'accès automatisé à Kaggle reste bloqué par le réglage navigateur enregistré. {'Le résultat transmis est consigné dans reports/severity_successor/kaggle_result.json ; il n’a pas été vérifié indépendamment par le navigateur.' if receipt else 'Aucune soumission ni amélioration publique n’est revendiquée pour ce successeur.'}
"""
(ROOT/'docs/severity_successor.md').write_text(text,encoding='utf-8')
print(json.dumps(manifest,indent=2))
