"""Verify diagnostic follow-ups and publish only supported candidates."""
from pathlib import Path
import sys,json,subprocess,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import frequency_curves as f
ROOT=f.ROOT
results={name:json.loads((ROOT/'reports'/name/'results.json').read_text()) for name in ['smooth_severity','coverage_interactions','frequency_curves']}
receipt_path=ROOT/'reports/frequency_curves/kaggle_result.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
if receipt:results['frequency_curves']['export']['kaggle_score']=receipt['public_score']
assert '57 passed' in (ROOT/'reports/frequency_curves_tests.log').read_text()
protected=f.r.digest(ROOT/"Calcul de la prime d'assurance.ipynb")
assert protected=='76248bdb51400fe399c670a9b71092cabe041cc13d3839b83846ac1499938429'
assert f.r.digest(ROOT/'deliverables/candidates/submission_variable_models.csv')=='70a369b8201fd6044b0938cdfb21e1bd2ac501d9fe85ad6972cb584f18481aeb'
verification=dict(status='verified',tests_passed=57,original_notebook_sha256=protected,
    incumbent_csv_unchanged=True,best_user_reported_public_score=receipt['public_score'] if receipt else 1365.745,
    source_hashes={name:f.r.digest(ROOT/'scripts'/name) for name in ['refine_smooth_severity.py','coverage_interactions.py','frequency_curves.py']},
    provenance=f.r.provenance(),xgboost_version=importlib.metadata.version('xgboost'))
for name,r in results.items():
    r['recommendation']=('new user-reported public incumbent' if receipt and name=='frequency_curves' else
        'experimental challenger; positive descriptive local interval' if r['export'] and r['gain']['interval_95'][0]>0 else
        'exploratory challenger; local gain uncertain' if r['export'] else 'retain 1365.745 recipe; no supported successor')
    f.r.write_json(ROOT/'reports'/name/'results.json',r)
    v=pd.read_csv(ROOT/'reports'/name/'nested_predictions.csv')
    train,_,_,_=f.r.load_data()
    assert np.array_equal(v['index'],train['index'])
    contract=train[['type_contrat']].copy();contract['observed']=v.observed;contract['selected']=v.pred;contract['incumbent']=v.incumbent
    rows=[]
    for contract_type,z in contract.groupby('type_contrat'):
        rows.append(dict(type_contrat=contract_type,rows=len(z),selected_RMSE=f.r.score(z.observed,z.selected)['RMSE'],
            incumbent_RMSE=f.r.score(z.observed,z.incumbent)['RMSE']))
    pd.DataFrame(rows).to_csv(ROOT/'reports'/name/'contract_performance.csv',index=False)
    if not r['export']:
        f.r.write_json(ROOT/'reports'/name/'verification.json',dict(verification,export=None,recommendation=r['recommendation']))
        continue
    path=ROOT/r['export']['file'];reproduced=ROOT/'reports'/name/'reproduced.csv'
    scripts=dict(frequency_curves='frequency_curves.py',smooth_severity='refine_smooth_severity.py',coverage_interactions='coverage_interactions.py')
    subprocess.run([sys.executable,str(ROOT/'scripts'/scripts[name]),'predict','--output',str(reproduced)],check=True,cwd=ROOT)
    assert path.read_bytes()==reproduced.read_bytes()
    assert f.r.digest(path)==r['export']['sha256']
    with threadpool_limits(limits=4):
        _,test,_,_=f.r.load_data();model_path=ROOT/'models'/f'{name}.joblib'
        model=joblib.load(model_path);v=model.details(test);incumbent=joblib.load(ROOT/'models/variable_models.joblib').details(test)
        np.testing.assert_array_equal(v.incumbent,incumbent.pred)
        if name=='frequency_curves':np.testing.assert_array_equal(v.montant_pred,incumbent.montant_pred)
        else:np.testing.assert_array_equal(v.proba_pred,incumbent.proba_pred)
        assert v.proba_pred.between(0,1).all() and (v.montant_pred>0).all()
        f.r.valid_submission(path,test)
    assert f.r.digest(model_path)==r['export']['model_sha256']
    manifest=dict(verification,export=r['export'],fresh_process_byte_identical=True,rows=len(v),
        incumbent_exact=True,severity_unchanged=name=='frequency_curves',frequency_unchanged=name!='frequency_curves')
    f.r.write_json(ROOT/'reports'/name/'verification.json',manifest)
    registry_path=ROOT/'reports/kaggle_submissions.json';registry=json.loads(registry_path.read_text())
    submitted=bool(receipt and name=='frequency_curves')
    entry=dict(r['export'],public_score=receipt['public_score'] if submitted else None,
        status='complete_user_reported' if submitted else 'verified_not_submitted',label=name,
        validation_gain_uncertain=r['gain']['interval_95'][0]<=0)
    existing=next((z for z in registry['submissions'] if z.get('sha256')==entry['sha256']),None)
    if existing:existing.update(entry)
    else:registry['submissions'].append(entry)
    if submitted:registry.update(best_user_reported_public_score=receipt['public_score'],best_user_reported_file=receipt['file'])
    f.r.write_json(registry_path,registry)
    queue_path=ROOT/'reports/submission_queue.json';queue=json.loads(queue_path.read_text())
    queue.update(next_submission=dict(entry,recommendation=r['recommendation']),
        state='frequency_curves_public_incumbent' if submitted else 'verified_variable_followup_ready; browser_restriction_unresolved')
    if submitted:queue.update(public_incumbent=receipt,best_user_reported_public_score=receipt['public_score'])
    f.r.write_json(queue_path,queue)
table='\n'.join(f"| {name} | {r['selected']['RMSE']:.9f} | {r['incumbent']['RMSE']:.9f} | {r['gain']['gain_RMSE']:.9f} | {r['export']['file'] if r['export'] else 'Aucun export'} |" for name,r in results.items())
text=f"""# Recherches après le score public 1365.745

Le score public de départ transmis par l'utilisateur est **1365.745** (`submission_variable_models.csv`). {'La correction de fréquence obtient désormais **1365.657**, également transmis par l’utilisateur : c’est le nouveau meilleur score public observé, encore 0.157 au-dessus de 1365.500.' if receipt else 'Aucun score public nouveau n’est encore connu.'} Les recherches suivantes évaluent la recette fixe du modèle à 1365.745, réapprise dans chaque pli externe. Sa RMSE externe **541.971031** diffère de **541.942903** : cette dernière évaluait l'ancienne procédure de sélection, qui choisissait parfois XGBoost et d'autres poids dans chaque pli.

Chaque recherche utilise trois plis externes et cinq plis internes séparés par client. Catégories, imputations, courbes, corrections, pondérations auxiliaires et sélection sont apprises dans les données d'apprentissage. Le coût des sinistres extrêmes reste intact. Les données et le leaderboard ont déjà été consultés : ces évaluations restent diagnostiques, sans test indépendant nouveau.

| Recherche | RMSE sélection | RMSE recette fixe à 1365.745 | Gain | Export |
| --- | ---: | ---: | ---: | --- |
{table}

## Gravité : courbes et interactions

`refine_smooth_severity.py` teste 19 configurations : régularisation Ridge 1/3/10/30/100, nombre de nœuds 3/4/5/6, regroupement des catégories avec seuils 5/15/30/50, pénalisation relative des catégories, et corrections Gamma/Poisson à lien log. 96 choix avec mélanges, dont le meilleur modèle actuel. Sa sélection externe se dégrade ; aucun CSV n'est exporté.

`coverage_interactions.py` teste 12 corrections hiérarchiques : courbes de caractéristiques véhicule × couverture, courbes conducteur × usage, ou les deux ; deux pénalisations et deux échelles d'interaction. Les interactions sont centrées avec des moyennes d'apprentissage pondérées. 37 choix avec mélanges et option inchangée. La sélection externe se dégrade ; aucun CSV n'est exporté.

## Fréquence : correction régularisée des probabilités originales

`frequency_curves.py` ajuste `logit(p_nouveau) = logit(p_original) + poids × correction(X)`. La correction apprend une log-loss binomiale pénalisée, sans poids de classe ni repondération des observations. Le gradient analytique est vérifié numériquement. Les probabilités utilisées pour apprendre les corrections proviennent de trois sous-plis séparés par client, avec une copie exacte du modèle de fréquence original. L'utilisation des probabilités d'apprentissage sur les mêmes clients est ainsi évitée.

Neuf configurations : courbe d'âge du véhicule, courbe avec couverture/usage et interaction âge × couverture, ou panneau compact avec âge conducteur, bonus, prix et puissance ; pénalités 100/1000/10000. 28 choix avec corrections à 25/50/100 % et option inchangée. La gravité conserve exactement la recette du modèle à 1365.745. La sélection se fait sur la RMSE de prime ; log-loss et Brier sont aussi enregistrés.

```json
{json.dumps(results['frequency_curves'],indent=2)}
```

Les intervalles bootstrap sont descriptifs, appariés par client, conditionnels aux prédictions ajustées ; ils ne couvrent pas toute l'incertitude d'apprentissage. Aucun score Kaggle nouveau n'est inféré des résultats locaux. Le seuil strictement inférieur à 1365.500 reste non établi.

57 tests passent, le fichier du meilleur modèle est préservé, et le notebook original reste identique. Chaque éventuel export est reproduit octet pour octet dans un processus distinct et vérifié sur 50 000 lignes. Les preuves et performances par contrat sont dans les sous-dossiers de `reports/`.

```powershell
.\\.venv\\Scripts\\python.exe scripts/refine_smooth_severity.py train
.\\.venv\\Scripts\\python.exe scripts/coverage_interactions.py train
.\\.venv\\Scripts\\python.exe scripts/frequency_curves.py train
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts/summarize_variable_followups.py
```

Les fichiers historiques sont conservés. L'accès automatisé à Kaggle reste bloqué par le réglage navigateur enregistré.
"""
(ROOT/'docs/variable_followups.md').write_text(text,encoding='utf-8')
print(table)
