"""Verify and document the follow-up experiments and user-reported Kaggle scores."""
from pathlib import Path
import json
import re
import sys
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import refine_severity as severity
import refine_frequency as frequency
import restart_xgboost as r
train,test,_,_=r.load_data()
s=json.loads((severity.OUT/'results.json').read_text())
f=json.loads((frequency.OUT/'results.json').read_text())
receipt_path=severity.OUT/'kaggle_result.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
assert f['export'] is None, 'Reassess report if the frequency procedure improves'
candidate=ROOT/s['export']['file']
assert candidate.read_bytes()==(severity.OUT/'reproduced.csv').read_bytes()
assert r.digest(candidate)==s['export']['sha256'];r.valid_submission(candidate,test)
model=joblib.load(ROOT/'models/severity_refinement.joblib')
old=joblib.load(ROOT/'models/restart_xgboost.joblib')
np.testing.assert_allclose(model.frequency.predict_proba(test)[:,1],old.reference_f.predict_proba(test)[:,1],rtol=0,atol=1e-12)
v=model.details(test);ref=pd.read_csv(ROOT/'deliverables/candidates/submission_reference.csv')
np.testing.assert_allclose(v.reference,ref.pred,atol=1e-6,rtol=0)
log=ROOT/'reports/refinement_tests.log';match=re.search(r'(\d+) passed',log.read_text())
assert match and not re.search(r'\d+ failed|\d+ error',log.read_text())
manifest=dict(status='verified',tests_passed=int(match.group(1)),fresh_process_byte_identical=True,
    frequency_unchanged=True,original_notebook_sha256=r.digest(ROOT/"Calcul de la prime d'assurance.ipynb"),
    candidate_sha256=r.digest(candidate),model_sha256=r.digest(ROOT/'models/severity_refinement.joblib'),
    scripts={name:r.digest(ROOT/'scripts'/name) for name in ['refine_severity.py','refine_frequency.py','summarize_refinement.py']},
    data_and_core=r.provenance(),kaggle_score=receipt['public_score'] if receipt else None,
    score_source=receipt['source'] if receipt else None,
    recommendation='new public incumbent reported by user; local gain uncertain' if receipt else 'original reference; exploratory severity-only candidate')
r.write_json(severity.OUT/'verification.json',manifest)
rows=[]
for name,z in [('Original',dict(selected=s['reference'],gain=dict(gain_RMSE=0.,interval_95=[0.,0.]))),('Severity only',s),('Frequency losses',f)]:
    rows.append(f"| {name} | {z['selected']['RMSE']:.6f} | {z['gain']['gain_RMSE']:.6f} | [{z['gain']['interval_95'][0]:.6f}, {z['gain']['interval_95'][1]:.6f}] |")
table='\n'.join(rows)
text=f"""# Suivi après les résultats Kaggle décevants

## Résultats transmis par l'utilisateur

| Fichier | Score Kaggle | Décision |
| --- | ---: | --- |
| Référence initiale | 1365.943 | Référence conservée |
| submission_anchored.csv | 1366.010 | Écarté |
| submission_restart_xgboost.csv | 1366.025 | Écarté |
| submission_severity_refinement.csv | {receipt['public_score'] if receipt else 'Inconnu'} | {'Nouveau meilleur score public transmis' if receipt else 'Exploratoire'} |

Les nouveaux scores sont ceux transmis par l'utilisateur le 12 octobre 2026, sans vérification indépendante par le navigateur. Ils sont associés aux empreintes des fichiers locaux dans les reçus et le registre de soumissions. Le seuil demandé, strictement inférieur à 1365.500, n'est pas atteint.

## Deux expériences séparées

La fréquence et la gravité ne sont plus modifiées simultanément. Chaque recherche utilise les mêmes trois plis externes et cinq plis internes séparés par client, avec la référence originale parmi les candidats. Ces données ont déjà été explorées ; il ne s'agit pas d'un nouveau test indépendant. Le score Kaggle n'entre pas dans les ajustements ni dans le classement des paramètres, mais cette recherche succède à la consultation du leaderboard.

### Gravité avec fréquence originale inchangée

34 configurations : Gamma avec décalages 0/300/600 ; Poisson quasi-vraisemblable avec décalages 0/600 ; effets d'âge linéaires ou splines ; pondération uniforme ou par probabilité de sinistre ; régularisation Gamma supplémentaire ; corrections multiplicatives XGBoost limitées (profondeurs 1–3, 80/240 arbres, L2 100). Les mélanges avec Gamma prennent les poids 0/0,1/0,25/0,5/1. Le décalage de 600 est une hypothèse de modélisation suggérée par le support observé, pas une franchise contractuelle connue.

Les probabilités auxiliaires et gravités de référence sont calculées hors client sur trois plis internes supplémentaires. Les probabilités auxiliaires proviennent d'une logistique compacte. Sur les seuls sinistres, pondérer une perte quadratique de gravité par p(x) vise la partie dépendante du modèle de l'erreur de prime : la densité des sinistres est déjà proportionnelle à p(x), tandis que l'erreur de prime dépend de p(x)². Les pertes Gamma/Poisson n'égalent pas exactement cette perte ; leur sélection se fait donc sur la RMSE de prime.

Pour XGBoost, la cible est le ratio coût/gravité de référence hors client, et le poids est p × gravité², normalisé à moyenne 1. L'identité `p × s² × (y/s − ratio)² = p × (y − s × ratio)²` est testée. Les sinistres extrêmes sont conservés. Les catégories et transformations apprises restent dans l'apprentissage de chaque pli. [GammaRegressor et sample_weight](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html), [objectifs XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

### Fréquence avec Gamma originale inchangée

16 configurations : C=0,03/0,1/0,3/1, pondération uniforme ou par gravité auxiliaire², log-loss ou Brier. Elles réutilisent exactement la préparation originale (dont l'encodage cible croisé par client). La gravité des poids exclut chaque client par apprentissage croisé. La perte Brier avec lien logistique est optimisée localement par L-BFGS, initialisée à la solution logistique ; les gradients et la convergence sont testés. Ce sont des pertes propres pour la probabilité conditionnelle lorsque les poids sont des fonctions des covariables. Les poids estimés et la restriction de modèle ne garantissent pas une amélioration.

## Résultats externes

| Procédure | RMSE | Gain face à l'original | Intervalle descriptif bootstrap clients 95 % |
| --- | ---: | ---: | --- |
{table}

Le gain local de gravité est trop petit et trop incertain pour annoncer une supériorité générale. La nouvelle procédure de fréquence est écartée et n'exporte aucun CSV. {'Le modèle de gravité est désormais le meilleur score public transmis, à 1365.913 ; il devient le benchmark des recherches suivantes.' if receipt else 'La référence originale reste recommandée.'}

## Challenger exploratoire de gravité

Configuration finale sélectionnée :

```json
{json.dumps(s['final_choice'],indent=2,ensure_ascii=False)}
```

Le fichier `deliverables/candidates/submission_severity_refinement.csv` conserve exactement les probabilités originales. La gravité est un mélange à 50 % de Gamma originale et à 50 % d'une correction multiplicative XGBoost (240 arbres, profondeur 2), limitée à âge du véhicule, couverture et usage. Son score Kaggle est **{receipt['public_score'] if receipt else 'inconnu'}**. {'Il est conservé comme nouvel incumbent public ; cela ne prouve pas une supériorité sur le classement privé.' if receipt else 'Il ne doit pas être présenté comme le meilleur modèle démontré.'}

Les 50 000 lignes, l'ordre et les index sont vérifiés ; le CSV est reproduit octet pour octet dans un nouveau processus, et le modèle sauvegardé est rechargé. Le notebook protégé est inchangé. {manifest['tests_passed']} tests passent.

```powershell
.\\.venv\\Scripts\\python.exe scripts/refine_severity.py train
.\\.venv\\Scripts\\python.exe scripts/refine_frequency.py train
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts/refine_severity.py predict --output reports/severity_refinement/reproduced.csv
.\\.venv\\Scripts\\python.exe scripts/summarize_refinement.py
```

Les résultats, choix de chaque pli et empreintes sont dans `reports/severity_refinement/` et `reports/frequency_refinement/`. Les scripts ne modifient ni le notebook original, ni les CSV précédents. Le cache dépend des données, des sources et des versions.
"""
(ROOT/'docs/refinement_after_kaggle.md').write_text(text,encoding='utf-8')
registry_path=ROOT/'reports/kaggle_submissions.json';registry=json.loads(registry_path.read_text())
entry=dict(label='Severity-only weighted XGBoost ratio; original frequency unchanged',file=str(candidate.relative_to(ROOT)),
    sha256=r.digest(candidate),public_score=None,status='experimental_not_submitted',validation_gain_uncertain=True)
if not any(z.get('sha256')==entry['sha256'] for z in registry['submissions']):registry['submissions'].append(entry)
r.write_json(registry_path,registry)
queue_path=ROOT/'reports/submission_queue.json';queue=json.loads(queue_path.read_text())
if not receipt:
    queue['state']='no_demonstrated_winner; exploratory_severity_candidate_available'
    queue['exploratory_candidate']=dict(file=str(candidate.relative_to(ROOT)),sha256=r.digest(candidate),kaggle_score=None,
        recommendation='original reference; gain uncertain')
r.write_json(queue_path,queue)
print(json.dumps(manifest,indent=2))
