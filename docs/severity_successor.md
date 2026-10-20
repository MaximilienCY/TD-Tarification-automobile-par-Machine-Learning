# Successeur de gravité — 12 octobre 2026

Le meilleur score public transmis par l'utilisateur est **1365.913** pour `submission_severity_refinement.csv`. L'objectif strictement inférieur à **1365.500** reste non atteint. Ce successeur a obtenu 1365.914 selon le résultat transmis par l’utilisateur ; il ne remplace pas le précédent meilleur modèle.

21 corrections prédéfinies, deux moyennes et quatre poids de mélange sont comparés, avec Gamma originale comme option supplémentaire : 93 choix. Trois plis externes séparés par client évaluent la sélection dans cinq plis internes. Les corrections et leurs pondérations auxiliaires sont ajustées uniquement dans l'apprentissage. Les sinistres extrêmes sont conservés. Les données ont déjà été explorées ; cette évaluation reste diagnostique.

| Modèle/procédure sur les mêmes plis | RMSE de prime |
| --- | ---: |
| Sélection du successeur | 542.022604848 |
| Recette fixe du modèle à 1365.913, réapprise par pli | 542.030725394 |
| Référence originale | 542.060129964 |

Gain de RMSE : **0.008120547** ; intervalle descriptif bootstrap apparié par client à 95 % : **[-0.018781685311986963, 0.03709055955707185]** (2 000 répétitions). Deux plis sur trois s'améliorent. L'intervalle inclut zéro : le gain reste incertain. La comparaison à la recette fixe diffère de l'ancienne évaluation, qui sélectionnait elle-même le modèle de gravité à chaque pli.

La fréquence logistique reste exactement identique. La gravité finale mélange à 50 % Gamma originale et à 50 % une correction multiplicative moyenne de trois XGBoost de 480 arbres, profondeur 2, sous-échantillonnage 0,8. Les graines sont 20261012, 20261013 et 20261014. Les variables sont âge du véhicule, type de contrat et usage. La cible est coût/Gamma hors client ; les poids sont probabilité auxiliaire × Gamma², normalisés à moyenne 1. Le modèle à 1365.913 utilisait une seule correction sans sous-échantillonnage.

```json
{
  "name": "bagged_480",
  "panel": "basic",
  "mode": "ratio",
  "depth": 2,
  "trees": 480,
  "lr": 0.025,
  "min_child_weight": 100.0,
  "reg_lambda": 100.0,
  "reg_alpha": 1.0,
  "max_bin": 256,
  "seeds": 3,
  "subsample": 0.8
}
```

Le CSV `deliverables/candidates/submission_severity_successor.csv` contient 50 000 lignes `index,pred`, alignées sur le test. Il est reproduit octet pour octet dans un processus distinct. Le modèle est rechargé, les probabilités sont inchangées, la recette du précédent meilleur modèle est reproduite exactement et son CSV est préservé. Les 43 tests passent. Le notebook original reste identique. Empreintes et contrôles : `reports/severity_successor/verification.json`.

```powershell
.\.venv\Scripts\python.exe scripts/improve_severity.py train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/improve_severity.py predict --output reports/severity_successor/reproduced.csv
.\.venv\Scripts\python.exe scripts/verify_severity_successor.py
```

L'accès automatisé à Kaggle reste bloqué par le réglage navigateur enregistré. Le résultat transmis est consigné dans reports/severity_successor/kaggle_result.json ; il n’a pas été vérifié indépendamment par le navigateur.
