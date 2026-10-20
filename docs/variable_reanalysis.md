# Réanalyse des variables et correction de gravité

Le score public **1365.745**, transmis par l'utilisateur le 12 octobre 2026, améliore 1365.913 de 0.168. Le seuil strictement inférieur à 1365.500 reste non atteint. Le successeur précédent, à 1365.914, est écarté. Reçus et empreintes associés : `reports/variable_models/kaggle_result.json` et `reports/severity_successor/kaggle_result.json`.

## Réanalyse

Chaque variable brute éligible et neuf variables dérivées sont étudiées sur les prédictions hors client du précédent meilleur modèle : fréquence observée/prédite, gravité observée/prédite, coût moyen, erreur de gravité pondérée par la probabilité auxiliaire, cohérence de direction dans trois plis, et part des coûts au carré. Les variables identifiantes et les sexes restent exclus. Les zéros de poids/cylindrée deviennent des mesures manquantes avec indicateurs. L'absence du second conducteur est explicite ; ses âges et permis nuls deviennent manquants pour les courbes, sans effacer l'indicateur d'absence. Les anomalies de permis sont signalées. Aucun sinistre n'est supprimé à l'entraînement.

Les groupes ci-dessous sont descriptifs, choisis après consultation des données. La dernière colonne retire les 25 coûts les plus élevés uniquement pour mesurer leur influence ; ce retrait n'est jamais utilisé pour ajuster le modèle. Pour Mercedes, le résidu relatif pondéré passe d'environ +22 % à +6 % ; pour les permis obtenus après 26 ans, de +11 % à +1 %. Une apparente association ne garantit donc pas un signal stable ni causal.

```text
            variable         level  claims  weighted_severity_relative_residual  top25_claims  severity_residual_without_top25_diagnostic
     marque_vehicule MERCEDES BENZ     118                             0.222560             4                                    0.055784
     modele_vehicule           206     123                             0.175045             1                                    0.126120
   licence_start_age  (26.0, 69.0]     461                             0.112322             5                                    0.014558
     marque_vehicule     __OTHER__     127                            -0.101452             0                                   -0.101452
     modele_vehicule          CLIO     192                             0.096550             2                                    0.066970
debut_vente_vehicule  (13.0, 18.0]     222                            -0.086827             0                                   -0.086827
        type_contrat       Median2     283                            -0.084053             1                                   -0.094619
     marque_vehicule        TOYOTA     108                            -0.082064             0                                   -0.082064
```

Les détails sont dans `reports/variable_reanalysis/residual_groups.csv`, `residual_groups_by_fold.csv` et `stable_descriptive_severity_groups.csv`. Le dictionnaire initial et les différences train/test restent dans `reports/restart_xgboost/variable_dictionary.csv`.

## Modèles et sélection

Quatre familles de variables : véhicule, conducteur, contrat et ensemble complet. Pour chaque famille : correction linéaire Ridge avec courbes splines quadratiques et catégories regroupées si moins de 15 sinistres d'apprentissage ; régularisations 10/100/1000/10000. Quatre XGBoost supplémentaires testent l'ensemble des variables (profondeurs 1/2, 150/400 arbres, L2 300, L1 10). Chaque candidat est mélangé au meilleur modèle précédent avec poids 0,25/0,5/1 ; le précédent modèle seul reste candidat. 61 choix, cinq plis internes et trois plis externes séparés par client.

Les corrections apprennent la cible coût/Gamma hors client − 1 avec poids probabilité auxiliaire × Gamma², normalisés à moyenne 1. Les catégories, imputations, courbes, paramètres et poids de mélange sont ajustés dans l'apprentissage de chaque pli. Les données ont déjà été explorées : cette validation ne devient pas un test indépendant.

| Procédure sur les mêmes plis | RMSE |
| --- | ---: |
| Sélection des variables étendues | 541.942903152 |
| Recette fixe du précédent meilleur modèle | 542.030725394 |

Gain descriptif : 0.087822243, intervalle bootstrap clients 95 % [-0.02903266884801212, 0.2086924317671446]. Les trois plis s'améliorent, mais l'intervalle inclut zéro.

Configuration finale : `{'kind': 'ridge', 'panel': 'all', 'alpha': 10.0}`, poids **0.25**. La fréquence reste exactement celle d'origine. Le fichier est `deliverables/candidates/submission_variable_models.csv`, 50 000 lignes index,pred ; il est reproduit octet pour octet dans un processus distinct. Les 46 tests passent. Le notebook protégé est inchangé.

```powershell
.\.venv\Scripts\python.exe scripts/reanalyse_variables.py
.\.venv\Scripts\python.exe scripts/variable_models.py train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/variable_models.py predict --output reports/variable_models/reproduced.csv
.\.venv\Scripts\python.exe scripts/verify_variable_models.py
```

Les commandes de vérification consignent les scores exactement transmis dans cette conversation ; elles ne les récupèrent pas sur Kaggle. L'accès automatisé au navigateur reste bloqué par le réglage enregistré.
