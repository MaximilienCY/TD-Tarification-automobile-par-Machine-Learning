# Reprise XGBoost : variables, fréquence et gravité

Cette recherche repart des variables brutes et conserve les expériences précédentes. Le notebook original est vérifié par SHA-256 à chaque chargement des données. Le fichier `submission_reference.csv` reste le témoin ayant le meilleur score public vérifié dans ce projet : 1365.943. Aucun score public du nouveau modèle ne doit être inventé.

## Variables et anomalies

`reports/restart_xgboost/variable_dictionary.csv` décrit toutes les colonnes, leur rôle, les valeurs manquantes, les cardinalités, les bornes et les différences train/test (statistique KS pour les nombres, modalités inconnues pour les catégories). `variable_groups.csv` donne les effectifs, taux de sinistre, gravité conditionnelle et coût moyen par modalité ou décile. Les catégories les plus rares sont regroupées uniquement pour cette présentation descriptive.

Trois panneaux sont comparés : compact actuariel, toutes les variables brutes admissibles, puis variables enrichies (département, puissance/poids et interactions contrat/usage). Chaque panneau existe en version brute avec indicateurs et en version nettoyée avec indicateurs. Poids/cylindrée non positifs et permis supérieurs à l'âge deviennent manquants dans la version nettoyée. Le second conducteur absent est explicite ; ses âges et permis sont manquants dans la version nettoyée, structurellement nuls dans la version brute. Les dates de commercialisation et durées n'ont pas de référence temporelle suffisamment documentée pour imposer un calcul d'exposition. Aucun sinistre n'est tronqué ou retiré.

Les catégories sont apprises uniquement sur le sous-ensemble utilisé pour ajuster chaque arbre. Une catégorie inconnue devient manquante pour XGBoost. Aucun encodage cible supplémentaire n'est appliqué. Les identifiants, sexes et cibles sont exclus par liste explicite de prédicteurs.

### Constats sur les 50 000 contrats d'apprentissage

Il existe 45 785 clients et 2 917 sinistres (5,834 %). La gravité moyenne parmi les sinistres vaut 1 771,84, sa médiane 1 236, son quantile 99 % 6 963,12 et son maximum 21 826,96. Les 25 plus grands coûts représentent 21,51 % de la somme des coûts au carré : leur suppression changerait l'objectif évalué.

| Variable / groupe | Effectif | Taux de sinistre | Coût moyen tous contrats |
| --- | ---: | ---: | ---: |
| Contrat Maxi | 32 360 | 7,48 % | 138,04 |
| Contrat Median1 | 4 638 | 3,00 % | 45,35 |
| Contrat Median2 | 8 759 | 3,23 % | 44,16 |
| Contrat Mini | 4 243 | 1,74 % | 24,62 |
| Usage professionnel | 3 667 | 7,88 % | 152,53 |
| Véhicule de 1 à 5 ans | 16 567 | 8,40 % | 159,15 |
| Véhicule de plus de 20 ans | 3 610 | 1,14 % | 14,28 |
| Second conducteur absent | 33 406 | 5,57 % | 98,24 |
| Second conducteur présent | 16 594 | 6,36 % | 113,70 |

Ces effets sont confondus entre eux : par exemple, âge du véhicule et couverture peuvent se recouvrir. La modalité `AllTrips` ne contient que 48 contrats, donc son taux empirique est fragile. Il ne faut pas imposer de monotonie à partir de ces seules moyennes.

La corrélation entre âge du véhicule et début/fin de vente vaut 0,959/0,949 ; celle entre âge du conducteur et ancienneté du permis vaut 0,919 ; puissance et prix sont corrélés à 0,887. Le panneau compact teste si ces redondances nuisent à la généralisation. La plus forte statistique KS train/test parmi les prédicteurs numériques vaut seulement 0,00556 (poids) : aucune forte différence marginale numérique n'apparaît ici, sans que cela prouve l'absence de changement de risque. Dans le test, 19 lignes portent une marque inconnue, 206 un modèle inconnu et 7 482 un code postal absent du train. Le département offre une représentation géographique moins fragmentée que les 12 790 codes postaux distincts d'apprentissage.

## Recherche et évaluation

- 60 configurations déterministes par composante, graine 20261012 ; panneaux et nettoyages équilibrés, objectifs de gravité alternés par blocs. Profondeur 1–5, taux 0,01–0,08, sous-échantillonnage et colonnes 0,7–1, L2 0,1–300, L1 parmi 0/0,01/0,1/1/10. Le poids minimal d'enfant explore 1–100 en fréquence et 5–200 en gravité. Aucun `scale_pos_weight` n'altère les probabilités.
- Fréquence `binary:logistic`, gravité `reg:gamma` ou `reg:squarederror`. La gravité est ajustée sur les seuls sinistres, divisée par sa moyenne d'apprentissage puis remise à l'échelle. Les prédictions négatives de la variante quadratique sont bornées à une petite valeur positive. [Documentation XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
- Jusqu'à 3 000 arbres, arrêt après 80 tours sans amélioration sur 20 % des clients du sous-ensemble d'apprentissage. Ces clients ne sont jamais ceux du pli évalué. Le modèle est ensuite réajusté sur tout son apprentissage au nombre de tours retenu ; les catégories sont réapprises localement.
- Trois plis externes, cinq internes. Les cinq meilleures fréquences par log-loss et gravités par RMSE conditionnelle sont retenues. Les fréquences brutes et calibrées, et les deux composantes de référence, sont combinées avec des poids 0/0,25/0,5/0,75/1 ; le produit est classé par RMSE de prime. La référence est toujours candidate.
- Calibration logistique croisée sur les prédictions internes hors pli : le calibrateur d'un pli voit les autres plis. Ces prédictions proviennent de modèles dont les apprentissages se recouvrent ; cette étape est une sélection interne et ne constitue pas une évaluation indépendante. Pour prédire le pli externe, le calibrateur utilise toutes les prédictions internes et jamais les étiquettes externes.
- L'évaluation externe réajuste seulement les composantes retenues et mesure RMSE/MSE/MAE/biais, log-loss/Brier/AUC, gravité conditionnelle, calibration et résultats par contrat. Bootstrap apparié par client (2 000 réplications), conditionnel aux prédictions apprises ; il n'intègre pas toute l'incertitude du réentraînement. Les données ont déjà été explorées avant cette recherche.

La sélection finale répète la recherche sur tout le train après l'évaluation externe. Si la borne inférieure du gain bootstrap n'est pas strictement positive, la référence reste recommandée ; le nouveau fichier est seulement un challenger. Les résultats locaux et le score Kaggle sont distingués. La décomposition fréquence × gravité est également illustrée dans l'[exemple actuariel scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).

## Reproduction

Depuis la racine du dépôt, dans PowerShell :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -r requirements-research.txt
.\.venv\Scripts\python.exe scripts/restart_xgboost.py all
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/build_restart_notebook.py
.\.venv\Scripts\python.exe scripts/restart_xgboost.py predict --output reports/restart_xgboost/reproduced.csv
```

Pour utiliser trois processus sur cette machine, remplacer uniquement la commande `restart_xgboost.py all` par `.\.venv\Scripts\python.exe scripts/run_restart_parallel.py`. Les recherches des partitions sont indépendantes ; leurs caches sont ensuite relus pour l'évaluation et l'export. Ne pas exécuter les versions parallèle et séquentielle simultanément sur le même cache.

La version parallèle mémorise les empreintes des partitions immuables pendant chaque sélection pour éviter de recalculer leur hachage à chaque essai. Un test vérifie que ses clés de cache restent exactement identiques à celles du pipeline canonique ; les paramètres, ajustements et résultats statistiques sont inchangés.

La recherche conserve un cache indexé par données, code, configuration et versions. Relancer la commande reprend les ajustements identiques déjà terminés. Les scores sont dans `reports/restart_xgboost/`, le modèle dans `models/restart_xgboost.joblib`, et le challenger dans `deliverables/candidates/submission_restart_xgboost.csv`. Les fichiers précédents ne sont pas remplacés. Le notebook exécuté est `notebooks/04_reprise_xgboost.ipynb`.

L'export vérifie les 50 000 index dans leur ordre, le schéma `index,pred`, des nombres finis non négatifs et la reproduction des prédictions originales à moins de 10⁻⁶. Le modèle est rechargé avant export ; la reproduction dans un second processus est vérifiée séparément. L'accès navigateur Kaggle reste soumis à la préférence enregistrée de blocage du site : aucune soumission ne sera déclarée sans confirmation effective de Kaggle.

La [documentation officielle du navigateur](https://learn.chatgpt.com/docs/browser) situe les exceptions de sites dans **Settings > Browser > Agent permissions**. Supprimer une permission personnalisée restaure les valeurs par défaut ; cela ne prouve pas que l'accès ou la soumission fonctionne. Cette modification appartient à l'utilisateur ; le pipeline d'apprentissage ne modifie aucun réglage de sécurité.

Après entraînement, régénérer le notebook et les prédictions dans un second processus, puis exécuter `.\.venv\Scripts\python.exe scripts/verify_restart.py` pour vérifier les artefacts et inscrire le challenger dans le registre local avec un score Kaggle nul tant qu'il n'est pas confirmé.
