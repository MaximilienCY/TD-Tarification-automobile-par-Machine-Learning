# Tarification automobile — CY Tech 2027

Deux modèles produisent la prime pure :

**prime = P(sinistre | contrat) × E(coût | sinistre, contrat)**

La sélection minimise la **MSE de prime**. La RMSE est également affichée en unités de coût. La comparaison complète et le score Kaggle sont documentés dans [les résultats](docs/results.md).

**Meilleur score public transmis : 1365.657**, pour [submission_frequency_curves.csv](deliverables/candidates/submission_frequency_curves.csv). Le modèle reste **fréquence × gravité** : la gravité est exactement celle du modèle à 1365.745, et la fréquence originale reçoit une correction régularisée de logit avec courbes d'âge du véhicule, couverture/usage et interaction âge × couverture. La procédure améliore les trois plis externes et la RMSE de **541,9710 à 541,9156** ; log-loss et Brier s'améliorent aussi. L'intervalle descriptif du gain vaut [0,00465 ; 0,11079], sans constituer une preuve sur un test indépendant. Les recherches supplémentaires de gravité sont écartées. [Résultats et commandes](docs/variable_followups.md), [notebook exécuté](notebooks/07_recherches_apres_1365_745.ipynb). **57 tests** passent ; le CSV est reproduit octet pour octet et le notebook protégé reste inchangé. Le seuil strictement inférieur à 1365.500 reste non atteint.

Une [deuxième recherche documentée](docs/research_round2.md) compare davantage de modèles, notamment EBM et XGBoost. Ses candidats et résultats sont séparés de la première livraison ; aucun nouveau score Kaggle n'est confirmé tant que l'accès au navigateur reste bloqué.

Le [challenger ancré](docs/anchored_model.md) conserve la gravité Gamma initiale et applique une correction limitée à la fréquence logistique. Son score Kaggle **1366.010**, transmis par l'utilisateur, est moins bon que la référence initiale à **1365.943** ; il est écarté.

La [reprise XGBoost](docs/restart_xgboost.md) étudie les variables brutes, compare trois panneaux avec deux traitements des anomalies, et réexécute une sélection fréquence × gravité dans trois plis externes et cinq plis internes. Elle conserve les résultats précédents. Commande complète : `.\.venv\Scripts\python.exe scripts/run_restart_parallel.py` (trois processus) ou `scripts/restart_xgboost.py all` (séquentiel). Ne pas lancer les deux en même temps. Les scores se trouvent dans `reports/restart_xgboost/` ; le notebook de synthèse est généré après la recherche avec `scripts/build_restart_notebook.py`.

Cette reprise est terminée : [résultats et configuration finale](docs/restart_results.md), [notebook exécuté](notebooks/04_reprise_xgboost.ipynb). La RMSE externe de la sélection XGBoost vaut **542,1281**, contre **542,0601** pour la référence initiale. Le score Kaggle **1366.025**, transmis par l'utilisateur, est également moins bon que **1365.943**. La référence reste recommandée. Le [CSV XGBoost](deliverables/candidates/submission_restart_xgboost.csv) est conservé pour traçabilité. Les 30 tests de cette reprise passent, l'export est reproduit octet pour octet dans un second processus et le notebook original est inchangé.

Le [suivi après ces scores](docs/refinement_after_kaggle.md) teste ensuite deux changements séparés : gravité pondérée/décalée et corrections XGBoost, puis logistique à pertes probabilistes pondérées. La nouvelle fréquence est écartée. La gravité seule gagne **0,004410 RMSE** en validation externe, avec un intervalle incluant zéro. Le [CSV de gravité](deliverables/candidates/submission_severity_refinement.csv), conservant exactement les probabilités originales, obtient **1365.913** sur Kaggle selon le résultat transmis par l'utilisateur : il devient notre meilleur modèle public observé, devant 1365.943. Le gain local reste incertain et l'objectif **strictement inférieur à 1365.500** n'est pas atteint. Le résultat et l'empreinte du fichier sont conservés dans `reports/severity_refinement/kaggle_result.json`.

La [recherche suivante](docs/severity_successor.md) (`scripts/improve_severity.py train`) compare des corrections voisines, des moyennes et des corrections additives avec ce modèle à 1365.913. Elle refait la sélection dans cinq plis internes et l'évalue dans trois plis externes séparés par client. La RMSE externe passe de **542,0307** pour la recette fixe du précédent meilleur modèle à **542,0226** pour la sélection, avec un intervalle de gain incluant zéro. Le [successeur](deliverables/candidates/submission_severity_successor.csv) mélange Gamma avec trois corrections XGBoost de 480 arbres ; la fréquence originale reste inchangée. Les **43 tests** passent et le CSV est reproduit octet pour octet. Son score Kaggle transmis, **1365.914**, n'améliore pas 1365.913 : il est écarté. Le [notebook](notebooks/05_successeur_gravite.ipynb) conserve les résultats locaux. Les données ayant déjà été étudiées, ces résultats restent des diagnostics et ne constituent pas un nouveau test indépendant.

## Organisation

Le score public **1365.745** a été obtenu par [submission_variable_models.csv](deliverables/candidates/submission_variable_models.csv), devant 1365.913 et 1365.914. La [réanalyse des variables](docs/variable_reanalysis.md) identifie les résidus de gravité par groupe et leur sensibilité aux grands sinistres. Le nouveau modèle ajoute à 25 % une correction Ridge avec courbes et catégories sur les variables véhicule, conducteur et contrat. Les probabilités originales restent inchangées. La RMSE externe de la procédure vaut **541,9429**, contre **542,0307** pour la recette précédente, avec un intervalle du gain incluant zéro. Le [notebook exécuté](notebooks/06_reanalyse_variables.ipynb) et les vérifications sont disponibles. Le score public est transmis par l'utilisateur et associé au SHA-256 du fichier ; il n'est pas vérifié indépendamment par le navigateur. Le seuil de 1365.500 reste non atteint.

```text
Data/                  train, test et exemple officiel de soumission
src/pricing/           préparation, modèles, validation, export
notebooks/             fréquence, gravité, tarification commune
reports/               scores, choix de modèles, provenance, score Kaggle
models/                modèle final généré localement (hors Git)
tests/                 contrôles de fuite, robustesse et format
docs/                  compétition, résultats, support de présentation
deliverables/          présentation commune PowerPoint
scripts/               génération des notebooks exécutés
run.py                 point d'entrée unique
submission.csv         fichier de prédictions final
```

`Calcul de la prime d'assurance.ipynb` est conservé **strictement à l'identique** à la racine. Il reste un document historique et ne fait pas partie du pipeline exécutable. Ses chemins relatifs d'origine n'ont pas été corrigés, conformément à la demande de ne jamais le modifier.

## Installation

Utiliser Python **3.12**. Dans PowerShell, depuis la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` liste les dépendances directes ; `requirements-lock.txt` fige l'environnement complet utilisé. Sous Linux/macOS, remplacer `.venv\Scripts\python.exe` par `.venv/bin/python`.

## Reproduire

```powershell
# Recherche des modèles et mélanges sur cinq plis par client
.\.venv\Scripts\python.exe run.py search

# Évaluation de toute la recherche : trois plis externes et trois internes
.\.venv\Scripts\python.exe run.py nested

# Diagnostic historique, entraînement final, modèle et submission.csv
.\.venv\Scripts\python.exe run.py finalize

# Vérifications et notebooks explicatifs exécutés
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/build_notebooks.py
```

`python run.py all` enchaîne recherche, validation imbriquée et entraînement final. Le calcul complet entraîne de nombreux modèles et prend plusieurs minutes. Les prédictions intermédiaires sont mises en cache dans `.cache/` ; leur clé inclut les données, le code, les versions et la configuration. Une modification de ces éléments entraîne un nouveau calcul.

Pour régénérer uniquement les prédictions à partir du modèle local :

```powershell
.\.venv\Scripts\python.exe run.py predict
```

Les notebooks lisent les résultats produits par ces commandes. Ils ne chargent aucun code depuis les cellules d'un autre notebook.

La [présentation commune](deliverables/presentation_tarification.pptx) résume la méthode, les résultats et leurs limites. Les trois notebooks livrés sont exécutés sans erreur. Une exécution séparée de `run.py predict`, depuis le dossier parent, reproduit exactement le fichier de soumission ; les empreintes sont consignées dans `reports/verification.json`.

## Méthode

- 50 000 contrats, 45 785 clients et 2 917 sinistres dans l'apprentissage.
- Partition par client : 40 013 contrats de développement et 9 987 de diagnostic historique.
- Fréquence : logistique, splines/interactions, HistGradientBoosting, CatBoost, LightGBM, constante.
- Gravité : Gamma, splines, Poisson à lien log, Ridge, CatBoost, LightGBM, constante.
- Comparaison de mélanges convexes prédéfinis ; aucune optimisation sur les cibles du diagnostic historique.
- Validation imbriquée avec sélection refaite dans chaque apprentissage externe ; comparaison au modèle de référence et au précédent modèle optimisé sur les mêmes contrats.
- Conservation des coûts extrêmes ; bootstrap apparié par client pour décrire l'incertitude du gain.

Les variables cibles, identifiants et sexes sont exclus des entrées prédictives. Les identifiants clients servent au découpage et à l'évaluation. Les modèles de gravité sont appris uniquement sur les coûts positifs. Les données officielles sont identiques à celles du dépôt, vérifiées par empreinte SHA-256.

## Soumission Kaggle

[Compétition privée](https://www.kaggle.com/competitions/cy-tech-2027-calcul-de-la-prime-dassurance/overview) : charger `submission.csv` via **Submit Prediction**. Le fichier contient exactement **50 000 lignes**, colonnes **index,pred**, dans l'ordre du fichier officiel `Data/prime_pred_sandbox.csv`.

Le leaderboard public ne porte que sur 33 % du test. Un meilleur score public ne garantit pas le meilleur score privé. Les résultats locaux, le diagnostic historique déjà examiné et le score public sont présentés séparément.

## Nettoyage et conservation

Les anciens notebooks de modélisation, les exports dupliqués, les anciens pickles et les dessins devenus inutiles ont été retirés du dossier actif. Les fonctions utiles sont migrées dans `src/pricing/reference.py` ; le protocole précédent reste documenté dans `reports/previous_model.json`.

Le commit original `566e1b5` reste dans l'historique Git. Une archive vérifiée `original-project-566e1b5.zip` se trouve également dans le dossier parent local. L'empreinte SHA-256 du notebook protégé est vérifiée lors de chaque exécution et dans les tests.
