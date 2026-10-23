# Modèle final de gravité automobile — XGBoost Gamma unique

Le modèle retenu prédit le **coût moyen d’un sinistre, sachant qu’un sinistre existe**.
Il utilise **quatre variables** et conserve le **découpage 80/20 initial par client**.
Aucun modèle de fréquence n’entre dans cette version finale.

**[Ouvrir le notebook de référence](gravite_xgboost/00_modele_final_gravite.ipynb)**
— synthèse exécutable du modèle retenu, du découpage 80/20 et de ses résultats agrégés.

## Modèle retenu et fichiers à utiliser

- **XGBoost Gamma avec lien logarithmique** : coût prédit positif.
- **Variables** : `type_contrat`, `prix_vehicule`, `anciennete_vehicule`, `classe_geographique`.
- **Réglages** : 400 arbres, profondeur 1, taux d’apprentissage 0,03 ; détails dans le manifeste.
- **Modèle prêt à utiliser** : [`gravite_final.joblib`](gravite_xgboost/models/gravite_final.joblib).
- **Configuration et contrôles** : [`gravite_final.json`](gravite_xgboost/models/gravite_final.json).
- **Scores agrégés** : [`scores.csv`](gravite_xgboost/reports/final/scores.csv).
- Les 50 000 coûts conditionnels du test sont générés localement par le script de prédiction ; les exports par contrat ne sont pas publiés.

Le prix et l’ancienneté sont repris du dataset, avec remplacement des manquants
par les médianes d’apprentissage. Le contrat est une catégorie observée. La classe
géographique (urbain dense, urbain intermédiaire, rural, inconnu) vient du référentiel
INSEE embarqué, rapproché des codes communaux. La taille de commune a été retirée.
Les catégories sont encodées en indicatrices ; les valeurs numériques ne sont pas
standardisées ni transformées en logarithme dans cette recette.

## Découpage final 80/20 et résultats

Le découpage sépare les clients du portefeuille complet avec
`GroupShuffleSplit(test_size=0.20, random_state=2026)` :

- **Apprentissage : 39 964 contrats, dont 2 356 sinistres positifs.**
- **Validation : 10 036 contrats, dont 561 sinistres positifs.**
- **Aucun client commun entre les deux jeux.**

**RMSE sur les 561 sinistres de validation : 1 463,81 €** ; **MAE : 930,49 €**.
Le modèle sauvegardé est ajusté uniquement sur les **2 356 sinistres d’apprentissage**.
Les lignes de validation ne servent pas à son ajustement et aucun réentraînement
sur ces lignes n’est effectué après l’évaluation.

La RMSE hors pli sur les trois plis par client du développement est de **1 448,34 €**.
Ce score et le score du jeu de validation 20 % relèvent de protocoles différents.
Les scores 90/10 des essais historiques ne sont pas ceux du modèle final 80/20.
Les données ont déjà été explorées dans plusieurs expériences ; ces résultats ne
constituent pas une validation externe neuve.

## Pourquoi conserver le modèle unique ?

La recherche initiale a comparé plusieurs réglages. Pour les essais Gamma, deux
niveaux de complexité ont principalement été comparés ; aucune recherche exhaustive
dédiée aux quatre variables finales n’a été menée. Les paramètres restent figés et
les 20 % de validation ne sont pas utilisés pour les optimiser.

Les modèles par tranches ajoutent une orientation et plusieurs experts sans gain
stable démontré. Sur trois plis du développement, le modèle unique obtient
1 448,34 € de RMSE, contre 1 450,00 € pour deux tranches. Sur le découpage historique
90/10, les petits écarts entre modèle unique, deux et trois tranches ne permettent
pas d’affirmer une supériorité générale. Le modèle unique est retenu pour sa simplicité.

La synthèse publiée explique la cible, les quatre variables, les transformations,
la validation, le choix du modèle unique et ses limites. Le notebook de recherche
complet et les rapports individuels sont conservés dans le travail local.

## Utiliser le modèle

Depuis la racine du dépôt, avec Python 3.12 :

```sh
python3 -m venv gravite_xgboost/.venv
gravite_xgboost/.venv/bin/python -m pip install -r gravite_xgboost/requirements.txt
gravite_xgboost/.venv/bin/python gravite_xgboost/predict_final.py --input Data/test.csv --output gravite_predictions.csv
```

Le fichier d’entrée doit conserver le format du test initial. La pipeline génère
les quatre variables utilisées ; le résultat contient `index,gravite_predite`.
Les sorties sont des **gravités conditionnelles**, pas des primes.

Pour reconstruire seulement l’export final, avec le même 80/20 :

```sh
gravite_xgboost/.venv/bin/python gravite_xgboost/run_final_notebook.py
```

Les dépendances sont figées dans `gravite_xgboost/requirements.txt`. Les données,
les sources et le modèle sont identifiés par leurs empreintes dans le manifeste.
Seuls les scores et effectifs agrégés sont publiés dans
[`reports/final/`](gravite_xgboost/reports/final/). Les découpages clients et les
prédictions par contrat restent locaux et sont exclus du suivi Git. Les limites
du rapprochement géographique sont détaillées dans le notebook de synthèse.
