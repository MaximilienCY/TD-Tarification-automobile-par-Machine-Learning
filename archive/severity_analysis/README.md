# Nouveau modèle de gravité : analyse et préparation

[Ouvrir le notebook](notebook.ipynb). Ce travail utilise les données locales du
dépôt et suit la partie « Analyse des données » de la roadmap, puis réalise le
découpage 80/20, la création du département et des interactions véhicule / conducteur,
ainsi que le traitement des NaN.

La cible est le montant du sinistre **strictement positif**. Le nombre de
sinistres sert à vérifier la population ; il ne fait pas partie des entrées du
modèle. La section 6 compare des ensembles de variables par validation croisée ;
aucun modèle final n'est ajusté ni sauvegardé dans ce dossier.

## Contenu du notebook

- Types, dictionnaire de toutes les variables, distributions, manquants,
  absences structurelles, anomalies, doublons et clients avec plusieurs contrats.
- Distribution des montants positifs, quantiles, répétitions et concentration
  du coût dans les gros sinistres.
- Corrélations **Pearson et Spearman** avec `montant_sinistre` chez les sinistrés
  de développement, redondances et coûts par groupe avec effectifs. Une synthèse
  après préparation couvre toutes les entrées, y compris les variables créées
  et les indicateurs de manquant. Les catégories ont une association η² ; les
  variables constantes sont signalées comme non calculables.
- Variable **`departement`**, catégorielle, extraite du champ `code_postal` et
  remplaçant le code détaillé dans les entrées. Les codes `2A` / `2B` sont
  conservés ; les départements d'outre-mer sont prévus sur trois caractères.
  Le notebook présente les coûts par département et une association descriptive
  η², sans donner de sens numérique au numéro de département ni conclure à un
  effet causal sur les réparations.
- Variable **`age_puissance_vehicule = anciennete_vehicule * din_vehicule`** :
  interaction multiplicative, avec conservation des deux variables sources.
  Son produit est recalculé après l'imputation de l'âge et de la puissance ;
  un indicateur conserve l'absence éventuelle d'une source. Les corrélations
  restent descriptives et ne démontrent pas encore un gain prédictif.
- Variable **`puissance_anciennete_permis1 = din_vehicule * anciennete_permis1`** :
  croisement de la puissance avec l'ancienneté du permis principal. Les codes
  de permis suspects sont rendus manquants avant le calcul ; les produits sont
  reconstitués à partir des sources imputées avec les médianes d'apprentissage.
- Variable **`vitesse_age_conducteur1 = vitesse_vehicule * age_conducteur1`** :
  croisement avec l'âge du conducteur principal, selon le choix retenu.
  La vitesse est une caractéristique du véhicule et ne décrit pas la conduite
  réellement observée. Les deux sources restent dans les entrées du modèle.
- Découpage par client avec `GroupShuffleSplit(test_size=0.20, random_state=2026)`
  sur le portefeuille complet, suivi du filtre positif dans chaque jeu :
  **2 356 sinistres d'apprentissage et 561 de validation**, sans client commun.
  Le 80/20 concerne les clients ; les proportions de sinistres peuvent différer.
- NaN numériques remplacés par les médianes des seuls sinistrés d'apprentissage ;
  catégories manquantes explicites et indicateurs de manquant. Les codes suspects
  et les absences du second conducteur sont distingués avant l'imputation.
- Choix des variables par trois plis séparés par client sur les 2 356 sinistres
  de développement : mêmes réglages de boosting Gamma pour chaque ensemble,
  médianes et encodages réappris dans chaque sous-train. Parmi les ensembles à
  moins de 5 € de la meilleure RMSE hors pli, le plus simple est recommandé.
  Cette tolérance est un choix pratique, pas un intervalle de confiance.

Les statistiques de coûts et les corrélations utilisent seulement le
développement. Côté validation, seul l'effectif positif nécessaire au découpage
est affiché. Les données ont déjà été explorées historiquement : cette réserve
ne constitue pas une validation externe neuve.

Les scores de sélection de la section 6 servent à choisir les variables et ne
constituent pas une évaluation finale indépendante. Le banc d'essai utilise
`HistGradientBoostingRegressor(loss='gamma')` de scikit-learn, avec réglages
fixes. Ses résultats ne sont pas ceux du XGBoost historique du dépôt. Un autre
modèle ou d'autres réglages peuvent modifier l'intérêt des variables.

Pour cette recette de comparaison, le choix parcimonieux retient **`type_contrat`,
`anciennete_vehicule` et `departement`** : RMSE hors pli **1 450,85 €**,
contre **1 461,50 €** pour la moyenne constante. L'ensemble contrat / prix /
ancienneté obtient **1 450,87 €** : cet écart de 0,02 € ne permet pas d'affirmer
une supériorité stable du département sur le prix. Les quatre variables ensemble
obtiennent **1 451,17 €**. Le prix reste donc une alternative crédible, sans gain
mesuré ici à l'ajouter. Les variables non retenues restent dans les jeux préparés.
Le choix est consigné dans `reports/selection_variables.json` ; la validation
20 % n'a pas été utilisée pour le faire.

## Réexécuter

Avec un environnement Python disposant des dépendances de `requirements.txt`,
ouvrir le notebook et exécuter les cellules dans l'ordre, ou depuis la racine du
dépôt :

```sh
python3 nouveau_notebook/run_notebook.py
```

Le script conserve les sorties dans le notebook et utilise son propre
interpréteur pour le noyau. Il ne télécharge rien et n'installe rien.
L'option `--in-process` permet aussi une exécution IPython sans socket locale,
utile dans un environnement restreint.

## Résultats

- `reports/` : tableaux agrégés, graphiques, synthèse, médianes utilisées et
  manifeste contenant les versions et empreintes des sources.
- `prepared/` : jeux locaux `train_gravite.csv`, `validation_gravite.csv`,
  `test.csv`, affectations des clients et préparation ajustée sauvegardée.
  Ce dossier est exclu du suivi Git.
- `preparation_gravite.py` : département, trois interactions et transformation
  réutilisable `PreparationGravite`. Les sorties restent des DataFrames avec
  catégories textuelles ; leur encodage dépendra du modèle choisi.

Les jeux préparés comportent **46 variables** : 28 variables candidates et
18 indicateurs de manquant. L'index technique est ajouté aux exports pour les
rapprochements et la cible seulement aux jeux d'apprentissage et de validation ;
ils ne font pas partie des variables explicatives.

Pour une future validation croisée, ajuster une nouvelle instance de
`PreparationGravite` dans chaque sous-train, par exemple dans une pipeline.
Réutiliser l'instance déjà ajustée sur tout le développement dans les plis
introduirait une fuite sur les médianes.

## Début et fin de vente

Les champs `debut_vente_vehicule` et `fin_vente_vehicule` concernent la
commercialisation du véhicule. Leur interprétation provisoire est une ancienneté
depuis le début et la fin de commercialisation du modèle automobile. Leur codage
et leur unité exacts restent à confirmer ; ils ne sont pas convertis en dates
calendaires et ne servent pas à calculer une durée d'exposition du contrat.
