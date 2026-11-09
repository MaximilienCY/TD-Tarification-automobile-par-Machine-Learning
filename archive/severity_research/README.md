# Nouveau modèle de gravité — démarche complète XGBoost

> **Modèle final retenu : XGBoost Gamma unique, section 16.** Quatre variables : type de contrat, prix et ancienneté du véhicule, classe géographique. Le découpage 80/20 initial est conservé : 2 356 sinistres d’apprentissage et 561 de validation, sans client commun. **RMSE validation : 1 463,81 € ; MAE : 930,49 €.** Le modèle final n’est ajusté que sur les 80 %. Les sections 0–15 documentent les expériences historiques.

Le point de départ est [01_gravite_xgboost_roadmap.ipynb](01_gravite_xgboost_roadmap.ipynb),
livré exécuté avec ses tableaux, graphiques et conclusions. Ce dossier est indépendant
des anciennes recettes : il utilise seulement `../Data/train.csv`, `../Data/test.csv`
et une copie de la roadmap retrouvée dans `previous/`.

Le notebook suit les cases de la roadmap : cadrage statistique ; description de toutes
les variables et des deux cibles ; anomalies, manquants, rareté et redondances ; découpage
par client ; préparation réutilisable ; créations et regroupements ; références ;
optimisation XGBoost ; sélection a posteriori ; erreurs, sens, calibration et sauvegarde.
La première recherche compare 58 essais XGBoost et trois références. Certains retraits de blocs déjà absents
sont sans effet et ne constituent pas des modèles indépendants.

Les variantes testent notamment plusieurs pertes, le logarithme avec/sans correction
interne de retransformation, les seuils de rareté, les familles marque–modèle, les
clusters techniques, les ratios, segments, interactions, imputations, la géographie
et la sensibilité aux coûts extrêmes. Aucune transformation n'est retenue d'office.

## Historique de la première recherche — résultats et limites

La sélection de développement conserve un XGBoost sur **type de contrat, prix du véhicule
et âge du véhicule** : 180 arbres, profondeur 3, taux d'apprentissage 0,02. Les créations
supplémentaires ne sont pas retenues. La validation croisée donne une RMSE de 1 449,73 €.

Sur les 561 contrats sinistrés réservés, la RMSE est de **1 465,77 €**, contre
1 473,70 € pour la moyenne et 1 457,21 € pour la référence Gamma métier. Le gain du
XGBoost face à la moyenne est de 7,92 €, avec un intervalle bootstrap par client
à 95 % de **[-4,85 ; 22,38] €** : le gain reste incertain. La référence Gamma fait mieux
sur ce holdout ; le XGBoost est conservé conformément au périmètre demandé. Les gros
sinistres restent peu prévisibles. Ces métriques concernent la gravité conditionnelle,
sans valeur de score Kaggle ou de validation de prime.

Le holdout sépare les clients et n'a pas servi à sélectionner la recette. Les fichiers
ont toutefois été explorés dans des travaux antérieurs : il ne remplace pas une cohorte
externe neuve. Le modèle réentraîné final utilise les 2 917 contrats sinistrés ; son score
propre sur des observations nouvelles reste inconnu.

## Exécuter

Utiliser Python 3.12 dans un environnement local. Depuis ce dossier :

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_notebook.py
```

Pour une lecture interactive, sélectionner ce même environnement comme noyau et
exécuter toutes les cellules dans l'ordre. Le script utilise son propre interpréteur
pour lancer le noyau. Il n'installe rien automatiquement. Les rapports et modèles
propres à ce dossier sont régénérés lors de l'exécution ; les anciens travaux ne sont
pas modifiés.

`gravite_pipeline.py` contient les transformations et l'estimateur sauvegardable,
affichés et expliqués dans le notebook. Tout le travail d'analyse, la recherche et
la validation restent dans le notebook. Les médianes, regroupements rares et clusters
sont appris sur les seuls sous-trains pendant la validation.

## Fichiers produits

- `reports/journal_essais.csv` : tous les essais, raisons et scores train/hors pli.
- `reports/configurations_essais.json` et `scores_par_pli.csv` : paramètres et stabilité.
- `reports/audit_toutes_variables.csv` et `relations_toutes_variables_cibles.csv` : audit exhaustif.
- `reports/selection_gelee_avant_holdout.json` : choix figé avant évaluation.
- `reports/holdout_*.csv` et `holdout_bootstrap_clients.json` : validation, erreurs et incertitude.
- `reports/importance_permutation_developpement.csv` : importance hors pli des sources brutes.
- `reports/gravite_test.csv` : 50 000 coûts conditionnels, colonnes `index,gravite_predite`.
- `models/gravite_pipeline.joblib` : pipeline complète rechargeable avec le module local.
- `models/gravite_xgboost.json` : booster natif, à accompagner des transformations.
- `reports/manifest.json` : versions, empreintes des données et contrôles de reproduction.
- `reports/couverture_roadmap.csv` et les images : correspondance avec la démarche et figures.

Pour recharger la pipeline depuis ce dossier, avec les mêmes dépendances :

```python
import joblib
import pandas as pd
from gravite_pipeline import SeverityEstimator

model = joblib.load('models/gravite_pipeline.joblib')
test = pd.read_csv('../Data/test.csv', dtype={'code_postal': str})
pred = model.predict(test)
```

`gravite_test.csv` **n'est pas une soumission de prime**. Il doit être joint par `index`
aux probabilités d'un modèle de fréquence validé, puis multiplié par ces probabilités.
Le notebook explique les contrôles nécessaires. Aucune soumission n'a été effectuée.

## Extension métier : géographie et profils automobiles

La section 8 du même notebook ajoute 20 comparaisons annoncées : classes Insee
urbain dense/intermédiaire/rural et sept niveaux détaillés ; population communale ;
profils performants selon trois règles puissance/poids/vitesse ; classes d'ancienneté,
dont ≥30 ; segments de prix ; croisements avec la couverture et présence des variables
brutes marque/modèle. Ces catégories métier n'avaient pas été testées dans la première
recherche, qui ne comparait que des préfixes et des regroupements techniques génériques.

Le référentiel externe est la grille Insee en géographie 2021, avec population 2018.
Les fichiers source, normalisés, les URL et les empreintes sont dans `assets/`.
`prepare_geography.py` régénère la table normalisée sans téléchargement automatique.
Les codes des arrondissements Paris/Lyon/Marseille sont rattachés à leur commune ;
les codes non rapprochés restent inconnus. La date des contrats étant absente,
la compatibilité historique de ce référentiel n'est pas garantie.

`metier_features.py` définit les nouveaux indicateurs et leur pipeline. « Profil
performant » est un proxy explicite, pas une catégorie constructeur certifiée.
« Véhicule ≥30 » n'atteste pas un statut collection ; la valeur assurée actuelle
n'est pas renseignée. Les effectifs et coûts observés sont affichés avant les modèles.

Les trois plis externes séparent les clients. Une sélection des 20 nouvelles
configurations sur trois plis internes est évaluée dans chaque pli externe, en plus
de la comparaison ordinaire. Cette analyse reste conditionnelle aux hyperparamètres
issus de la recherche initiale et aux données déjà explorées. Le holdout initial
n'est pas utilisé pour mesurer ou sélectionner cette extension.

Les résultats et la validation imbriquée sont dans `reports/metier/`. Le meilleur
candidat de développement et ses 50 000 prédictions sont conservés séparément dans
`models/gravite_metier_candidat.joblib` et `reports/metier/gravite_test_candidat.csv`.
La pipeline initiale n'est pas remplacée automatiquement. Le statut exploratoire,
les variables du candidat et les limites sont consignés dans `manifest_metier.json`.

Résultat de l’extension : meilleur candidat ordinaire `geo_3` (RMSE 1 449,28 €), soit seulement 0,45 € de gain. La sélection imbriquée ne confirme pas un gain stable ; consulter les intervalles dans le notebook. Le candidat comporte les trois variables initiales et `geo_densite3`.

## Recherche élargie — variables proposées et anciennes variables

La **section 9** retrace la recherche élargie historique. Elle remet en concurrence
les variables brutes auparavant non retenues et les nouvelles créations : véhicule
récent, prix relatif stabilisé par âge, puissance/poids, indicateurs jeune/novice,
âge d'obtention du permis, bonus minimal, interactions, géographie, familles et
clusters automobiles. Le second conducteur absent ne devient pas un faux jeune ou
novice ; les permis incohérents sont nettoyés avant les calculs.

La version complète comporte 62 variables avant one-hot. Les modalités marque/modèle
et les communes/départements sont comparés en one-hot ou via des moyennes hiérarchiques
lissées de gravité, des taux de fréquence auxiliaires, ou les deux. Chaque encodage
supervisé d'apprentissage exclut le client de la ligne, dans un découpage interne
au sous-train courant. Les taux auxiliaires et prix de référence utilisent uniquement
les contrats du sous-train autorisé, sans clients de validation.

Le plan compare 16 ensembles × deux objectifs XGBoost (quadratique/Gamma) × deux
réglages, soit 64 candidats de départ. Les meilleurs représentants des deux objectifs
subissent ensuite des ablations. Cette procédure est rejouée intégralement dans trois
plis externes du développement. Le holdout initial n'est pas réutilisé pour sélectionner
ou mesurer ces nouveaux modèles ; les limites liées aux données déjà explorées restent
applicables.

Les résultats sont dans `reports/variables_elargies/` : classement, scores par pli,
choix externes, prédictions hors apprentissage, incertitude, configurations, audit et
variables présentes dans chaque modèle. Deux modèles finaux sont réentraînés :

- `models/gravite_variables_elargies_selectionne.joblib` : recette choisie sur le développement.
- `models/gravite_variables_elargies_toutes_variables.joblib` : version complète, conservée pour comparaison.

Les prédictions correspondantes sont `gravite_test_selectionne.csv` et
`gravite_test_toutes_variables.csv` dans ce sous-dossier de rapports. Ce sont des gravités
conditionnelles, pas des primes ni des soumissions Kaggle. Les JSON de booster doivent
être accompagnés des transformations ; les pipelines joblib et le module
`enriched_features.py` permettent une utilisation complète.

Pour exécuter uniquement la section 9 et ses deux cellules de contexte, en conservant
les sorties antérieures :

```sh
python run_notebook.py --section 9
```

Le lancement sans argument réexécute tout le notebook. Les contrôles ciblés de fuite,
prix de référence, second conducteur, catégories inconnues et rechargement se lancent
avec `python -m pytest -q -p no:cacheprovider tests/test_enriched_features.py` (pytest
est une dépendance optionnelle de vérification).

## Variables qualitatives — gravité uniquement

La section 10 étudie explicitement la définition et la sélection des catégories :
ancienneté, valeur/gabarit et performance des véhicules ; identité automobile avec
repli des modèles rares vers leur marque ; profils d'âge et d'expérience des
conducteurs ; bonus, couverture et usage ; classes géographiques ; croisements
couverture × âge du véhicule / puissance-poids / usage / géographie.

Elle compare des seuils métier aux quartiles appris sur le sous-train, des seuils
de rareté 10/25, le modèle complet au premier token, une représentation entièrement
qualitative à une représentation mixte et à l'ajout des variables brutes. Les
catégories sont créées sans consulter les coûts. Les coûts moyens, médianes et
effectifs sinistrés de chaque modalité sont ensuite examinés. Les groupes de moins
de 30 sinistres sont signalés. Une absence de second conducteur possède sa propre
catégorie et n'est jamais assimilée à un jeune ou novice.

Le plan comporte 64 candidats (16 représentations × deux objectifs XGBoost × deux
réglages), puis des ablations. Toute cette sélection est évaluée dans des plis
externes par client. Aucun modèle de fréquence n'est entraîné, et aucun taux de
sinistres n'est utilisé comme encodage dans ces nouvelles pipelines.

Les rapports sont dans `reports/variables_qualitatives/` : descriptions de toutes
les modalités, audit de granularité, scores et paramètres, choix par pli externe,
incertitude, décisions sur les variables et manifeste. Deux pipelines sont
sauvegardées : `models/gravite_qualitative_complet.joblib` pour conserver toutes les
catégories définies et `models/gravite_qualitative_selectionne.joblib` pour la
recette choisie sur le développement. Leurs 50 000 prédictions sont respectivement
`gravite_test_complet.csv` et `gravite_test_selectionne.csv` dans ce dossier de
rapports. Ces fichiers contiennent uniquement la gravité conditionnelle.

Réexécution ciblée : `python run_notebook.py --section 10`. Le lancement sans
argument conserve l'exécution complète de toutes les étapes. Les contrôles sont
`python -m pytest -q -p no:cacheprovider tests/test_qualitative_features.py`.

## Retrait de la taille de commune — expérience historique

La section 11 retire uniquement `classe_taille_commune`. La classe géographique
urbain dense / urbain intermédiaire / rural reste présente. Les réglages Gamma
et les trois plis par client sont identiques à ceux du modèle à cinq variables.
Le RMSE global des 2 356 prédictions hors pli passe de **1 448,379763 €** à
**1 448,339904 €** (−0,039859 €) : l’écart est négligeable, sans dégradation mesurée.
Aucune nouvelle recherche de paramètres ni évaluation du holdout n’a été ajoutée.

Cette expérience a enregistré `models/gravite_sans_taille_commune.joblib`, réentraîné
sur les 2 917 sinistres positifs. Ses quatre variables sont `type_contrat`,
`prix_vehicule`, `anciennete_vehicule` et `classe_geographique`. Le référentiel
INSEE reste nécessaire pour cette dernière. Le modèle à cinq variables du §10
reste conservé sous son nom historique pour retracer la comparaison.

Les résultats sont dans `reports/ablation_taille_commune/` : comparaison globale,
scores par pli, prédictions hors pli, manifeste et `gravite_test.csv` (50 000 coûts
conditionnels). Les prédictions après rechargement et l’absence de dépendance à la
population communale ont été vérifiées. Ce score porte sur le développement déjà
exploré ; il ne constitue pas une mesure sur une nouvelle cohorte indépendante.

Réexécution ciblée : `python run_notebook.py --section 11`.

## Répartition descriptive des montants

La section 12 affiche la distribution de `montant_sinistre` après le filtre
`montant_sinistre > 0` : seuls les 2 917 montants positifs sont représentés.
L’histogramme couvre toute la plage des coûts ; la répartition par tranches et
ses pourcentages portent également sur les seuls sinistres positifs. La figure
actualisée est `reports/distribution_sinistres/repartition_montants_positifs.png`.
Il s’agit d’une description de tout le fichier, sans nouvelle sélection ni
modification du modèle à quatre variables du §11.

Les deux figures, les statistiques JSON, les tranches et les quantiles sont dans
`reports/distribution_sinistres/`. Réexécution ciblée :
`python run_notebook.py --section 12`.

## Essai de deux modèles de gravité par tranche

La section 13 compare le XGBoost Gamma unique actuel à un système comportant :
un modèle XGBoost d’orientation entre coûts <2 000 € / ≥2 000 €, deux experts
XGBoost Gamma, puis une moyenne pondérée par les probabilités d’orientation.
Les probabilités sont prédites chez les sinistrés, à partir des mêmes quatre
variables. La tranche réelle d’un contrat de validation n’est jamais utilisée
pour sélectionner un expert. Il ne s’agit pas d’un modèle de fréquence.

Le seuil et les paramètres sont fixés ; aucune nouvelle recherche n’est faite.
Les prétraitements des trois composants sont ajustés sur leurs seuls sous-trains.
La comparaison utilise exactement les mêmes trois plis par client et les mêmes
2 356 coûts positifs du développement ; le RMSE actuel a été reproduit à l’identique.

- XGBoost unique actuel : RMSE **1 448,34 €**, MAE **924,34 €**.
- Deux experts avec orientation probabiliste : RMSE **1 450,00 €**, MAE **927,12 €**.
- Gain actuel − nouveau : **−1,66 €**, intervalle descriptif bootstrap par client
  à 95 % **[−4,49 ; +1,20] €**. Aucun gain n’est démontré par cet essai.
- L’orientation est peu discriminante (AUC **0,586**), malgré une amélioration
  du Brier et de la perte logarithmique face à une proportion constante.

**Décision : conserver le modèle à quatre variables du §11.** La comparaison ne
prouve pas que toute segmentation serait inutile, mais cette recette n’apporte
pas d’amélioration et ajoute trois composants à ajuster. Le fichier du modèle
actuel et ses prédictions n’ont pas été remplacés. Les données ayant déjà été
explorées, ces résultats restent descriptifs et le holdout n’est pas réévalué.

Le candidat est sauvegardé séparément dans `models/gravite_deux_tranches.joblib`.
Les rapports de `reports/modele_tranches/` comprennent la comparaison globale,
les scores par pli, les effectifs, les prédictions hors pli des deux modèles,
les diagnostics d’orientation et par tranche, le bootstrap, une figure et le
manifeste. Les 50 000 coûts conditionnels de `gravite_test.csv` ont été vérifiés
après rechargement. Les contrôles vérifient également l’indépendance aux cibles
fournies en prédiction et à la population communale exclue.

Réexécution ciblée : `python run_notebook.py --section 13`.
Contrôles : `python -m pytest -q -p no:cacheprovider tests/test_gravite_tranches.py`.

## Essai du modèle à deux tranches avec découpage 90/10

La section 14 repart des 2 917 sinistres strictement positifs de tout `train.csv`.
Un seul `GroupShuffleSplit(test_size=0.10, random_state=2026)` sépare les clients :
2 625 sinistres de 2 606 clients pour l’apprentissage (89,99 % des lignes), et
292 sinistres de 290 clients pour la validation (10,01 %). Aucun client commun.
Les petits/grands sinistres comptent 1 888/737 dans le train et 222/70 en validation.

Les quatre variables, le seuil à 2 000 € et tous les réglages du §13 sont inchangés.
Le XGBoost unique est réentraîné sur les mêmes 90 % pour fournir une comparaison
sur le même jeu de validation, sans réutiliser les anciens modèles ajustés sur tout.

- Deux experts : RMSE apprentissage **1 466,89 €**, validation **1 226,01 €** ;
  MAE validation **868,66 €**.
- XGBoost unique : RMSE apprentissage **1 464,60 €**, validation **1 227,98 €** ;
  MAE validation **870,86 €**.
- Sur ce découpage, gain de RMSE des deux experts : **1,97 €**, faible.

Ce score sur 292 observations d’un découpage nouveau n’est pas directement
comparable aux RMSE de validation croisée précédents et ne démontre pas à lui seul
un gain stable. Le redécoupage réutilise des données déjà explorées, y compris
certaines anciennes lignes réservées ; il n’est pas une validation externe neuve.
Aucun choix de graine, seuil ou hyperparamètre n’a été effectué selon ce résultat.

Les modèles restent ajustés sur les 90 % seulement, et sont enregistrés séparément :
`models/gravite_deux_experts_train90.joblib` et `models/gravite_unique_train90.joblib`.
Les anciens modèles sont conservés. Les rapports de `reports/tranches_90_10/`
contiennent les rôles des contrats et clients, les effectifs, les scores train et
validation, les prédictions de validation, le manifeste et les 50 000 prédictions
conditionnelles `gravite_test_train90.csv` de cette version à deux experts.

Réexécution ciblée : `python run_notebook.py --section 14`.

## Essai à trois tranches sur le même découpage 90/10

La section 15 compare trois experts Gamma (<1 500 €, 1 500–<3 000 €, ≥3 000 €)
aux versions unique et à deux experts. Un modèle XGBoost multiclasses prédit les
trois probabilités et les coûts des experts sont pondérés par ces probabilités,
sans utiliser les montants réels de validation. Les quatre variables et les
réglages des experts sont conservés. L’orientation multiclasses produit trois
arbres par tour ; son budget total d’arbres est supérieur à celui de l’orientation
binaire à nombre de tours identique.

Les rôles du §14 sont repris à l’identique, sans nouveau tirage : 2 625 sinistres
de train et les mêmes 292 de validation, sans client commun. Les trois groupes
comptent respectivement 1 617 / 683 / 325 sinistres dans le train et 195 / 61 / 36
dans la validation. Aucun seuil ou hyperparamètre n’a été sélectionné sur ce score.

- RMSE validation du modèle unique : **1 227,98 €**.
- RMSE validation à deux tranches : **1 226,01 €**.
- RMSE validation à trois tranches : **1 226,37 €**.

Le passage de deux à trois tranches ajoute **0,36 €** au RMSE, un écart négligeable.
La MAE s’améliore légèrement, de **868,66 €** à **864,24 €**. Pour le critère principal
RMSE, cette recette ne justifie pas l’ajout d’un troisième expert. Cette validation
ayant déjà été consultée, la comparaison reste exploratoire et ne démontre pas
une supériorité générale entre les architectures.

Le candidat, ajusté seulement sur les 90 %, est enregistré séparément dans
`models/gravite_trois_tranches_train90.joblib`. Les rapports de
`reports/trois_tranches_90_10/` incluent le découpage identique, les effectifs,
les scores train/validation des trois modèles, leurs prédictions sur les mêmes
292 contrats et les diagnostics de l’orientation. Les 50 000 coûts conditionnels
du test sont dans `gravite_test_train90.csv`. Les anciennes versions restent
inchangées. Les contrôles vérifient les probabilités, la combinaison des experts,
le rechargement, les catégories inconnues et l’absence de dépendance aux cibles
présentes en prédiction ou à la taille communale.

Réexécution ciblée : `python run_notebook.py --section 15`.
Contrôles : `python -m pytest -q -p no:cacheprovider tests/test_gravite_trois_tranches.py`.

## Modèle final publié — version unique avec 80/20

La section 16 met en avant la décision finale : un seul XGBoost Gamma à quatre
variables, avec le découpage 80/20 initial par client du portefeuille complet.
Les 39 964 contrats d’apprentissage contiennent 2 356 sinistres positifs ; les
10 036 contrats de validation en contiennent 561. Les clients sont disjoints.
Le modèle exporté est ajusté seulement sur les 2 356 coûts d’apprentissage,
sans réajustement sur la validation et sans nouveau choix de paramètres.

- RMSE validation 20 % : **1 463,81 €** ; MAE : **930,49 €**.
- RMSE hors pli du développement, selon les trois plis déjà définis : **1 448,34 €**.
- Modèle canonique : `models/gravite_final.joblib` et `models/gravite_final.json`.
- Rapports de référence : `reports/final/` (découpage, effectifs, scores, prédictions
  de validation, 50 000 gravités du test et manifeste).

Les expériences à deux ou trois tranches et les scores 90/10 sont historiques ;
ils ne remplacent pas ce modèle final et ne sont pas présentés comme des scores
80/20. Les données ont déjà été explorées : l’évaluation n’est pas externe neuve.
Le modèle unique est retenu pour sa simplicité et l’absence de gain stable des
segmentations. Le référentiel géographique et les transformations sont embarqués.

Depuis la racine du dépôt :

```sh
python gravite_xgboost/predict_final.py --input Data/test.csv --output gravite_predictions.csv
python gravite_xgboost/run_notebook.py --section 16
```

Le modèle canonique joblib est versionné ; les poids des expériences historiques
peuvent être régénérés par le notebook et restent exclus du suivi Git.
