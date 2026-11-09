# Analyse du modèle principal — fréquence corrigée et gravité composite

## 1. Quel modèle est réellement retenu ?

Le modèle principal est la recette `frequency_curves_1365_657`, associée au score
public **1365,657** dans le reçu du 12 octobre 2026. Ce score a été communiqué par
l'utilisateur ; il n'a pas été vérifié à nouveau sur Kaggle. Son CSV est désormais
`submission.csv` et sa copie immuable `deliverables/submission_frequency_curves.csv`.
Le notebook de référence est `modele_principal.ipynb`. Les anciens notebooks et
résultats se trouvent dans `archive/`.

Ce choix corrige une incohérence du dépôt reçu : `run.py`, `reports/final_model.json`
et le CSV racine décrivaient encore une ancienne combinaison CatBoost/Gamma, alors
que le README recommandait déjà le dernier modèle. L'entrée unique est maintenant
alignée sur la recette ci-dessous. Aucune nouvelle recherche de paramètres n'a été
effectuée pour obtenir un meilleur résultat apparent.

## 2. Problème, cible et données

Pour les caractéristiques X d'un contrat, nous estimons :

**prime(X) = P(sinistre | X) × E[montant | sinistre, X].**

Cette identité vise le coût moyen de la période observée. Elle ne définit pas un
tarif commercial incluant frais, marge ou taxes. L'unité de durée et l'exposition
ne sont pas suffisamment documentées pour annualiser le montant.

Le train contient 50 000 contrats, 45 785 clients et 2 917 sinistres, soit **5,834 %**.
Le test contient 50 000 prospects, sans les cibles. Les coûts sont exactement nuls
en l'absence de sinistre. Les coûts positifs ont une moyenne de 1 771,84 €, une
médiane de 1 236 €, un minimum de 600,60 € et un maximum de 21 826,96 €.

La fréquence apprend sur tous les contrats. Les modèles de gravité apprennent
uniquement sur les sinistres positifs. Au moment de prédire, on estime une gravité
**conditionnelle** pour chaque prospect : on ne sait pas encore lesquels auront
un accident. Aucune classification oui/non ne met les primes à zéro.

Les identifiants de client servent aux plis, jamais comme prédicteurs. Les autres
identifiants, les deux sexes et les cibles sont exclus des entrées. Le test ne sert
ni à apprendre les imputations ni à sélectionner les paramètres. L'index du CSV
reste une clé de ligne, de 50 000 à 99 999.

## 3. Fréquence : logistique, puis correction de logit

La base estime `p0 = sigmoid(intercept + coefficients × variables préparées)`.
C'est une régression logistique L2, `C = 0,1`. Les poids de classes ne sont pas
rééquilibrés : la sortie doit représenter une probabilité de sinistre.

La préparation conserve la recette historique :

- Poids et cylindrées nuls traités comme mesures manquantes.
- Imputation de poids, cylindrée et ancienneté du véhicule par médiane du modèle,
  puis de la marque, puis globale, apprises dans l'apprentissage autorisé.
- `log1p` sur bonus, puissance, poids et cylindrée ; mise à l'échelle des entrées
  numériques non binaires.
- Encodage indicateur des petites catégories et regroupement des catégories rares
  au-dessous de 50 contrats d'apprentissage.
- Encodage cible lissé de la marque, du modèle et du département. Le lissage vaut
  100. Les valeurs d'apprentissage sont produites par cinq plis séparés par client ;
  leur prior exclut également les clients encodés.

Le dernier modèle ajoute :

**p = sigmoid(logit(p0) + delta(X)).**

`delta` est une correction linéaire sur des **bases de splines quadratiques** de
l'âge du véhicule, des indicateurs de contrat et d'usage, et des interactions
entre les courbes d'âge et le contrat. Quatre nœuds sont utilisés. Pour cette
correction, les nœuds sont uniformes par défaut ; ce ne sont pas les nœuds par
quantiles de certaines anciennes variantes Gamma. Les interactions sont multipliées
par 0,5 avant estimation.

La correction minimise la somme des log-loss binomiales avec pénalisation L2
`alpha = 100`. Sa pénalité porte sur tous ses coefficients, y compris son
intercept. Son poids final est 1. Cela signifie qu'on ajoute toute la correction
au logit ; cela ne remplace pas la probabilité par 100 % de sinistre.

Pour apprendre cette correction, `p0` est calculée hors client dans trois sous-plis.
La base est ensuite réapprise sur tout l'apprentissage autorisé pour prédire les
clients exclus. Cette précaution évite d'entraîner la correction sur des
probabilités artificiellement bonnes obtenues sur les mêmes clients.

## 4. Gravité : Gamma conservé comme ancre, avec deux corrections

Le dernier modèle n'est plus un Gamma pur. Sa formule exacte est :

**s = 0,375 × sGamma + 0,375 × sXGBoost + 0,25 × sRidge.**

Ces poids viennent de deux mélanges successifs : 50/50 Gamma–XGBoost, puis 75 % de
ce mélange et 25 % de Ridge. Il ne s'agit pas de trois équipes prédisant une
fréquence : les trois termes représentent des montants conditionnels.

### Gamma

`sGamma = exp(intercept + coefficients × entrées préparées)`.
Le Gamma à lien log, `alpha = 0,1`, utilise le type de contrat, l'ancienneté du
véhicule et l'utilisation. Il optimise sa déviance Gamma régularisée, pas
directement la RMSE. L'âge est imputé et standardisé ; les catégories sont
encodées avec regroupement des modalités rares. L'exponentielle garantit la
positivité.

### Correction XGBoost

Elle utilise les mêmes trois variables métier. Le modèle apprend le ratio
`coût / Gamma_hors_client`, puis multiplie la gravité Gamma par ce ratio prédit.
La sortie monétaire est bornée inférieurement à `1e-8`, pour éviter un montant
négatif ou nul. Ce plancher est une protection numérique, pas une franchise.

Configuration : **240 arbres, profondeur 2, learning_rate 0,025,
min_child_weight 100, L2 100, L1 1, seed 20261012**, histogrammes et catégories
natives. La valeur initiale du ratio est 1. Le modèle principal ne contient pas
les trois XGBoost de 480 arbres d'un successeur historique : ce successeur a été
écarté.

### Correction Ridge

Elle apprend `coût / Gamma_hors_client − 1`, sur le panneau complet de variables.
Elle utilise une régression Ridge, `alpha = 10`, des splines quadratiques à quatre
nœuds, des indicateurs d'anomalies et des catégories regroupées au-dessous de
15 sinistres d'apprentissage. La sortie est
`sRidge = sGamma × max(1 + correction_Ridge, 1e-8)`.

Son panneau ajoute notamment prix, puissance, poids, cylindrée, ratio
puissance/poids, vitesse, marque/modèle, énergie/type, caractéristiques des
conducteurs et permis, bonus, paiement, durées et département. L'absence du second
conducteur est explicite ; ses âges/permis structurellement nuls deviennent
manquants pour ces courbes. Les incohérences de permis sont signalées, sans
prétendre connaître la correction réelle. La variable de fin de vente n'entre
dans aucune des composantes retenues.

Les cibles auxiliaires et pondérations des corrections sont produites hors
client. Pour les seules lignes sinistrées, le poids est proportionnel à
`p_auxiliaire × sGamma_auxiliaire²`, puis normalisé à moyenne 1.
La probabilité auxiliaire de gravité provient d'une logistique compacte ; elle
est distincte de la fréquence finale corrigée.

Pourquoi ces poids ? Pour un ratio r, la quantité
`p × s² × (y/s − r)²` égale `p × (y − s×r)²`.
La densité des sinistrés étant déjà proportionnelle à leur probabilité de
sinistre, cette pondération rapproche la correction de la partie de l'erreur de
prime dépendant de la gravité. Ce raisonnement n'assure pas un gain en pratique :
les probabilités auxiliaires sont estimées et les modèles restent contraints.

## 5. Quelles variables sont vraiment conservées ?

Il faut distinguer les composants. Le **Gamma seul** utilise trois variables.
Le **modèle principal complet** en utilise davantage, surtout à travers la base
logistique et la correction Ridge. Le département est utilisé dans l'encodage
cible de fréquence et dans la Ridge ; il n'est pas une variable du Gamma.
La marque et le modèle automobile suivent la même distinction.

Le notebook produit un inventaire explicite par variable et par composante, les
coefficients ajustés et une importance par permutation sur des clients exclus.
Les corrélations et les coefficients ne prouvent pas la causalité. L'importance
peut être répartie entre des variables corrélées ; permuter des groupes crée
parfois des profils peu réalistes. Ces mesures servent à comprendre la recette,
pas à refaire une sélection sur la validation.

L'extraction historique de département de la base logistique conserve les deux
premiers caractères. Le traitement métier plus récent sait distinguer les DOM.
L'extension à de nouvelles zones demande donc une harmonisation avant un usage
réel hors du périmètre des données actuelles.

## 6. Apprentissage, validation et test

La recherche historique de fréquence comparait neuf configurations et les poids
0,25/0,5/1, avec une option inchangée : **28 choix**. Elle choisissait sur cinq
plis internes, puis évaluait trois plis externes séparés par client. L'ancre de
gravité était maintenue fixe. Tous les plis externes ont retenu la même correction
`age_coverage`, alpha 100, poids 1.

Le diagnostic exécuté pour cette analyse **réapprend la recette fixe** dans les
mêmes trois plis externes. Il contrôle la reproduction, mais ne relance pas les
28 choix internes et ne doit pas être présenté comme une nouvelle sélection
imbriquée indépendante. Il ajoute des calibrations, des erreurs par groupe et
des perturbations de variables sur les clients exclus de chaque apprentissage.

Après vérification, le modèle final apprend sa fréquence sur 50 000 contrats et
ses gravités sur 2 917 sinistres. Le test est ensuite prédit une seule fois avec
ces paramètres figés. Il n'a pas de cibles connues : aucune RMSE de test ne peut
être calculée localement.

## 7. Résultats : les gains et les contreparties

Dans l'évaluation historique de cette procédure, la RMSE de prime passe de
**541,971031 à 541,915571 €**, un gain de **0,055460 €**, soit environ **0,0102 %**.
La MSE passe de 293 732,599 à 293 672,486 €². La log-loss de fréquence passe de
0,214044321 à 0,213930167 et le Brier de 0,054085306 à 0,054071441.

En revanche, la **MAE se dégrade**, de 190,732318 à 191,090762 €, soit environ
0,3584 €. La sélection optimise la MSE, pas toutes les métriques simultanément.
Le biais moyen passe de −0,3664 à +0,1239 € et le ratio du total prédit/observé
de 0,996455 à 1,001199.

Les trois plis externes gagnent en RMSE, mais tous les segments n'en bénéficient
pas : le segment Median1 est très légèrement moins bon. Le bootstrap apparié par
client, 2 000 tirages, donne un intervalle descriptif du gain
**[0,004650 ; 0,110786] €**. Il est conditionnel aux prédictions ajustées et ne
couvre pas toute l'incertitude d'apprentissage ou de recherche humaine.

Le score public communiqué passe de **1365,745 à 1365,657**, un gain de 0,088.
Il concerne d'autres observations ; il ne doit pas être confondu avec la RMSE
locale. Les données, variables et leaderboard ont déjà été consultés plusieurs
fois : ces diagnostics ne constituent pas un nouveau test indépendant. Le score
privé n'est pas connu. Le seuil historique 1365,500 n'est pas atteint.

## 8. Forces, faiblesses et améliorations utiles

**Forces.** Décomposition métier cohérente ; plis par client ; encodages et cibles
auxiliaires hors client ; régularisation et mélanges qui limitent les corrections ;
coûts extrêmes conservés ; sorties positives ; format et ordre des index contrôlés ;
recette, fichier évalué et code reliés explicitement.

**Faiblesses.** Gain faible ; seulement 2 917 sinistres pour de nombreuses catégories ;
grands coûts très influents ; modèle composite moins simple qu'un GLM ; MAE plus
élevée ; certains segments légèrement dégradés ; absence de cohorte fraîche ou
temporelle ; nombreuses décisions informées par les données déjà explorées.
Le modèle fournit une moyenne conditionnelle, pas un intervalle fiable pour le
coût de chaque client. Les corrélations de parc automobile peuvent répartir ou
masquer l'importance des variables.

L'écart entre score public et RMSE locale ne permet pas, à lui seul, d'identifier
une cause. Sans coûts du test et définition complète de son échantillonnage,
attribuer cet écart à une anomalie particulière serait spéculatif.

**Priorités.** Obtenir un dictionnaire des unités, dates et expositions ; réserver
une nouvelle cohorte pour une validation finale ; enrichir l'historique de
sinistres ; suivre calibration et erreurs par segment ; vérifier les dérives de
catégories et de valeurs manquantes. La structure actuelle peut servir à
l'exercice ; elle ne suffit pas à déclarer un tarif commercial déployable.

## 9. Reproduction et provenance

`run.py` pilote uniquement le modèle principal : `info`, `train`, `diagnostic`,
`predict`, `verify`. Les anciennes recherches restent dans
`archive/run_legacy.py` et les modules partagés dans `scripts/`/`src/`.

`reports/final_model.json` décrit le réentraînement courant et
`reports/main/verification.json` ses contrôles. Les preuves historiques restent
dans `reports/main/evidence/`. Le score communiqué reste rattaché au CSV soumis.
Les fichiers texte récupérés sous Linux peuvent avoir des LF au lieu de CRLF :
le contrôle vérifie ce changement de fins de ligne explicitement, sans prétendre
que leurs empreintes brutes sont identiques.

Le modèle local est `models/main_model.joblib`. Le réentraînement utilise les
mêmes paramètres sur CPU et compare toutes ses primes au fichier évalué. Les
différences numériques et les versions réelles sont enregistrées, y compris si
elles proviennent de la plateforme. Une prédiction recalculée n'obtient pas
automatiquement le score Kaggle du fichier de référence.

Références techniques primaires :
[GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html),
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html),
[paramètres XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).


## 10. Diagnostic réexécuté lors de cette analyse

La recette fixe reproduit la RMSE **541,915570844 €**. L’AUC de fréquence vaut
**0,653276**, contre 0,652240 avant correction : le pouvoir de classement reste
modeste. Sur les seuls sinistrés, la RMSE de gravité vaut **1 452,479870 €** et
la MAE **930,882083 €**. Ces mesures n’ont pas les mêmes cibles ni les mêmes unités
de lecture qu’une probabilité ou qu’une erreur de prime sur tous les contrats.

Les groupes sont permutés sur les clients exclus de chaque apprentissage, trois
fois par pli. Les hausses moyennes de MSE de prime sont d’environ **2 077 €² pour
l’âge du véhicule**, **2 011 €² pour le contrat et l’usage**, **920 €² pour les
caractéristiques techniques/prix**, **697 €² pour conducteur principal/bonus**,
**208 €² pour le département**, **124 €² pour marque/modèle/énergie/type**, et
**24 €² pour le second conducteur**. La fin de vente, contrôle inutilisé, donne
exactement zéro. L’écart-type de ces perturbations ne mesure pas toute l’incertitude ;
les colonnes corrélées et les profils artificiels limitent leur interprétation.

Les **30 plus gros sinistres**, soit 1 % des sinistrés, expliquent environ
**23,7 % de l’erreur quadratique**. Ils restent intégralement dans les apprentissages.

Les 35 tests du périmètre actif passent. Les contrôles propres aux anciens
benchmarks ont été conservés dans `archive/tests/`, avec leurs dépendances
optionnelles ; ils ne sont pas présentés comme des tests du nouveau point d’entrée.
Le modèle rechargé reproduit les 50 000 primes du CSV évalué avec un écart maximal
inférieur à **2,3 × 10⁻⁹ €**. Le CSV évalué reste conservé octet pour octet dans
la représentation locale LF ; les octets du reçu Windows CRLF sont vérifiés après
normalisation explicite.
