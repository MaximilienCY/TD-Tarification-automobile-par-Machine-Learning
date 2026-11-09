# Historique du projet

Point au 20 octobre 2026. Le projet a été simplifié : il ne reste que les données, trois notebooks (fréquence, gravité, prime finale) et la soumission. Tout ce qui a été retiré reste récupérable (voir la dernière section).

## Répartition

- **Fréquence** (probabilité d'au moins un sinistre) : Maximilien, y compris pour l'oral.
- **Gravité** (montant attendu si sinistre) : le binôme.

## Métrique et format Kaggle

- Compétition privée CY Tech 2027, fin le 10 novembre 2026 à 23:59 (heure de Paris).
- Métrique : RMSE de la prime sur le test (la consigne parle de MSE ; même classement). Leaderboard public sur environ 33 % du test, privé sur le reste.
- Soumission : `index,pred`, 50 000 lignes, index de 50000 à 99999.

## Scores Kaggle publics

Scores obtenus et transmis par le binôme.

| Fréquence | Gravité | Kaggle public |
|---|---|---|
| logistique de référence | Gamma de référence | 1365.943 |
| logistique + correction XGBoost | Gamma de référence | 1366.010 (écarté) |
| reprise complète XGBoost (fréquence et gravité) | — | 1366.025 (écarté) |
| logistique de référence | Gamma + XGBoost | 1365.913 |
| logistique de référence | Gamma + XGBoost (3 graines) | 1365.914 (écarté) |
| logistique de référence | Gamma + courbes | 1365.745 |
| **logistique + courbes** | **Gamma + courbes** | **1365.657 (meilleur, = `submission.csv` actuel)** |

## Fréquence

La **logistique de référence** est la régression logistique de Maximilien : 19 variables, marque / modèle / département remplacés par un taux de sinistres lissé (m = 100), C = 0,1. Toutes les soumissions sont construites sur elle.

| Étape | Résultat | Décision |
|---|---|---|
| Logistique de référence | log-loss validation 0,2138 contre 0,2225 pour le taux moyen ; Gini 0,313 ; bien calibrée (583 sinistres prédits pour 584 observés) | base |
| Audit de la logistique | modèle sain, au plafond de l'information disponible ; XGBoost, variables en classes, variables métier ne font pas mieux | corrections appliquées (notebook 01, étape 1) |
| Traitement marque / modèle (notebook 01, 4.5) | 13 variantes (lissage, taux hiérarchique, clé marque\|modèle, regroupement des rares…) : aucune ne bat la version actuelle ; sans marque ni modèle, jeu égal | version actuelle gardée |
| Tests de variables (notebook 01, 4.6) | indispensables : type de contrat, âge du véhicule, bonus, âge du conducteur ; plus petit (6 ou 12 variables) = pire ; département, marque, modèle, classes, interactions = rien ; seul gain net : 3 variables métier (jeune 2e conducteur, puissance / poids, malus), −0,00031 de log-loss CV | variables métier à intégrer, puis à juger sur la RMSE de la prime |
| Solidité de la validation croisée (notebook 01, 4.8) | 5 tirages, avec et sans stratification, 3 / 5 / 10 paquets : log-loss CV entre 0,21404 et 0,21443, gain des variables métier entre −0,00027 et −0,00036 dans tous les cas | pas de coquille ; découpage gardé (stratifié, 5 paquets, graine 42) |
| Catégorielles en taux de sinistres au lieu du one-hot (notebook 01, 4.9) | les 7 variables en taux : +0,00024 de log-loss CV (pire), Gini 0,297 → 0,294 ; une à une : égal ou légèrement pire | one-hot gardé |
| Sélection pas à pas des variables et de leur forme (notebook 01, 4.10) | 39 candidates : 6 variables retenues (âge du modèle de véhicule, type de contrat en taux, cylindrée, bonus, âge du conducteur, jeune 2e conducteur) ; log-loss CV 0,21399, Gini CV 0,306 ; validation 0,21369 (meilleure) ; actuel + métier : CV 0,21396 mais validation 0,21380 | 2 candidats à départager sur la prime : 6 variables, et actuel + métier |
| Optimisation des hyperparamètres (notebook 01, 4.11) | C × pénalité (L2, L1) et lissage m sur les 2 candidats : le défaut (L2, C = 0,1) est déjà le meilleur ; L1 jamais meilleure ; m = 1000 un peu mieux pour actuel + métier ; validation : 6 variables 0,21368, actuel + métier réglé 0,21373 | réglages quasi sans effet ; départage sur la prime |
| **Candidats jugés sur la prime** (notebook 01, 4.12, et notebook 03) | « actuel + 3 variables métier, m = 1000 » × Gamma + courbes : RMSE locale **541,8177** contre 541,9266 pour la meilleure combinaison connue ; meilleur dans les 5 plis, dont la validation (verdict « mieux ») ; sélection à 6 variables : 541,9826 (égal) | **`submission_candidat.csv` à soumettre sur Kaggle** |
| **Maximisation de la logistique** (notebook 01, 4.13) | courbes (15 variables), interactions avec formule et usage, formule × usage, réservoir de 4.10 : seul `novice2` passe le test ; C inchangé ; mélange avec le binôme dans le bruit. **Modèle final : 23 variables, 30 colonnes, m = 1000, C = 0,1** ; RMSE prime locale **541,8086** (meilleure), log-loss 0,2137, Gini 0,309, calibration 2 919 / 2 917 | **logistique au plafond** ; `submission_candidat.csv` = modèle final × Gamma + courbes ; suite : XGBoost |
| **Élimination des variables inutiles** (notebook 01, 4.14) | de 23 à 11 variables, la prime ne bouge pas ; à 7, elle se dégrade. **Modèle final : 14 variables, 20 colonnes** (âge du véhicule, puissance, poids, puissance / poids, marque, formule, bonus, malus, fréquence de paiement, usage, âge du conducteur, jeune 2e conducteur, 2e conducteur novice, département) ; RMSE prime locale **541,7917** ; bat le binôme dans les 5 paquets (−0,133) ; calibration 2 922 / 2 917 | remplacé par le modèle E (4.15) ; prédictions dans `frequence_14variables_*` |
| **Choix des variables par logique métier** (notebook 01, 4.15) | 5 modèles construits comme un actuaire (une variable par facteur de risque). **E** = modèle de 4.13 sans les 6 variables « non causales » (fréquence et mode de paiement, ancienneté client, durée du contrat, marque, modèle) : **17 variables, 22 colonnes** ; log-loss 0,21357 (meilleure), Gini 0,311, RMSE prime locale **541,7941** ; calibration 2 921 / 2 917. Écarts avec 14 et 23 variables dans le bruit | **modèle final = E** ; **`submission_candidat.csv` = E × Gamma + courbes** (**Kaggle 1365.915**, moins bon que 1365.657 au public ; écart dans le bruit d'après une simulation sur 33 % des contrats). Notebook 01 à réexécuter (section 4.15 écrite mais pas encore exécutée) |
| **XGBoost** (notebook 01b) | 35 colonnes (brutes + 5 calculées, catégories natives, taux lissés hors pli pour marque / modèle / département / commune) ; 60 réglages × nombre d'arbres choisis sur les paquets 2 à 5. Seul : jeu égal avec E en log-loss. 20 modèles (4 réglages × 5 graines) + recalibration de Platt : log-loss 0,21349, RMSE prime 541,637 (mieux que E sur la prime). Contraintes métier, colonnes réduites, correction de E par XGBoost : écartés. **Mélange 0,5 XGBoost recalibré + 0,5 E : log-loss 0,21327, Gini 0,316, RMSE prime 541,614**, mieux que E (log-loss et prime) et que le binôme (−0,313, 5 paquets sur 5) | **meilleur modèle de fréquence en local** ; `submission_melange.csv` (pas encore soumis) |
| Mélange 75 % CatBoost + 25 % logistique (binôme) | légèrement meilleur en local, jamais noté sur Kaggle | abandonné |
| Correction XGBoost du logit (binôme) | Kaggle 1366.010, moins bon | écarté |
| Autres réglages de la logistique : C, pondérations, perte Brier (binôme) | pas de gain local | écarté |
| **Logistique + courbes** (binôme) | la probabilité de la logistique est corrigée sur l'échelle du logit par une courbe de l'âge du véhicule, la couverture / l'usage et leur interaction (correction logistique pénalisée) ; Kaggle 1365.657 | **fréquence actuelle à battre** |

## Gravité (binôme)

| Gravité | Principe | Décision |
|---|---|---|
| Gamma de référence | régression Gamma (lien log, alpha 0,1) sur âge du véhicule, type de contrat, usage | gardée (réécrite dans le notebook 02) |
| Gamma + XGBoost | 50 % Gamma, 50 % Gamma corrigée par un XGBoost très régularisé (profondeur 2, 240 arbres) | gardée (prédictions figées) |
| Gamma + courbes | la précédente + 25 % d'une correction Ridge avec courbes et catégories (véhicule, conducteur, contrat) | gardée (prédictions figées), utilisée dans la meilleure soumission |
| Autres essais : CatBoost, LightGBM, Gamma décalée ou pondérée, Poisson, interactions couverture, courbes seules, trois graines XGBoost | pas de gain confirmé | écartés |

## Évaluation locale (notebook 03)

Prédictions hors pli sur le train (5 plis par client, `StratifiedGroupKFold`, graine 42 ; le pli 1 est le jeu de validation du notebook 01). L'ordre des RMSE locales reproduit l'ordre des scores Kaggle : 542,0441 (1365.943), 542,0282 (1365.913), 541,9951 (1365.745), 541,9266 (1365.657). Prime constante : 545,3974.

## Où retrouver ce qui a été retiré

- **Code complet du binôme** (`src/`, `scripts/`, `tests/`, `reports/`, `docs/`, notebooks 01 à 07) : branche `main`, commit `602f280`. Pour voir un fichier : `git show 602f280:docs/results.md`.
- **Étapes de Maximilien sur la fréquence** (corrections de l'audit, tests marque / modèle) : branche `amelioration-logistique`.
- **Présentation PowerPoint et 11 soumissions candidates** : dossier `deliverables (retiré du projet)/`, à côté du dossier projet. Également dans le commit `602f280`.
- **Prédictions des gravités du binôme** : `Data/predictions/binome_composantes_*.csv`, extraites le 20 octobre 2026 avec son code. Elles reproduisent ses soumissions Kaggle à moins de 1e-8 € près.
