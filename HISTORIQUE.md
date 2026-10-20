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
