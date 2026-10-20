# Compétition CY Tech 2027

Vérification dans le compte Kaggle connecté le 10 octobre 2026.

- [Compétition](https://www.kaggle.com/competitions/cy-tech-2027-calcul-de-la-prime-dassurance/overview)
- [Données](https://www.kaggle.com/competitions/cy-tech-2027-calcul-de-la-prime-dassurance/data)
- [Règles](https://www.kaggle.com/competitions/cy-tech-2027-calcul-de-la-prime-dassurance/rules)
- Compétition privée ; règles déjà acceptées dans le compte.
- Maximum : trois membres par groupe, 50 soumissions par jour.
- Leaderboard public calculé sur environ 33 % du test ; privé sur les 67 % restants.
- Fin affichée : 10 novembre 2026 à 23:59:59, heure de Paris.
- Avant ce travail : meilleur score public de l'équipe **1365.943**, dernière soumission **1366.006**, rang public **1**.

La consigne décrit la MSE mais écrit la formule de sa racine (RMSE). Nous calculons et exportons les deux. Leur classement est identique sur un jeu d'évaluation donné. Le score retourné par Kaggle est enregistré tel quel, sans le confondre avec une mesure locale.

Le fichier officiel `prime_pred_sandbox.csv` contient bien **index,pred**, contrairement au texte descriptif qui mentionne `id`. Les index vont de 50000 à 99999. Le fichier téléchargé sert de référence pour le format.

## Vérification des fichiers officiels

Le ZIP téléchargé depuis Kaggle contient `train.csv`, `test.csv` et `prime_pred_sandbox.csv`. Les deux jeux de données sont identiques octet par octet aux fichiers du dépôt :

| Fichier | SHA-256 |
| --- | --- |
| train.csv | 8b5ca8a59c214269056ed1720de33619c10968549c383ec350898a659b97bfc5 |
| test.csv | 68cdf87201974688d754d846c2d8ff33c5aa2f9c011e84fba92018186cd8f569 |
| prime_pred_sandbox.csv | 7ace601eadbe3ceb8846d09ba149703eb14674f97b8d1c68abf3418e9d72e4bd |

## Périmètre

Deux modèles distincts sont conservés : probabilité de sinistre et coût conditionnel positif. Leur produit donne la prime pure. Les identifiants, les cibles et le sexe des conducteurs sont exclus des variables prédictives. Aucun coût caché du test n'est utilisé. Les identifiants clients servent uniquement à la séparation des plis et au bootstrap.

Les durées d'observation et la date de mesure des variables ne sont pas documentées précisément. La prédiction porte sur le coût de la période du jeu de données ; elle ne prétend pas être un tarif commercial annuel prêt à déployer.
