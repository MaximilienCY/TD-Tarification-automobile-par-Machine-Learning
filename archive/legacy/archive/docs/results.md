# Résultats et limites

Entraînement réalisé le 10 octobre 2026. Vérification de livraison le 11 octobre 2026.

## Modèle final sélectionné

- **Fréquence : 75 % CatBoost + 25 % régression logistique de référence.** CatBoost utilise 700 arbres de profondeur 6, un taux d'apprentissage de 0,04 et une régularisation L2 de 30.
- **Gravité : Gamma à lien log, alpha = 0,01.** Les variables sont ancienneté du véhicule, type de contrat et utilisation.
- **Prime : produit des deux composantes.** Aucun facteur de recalage n'est ajusté à partir du leaderboard.

Le modèle final apprend sur les 50 000 contrats pour la fréquence et sur les 2 917 sinistres pour la gravité. Les modèles plus complexes de gravité ont été comparés mais n'ont pas gagné la sélection de prime.

## Comparaison locale

La recherche compare 12 candidats de fréquence, 14 de gravité et des mélanges convexes prédéfinis : **1 215 combinaisons**. Les cinq plis de développement sont partagés entre les composantes et séparés par client. RMSE de recherche du gagnant : **550,065781**, contre **550,207193** dans l'ancienne recherche. Ces scores sont optimistes après sélection et les catalogues diffèrent.

La validation imbriquée répète le catalogue entier sur trois plis internes à chacun des trois plis externes. Elle compare les procédures sur les mêmes 40 013 contrats :

| Procédure | MSE | RMSE |
| --- | ---: | ---: |
| Nouvelle sélection imbriquée | 302 890,789 | **550,355148** |
| Logistique × Gamma de référence | 302 925,011 | 550,386238 |
| Ancien couple optimisé, configuration fixe | 302 957,274 | 550,415546 |
| Prime constante | 306 411,975 | 553,544917 |

Le gain face à la référence vaut **0,031091 RMSE**. L'intervalle descriptif à 95 % par bootstrap des clients est **[-0,117429 ; 0,190997]**. Il contient zéro : la supériorité n'est pas statistiquement établie. Le nouveau processus gagne sur un pli et perd légèrement sur deux. Les choix varient entre plis, signe de l'instabilité des petits écarts.

Attention : la ligne « ancien couple optimisé » ci-dessus évalue **la configuration finale fixe** de l'ancien projet. Elle n'est pas la même procédure que l'ancienne sélection imbriquée archivée, qui obtenait 550,380829.

## Diagnostic historique

Le jeu historique de 9 987 contrats a déjà été examiné dans le projet original. Il n'a pas servi à choisir les paramètres ou mélanges dans cette exécution. Il reste un diagnostic, pas un nouveau test indépendant.

| Modèle | RMSE historique |
| --- | ---: |
| Nouveau modèle | **507,318911** |
| Ancienne optimisation | 507,598671 |
| Référence | 507,647017 |

Fréquence du nouveau modèle : AUC **0,674502**, Brier **0,052272**, log-loss **0,207353**. Le ratio coût prédit / observé vaut **1,02637**. Les notebooks présentent la calibration par déciles.

## Kaggle

Les données officielles téléchargées correspondent exactement à celles du dépôt. L'exemple officiel confirme `index,pred`, 50 000 lignes et index de 50000 à 99999.

Le score public existant est **1365.943**, obtenu par le modèle de référence. Les prédictions de cette soumission ont été téléchargées et sont numériquement identiques à l'ancien export `output/prime/submission.csv`. La dernière ancienne soumission (v4) affiche **1366.006**.

**Le nouveau fichier n'a pas encore de score Kaggle confirmé.** L'upload dans le navigateur intégré a échoué : d'abord un jeton de session manquant, puis une erreur réseau après rechargement. Le bouton Submit est resté désactivé. Le fichier `submission.csv` est prêt pour un upload dans un navigateur fonctionnel. L'historique vérifié est enregistré dans `reports/kaggle_submissions.json`.

Le score Kaggle public n'est pas interchangeable avec la RMSE locale : les populations sont différentes et la consigne emploie à la fois « MSE » et la formule de la RMSE. Les scores Kaggle sont enregistrés tels qu'affichés. Le public ne représente que 33 % du test, le privé les 67 % restants. Nous ne prétendons donc pas avoir trouvé un optimum universel ou garanti un meilleur classement final.

## Reproductibilité et contrôles

- Données, versions et code identifiés par empreintes dans `reports/selection.json` et `reports/final_model.json`.
- Modèle local : `models/premium.joblib`, rechargé et contrôlé après export.
- Prédictions détaillées : `reports/test_components.csv`, avec probabilité, montant et produit.
- Contrôle de l'ordre des index et du schéma par rapport au fichier officiel.
- Prédictions finies, non négatives ; probabilités dans [0,1], coûts conditionnels positifs.
- Huit tests fonctionnels passent : partition par client, exclusion des cibles/identifiants/sexes, catégories inconnues, manquants, sérialisation, export et notebook protégé.
- Un avertissement Pandas concernant une future version majeure apparaît lors du test des catégories inconnues LightGBM. Le comportement est valide dans l'environnement Pandas 3.0.6 figé.

Le notebook protégé conserve son SHA-256 original : `76248bdb51400fe399c670a9b71092cabe041cc13d3839b83846ac1499938429`.

## Suite utile

Soumettre le fichier final et enregistrer le résultat exact avec son empreinte. Conserver la référence comme candidat de comparaison, car le gain local reste incertain. Pour une amélioration robuste, la priorité serait une nouvelle cohorte de validation et une meilleure description des durées d'exposition et de la date des variables. Les grands sinistres expliquent une part importante de l'erreur, ce qui limite la portée de différences très faibles entre modèles.
