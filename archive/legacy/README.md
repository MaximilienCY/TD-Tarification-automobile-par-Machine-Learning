# Tarification automobile — modèle principal

Le point de lecture est **[modele_principal.ipynb](modele_principal.ipynb)** : données,
variables, code exact, apprentissage, validation, erreurs, interprétation et limites.
L’[analyse détaillée](docs/modele_principal.md) fournit une lecture sans exécuter de code.
Les [schémas de présentation](docs/schemas/README.md) résument le parcours complet,
la fréquence, la gravité et la validation en quatre vues, avec PDF et fichier
Excalidraw modifiable.
La [présentation de l’histoire du modèle](deliverables/presentation_histoire/README.md)
retrace les choix, les essais abandonnés et les retours de décision, avec un guide
oral pour 25 à 30 minutes et des annexes techniques.

## Modèle retenu

Recette **frequency_curves_1365_657**, associée au score public **1365,657** communiqué
par l’utilisateur. Le reçu historique et les preuves sont conservés dans
`reports/main/evidence/` ; ce score n’a pas été vérifié à nouveau sur Kaggle.

- **Fréquence** : logistique L2 C = 0,1, puis correction de logit par courbes d’âge
  du véhicule, couverture/usage et interactions âge × couverture ; alpha = 100.
- **Gravité** : 37,5 % Gamma à lien log, alpha = 0,1 ; 37,5 % de son montant corrigé
  par XGBoost (240 arbres, profondeur 2) ; 25 % de son montant corrigé par Ridge
  sur les variables étendues, alpha = 10.
- **Prime pure = probabilité × gravité conditionnelle**. La soumission conserve
  toutes les probabilités, sans seuil de classification.

**[submission.csv](submission.csv)** est désormais le fichier de ce modèle :
50 000 lignes, colonnes **index,pred**, index 50 000 à 99 999. La copie du CSV évalué
est [deliverables/submission_frequency_curves.csv](deliverables/submission_frequency_curves.csv).
`pred` est une prime ; ce n’est pas la RMSE.

La recherche historique réduit la RMSE locale de **541,971031 à 541,915571 €**,
un gain marginal de **0,055460 €**. La MAE augmente de 190,732318 à 191,090762 €.
Le bootstrap descriptif par client donne [0,004650 ; 0,110786] € pour le gain de
RMSE ; il ne couvre pas toute l’incertitude. Ces données et le leaderboard ont
été explorés auparavant. Le score privé n’est pas connu.

## Installation et reproduction

Python **3.12**, dans un environnement local. Le modèle utilise uniquement le CPU.
`requirements-main.txt` suffit ; les dépendances des anciens benchmarks ne sont
pas nécessaires. Sous Linux/macOS :

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-main.txt
.venv/bin/python run.py info
.venv/bin/python run.py train
.venv/bin/python run.py diagnostic
.venv/bin/python run.py verify
.venv/bin/python -m pytest -q
.venv/bin/python run.py predict --output reports/main/submission_reproduite.csv
```

Sous Windows, utiliser `.venv\Scripts\python.exe`. Sélectionner cet environnement
comme noyau du notebook. Si le volume du projet ne supporte pas les permissions
requises pour un environnement Python, créer cet environnement sur le disque
local puis lancer les mêmes commandes avec son interpréteur.

- `train` réapprend **la recette figée**, sans sélection de nouveaux modèles.
- `diagnostic` réapprend cette recette dans trois plis séparés par client et
  mesure la calibration, les erreurs et les sensibilités aux variables. Il ne
  relance pas la sélection imbriquée historique et n’est pas un test neuf.
- `verify` contrôle le CSV évalué, les composantes, les entrées interdites,
  les catégories inconnues, les manquants et le notebook historique.
- `predict` écrit des prédictions recalculées, séparées par défaut du CSV soumis.

`run.py` fonctionne aussi depuis un autre dossier en lui donnant son chemin.
Le modèle généré est `models/main_model.joblib` (local, hors Git). Les versions,
empreintes et écarts de reproduction se trouvent dans `reports/final_model.json`.
Le réentraînement sur Linux est contrôlé numériquement sur les 50 000 primes ;
une identité d’octets entre plateformes n’est pas supposée. Une nouvelle sortie
ne reçoit pas automatiquement le score Kaggle de la soumission historique.

## Organisation

```text
modele_principal.ipynb               analyse pédagogique principale, exécutée
submission.csv                      soumission du modèle retenu
run.py                              entrée du modèle principal uniquement
src/pricing/main_model.py           recette figée, entraînement et contrôles
src/pricing/ et scripts/             composants Python partagés
Data/                               train, test et exemple officiel
reports/final_model.json             modèle actuellement réentraîné
reports/main/                       diagnostics, coefficients, figures et preuves
docs/modele_principal.md             analyse détaillée
models/main_model.joblib             modèle généré localement
archive/                            versions et résultats antérieurs
```

Les anciens notebooks, soumissions, rapports, présentation et tests de benchmarks
sont dans [archive/](archive/README.md). Leur contenu a été préservé. Le notebook
historique `Calcul de la prime d'assurance.ipynb` a été déplacé sans changer ses
octets. Son contrôle distingue les fins de ligne Windows/Linux.
Les modules Python partagés restent dans `src/` et `scripts/` parce que le modèle
principal les réutilise. Les commandes de recherche écrivent dans `archive/` et
ne remplacent plus le CSV principal. `archive/run_legacy.py` conserve l’entrée
ancienne ; ses dépendances et preuves sont historiques.

## Limites

2 917 sinistres seulement, une queue de coûts très influente, une recette composite,
une MAE légèrement plus élevée et certaines erreurs par segment : ces limites sont
analysées dans le notebook. Les unités d’exposition et dates ne permettent pas de
présenter ce coût attendu comme un tarif commercial annuel déployable.
Les priorités suivantes sont une cohorte fraîche de validation et une meilleure
documentation des données, plutôt qu’une garantie de meilleur score sur Kaggle.
