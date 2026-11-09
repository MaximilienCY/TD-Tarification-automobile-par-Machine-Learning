# Modèle global de tarification automobile — version principale

**Prime pure = probabilité de sinistre × coût conditionnel du sinistre.**

Cette version fusionne la gravité de `main` et la fréquence de `projet-simplifie`.
Le notebook principal est [modele_global.ipynb](modele_global.ipynb) ; `run.py`
est le point d’entrée pour entraîner et prédire. Les versions précédentes sont
dans [archive/](archive/README.md).

## Composants retenus

- **Gravité :** le XGBoost Gamma publié de `main`, conservé sans réentraînement.
  Variables : `type_contrat`, `prix_vehicule`, `anciennete_vehicule`,
  `classe_geographique`. Les sources, le référentiel et l’artefact original
  sont dans `components/severity/`. La proposition exploratoire à trois variables
  archivée n’est pas substituée à ce modèle publié.
- **Fréquence :** recette finale de `projet-simplifie` (`682b3c5`) : mélange
  **50 % logistique E à 17 variables métier + 50 % XGBoost recalibré**.
  XGBoost moyenne quatre réglages et cinq graines, soit 20 modèles.
  L’implémentation réutilisable est dans `src/pricing_model/frequency.py`.
- **Modèle global :** produit des deux prédictions, sur les mêmes contrats et
  dans l’ordre du fichier d’entrée. Aucune cible n’est utilisée à la prédiction.

Les matrices de préparation de la fréquence ont été comparées aux deux notebooks
sources sur 5 000 contrats d’apprentissage et 200 de test : égalité numérique à
`1e-10`, dont les 35 colonnes XGBoost. Les quatre
réglages, les graines et le poids du mélange sont conservés. Le runtime commun
utilise XGBoost CPU 3.0.2, compatible avec l’artefact de gravité ; la branche de
fréquence déclarait 3.4.1. Les probabilités de cette intégration sont régénérées,
avec le découpage commun ci-dessous, et ne sont pas présentées comme identiques
aux anciens exports de cette branche.

## Découpage commun

`GroupShuffleSplit(test_size=0.20, random_state=2026)` sur `id_client` :

- Apprentissage : **39 964 contrats**, dont **2 356 sinistres positifs**.
- Validation : **10 036 contrats**, dont **561 sinistres positifs**.
- Aucun client commun.

La fréquence est ajustée sur les seuls 80 %. Les taux cibles sont calculés hors
pli par client. La calibration de Platt utilise les probabilités hors pli du
développement, en excluant le premier pli comme dans la recette source. Le
mélange reste fixé à 50/50. La gravité publiée reste inchangée, déjà ajustée sur
les sinistrés de ces mêmes 80 %. Aucun composant n’est réentraîné sur les 20 %.

**RMSE de prime sur les 10 036 contrats de validation : 535.21 €**,
contre **539.12 €** pour la prime constante apprise sur les 80 %.
Les mesures détaillées sont dans [reports/validation.json](reports/validation.json).
Les anciens scores Kaggle et les scores des autres couples fréquence / gravité
ne sont pas attribués à cette version. Les données ayant déjà été explorées,
ce découpage ne constitue pas une validation externe neuve.

## Utilisation

Avec Python 3.12 et les dépendances de `requirements.txt` :

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py verify
.venv/bin/python run.py predict --input Data/test.csv --output outputs/submission.csv --components outputs/test_components.csv
```

Pour reconstruire la fréquence, conserver la gravité, évaluer puis exporter :

```sh
.venv/bin/python run.py train --threads 2
```

Cette commande ajuste les 20 modèles dans cinq plis de calibration puis les 20
modèles finaux. Le modèle combiné prêt à utiliser est `models/global_model.joblib`,
avec ses paramètres, origines et contrôles dans `models/global_model.json`.

La soumission courante est **`outputs/submission.csv`**, colonnes **`index,pred`**.
`pred` est la prime pure ; les composantes sont dans `outputs/test_components.csv`.
Les exports individuels restent locaux et sont ignorés par Git. Le modèle et
les résultats agrégés sont versionnés. Aucune soumission Kaggle n’est automatique.

## Organisation

```text
modele_global.ipynb          notebook principal
run.py                      entraîner, prédire, vérifier
src/pricing_model/          fréquence et combinaison
components/severity/        gravité publiée inchangée
models/                     modèle global et manifeste
reports/                    évaluation agrégée
Data/                       données source communes
outputs/                    soumission et prédictions locales
archive/                    versions précédentes et recherches
tests/                      contrôles du modèle actuel
integration_sources.json    branches et commits d’origine
```

Tests : `python -m pytest`. Les tests des archives ne sont pas collectés.
