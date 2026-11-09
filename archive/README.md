# Versions précédentes et recherches

La version principale est [`../modele_global.ipynb`](../modele_global.ipynb),
avec `../run.py` comme point d'entrée.

- `legacy/` : ancien dossier `previous/`, pipelines, livrables et soumissions.
- `severity_research/` : ancien `gravite_xgboost/`. Son modèle publié de gravité
  est repris dans `../components/severity/` sans modification de l'artefact.
- `severity_analysis/` : ancien `nouveau_notebook/`, analyse, interactions et
  sélection exploratoire de variables de gravité.
- `frequency_projet_simplifie/` : notebooks et documentation de la branche
  `projet-simplifie` au commit `682b3c5`, sources du modèle de fréquence.
- `README_before_merge.md` : documentation principale avant l'intégration.

Les chemins des notebooks historiques correspondent à leur organisation d'origine.
Ils sont conservés comme référence ; exécuter le notebook principal pour la version
actuelle. Les données, modèles et exports individuels historiques locaux restent
dans ces archives ; leurs fichiers générés ne sont pas ajoutés à la publication Git.
Les fichiers de la branche de fréquence restent accessibles dans son historique fusionné.
