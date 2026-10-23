# Tarification automobile — CY Tech 2027

**Prime pure = probabilité de sinistre × montant attendu si sinistre**

## Organisation

```
Data/
  train.csv, test.csv           données Kaggle
  predictions/                  prédictions sauvegardées (fréquence, gravités)
  resultats/                    résultats longs à calculer (sélection des variables)
notebooks/
  01_frequence.ipynb            modèle de fréquence (régression logistique)
  01b_frequence_xgboost.ipynb   modèle de fréquence XGBoost et mélange avec la logistique
  02_gravite.ipynb              modèle de gravité (3 gravités retenues)
  03_prime_finale.ipynb         fréquence × gravité → submission.csv
submission.csv                  meilleure soumission confirmée sur Kaggle (colonnes index, pred)
submission_candidat.csv         logistique E × gravité (Kaggle 1365.915)
submission_melange.csv          mélange XGBoost + logistique E × gravité, meilleur en local, pas encore soumis
HISTORIQUE.md                   tout ce qui a été fait, et où le retrouver
td_data_science.pdf             sujet du TD
```

## Lancer le projet

Avec l'environnement Python décrit dans `requirements.txt`, exécuter les notebooks dans l'ordre 01, 01b, 02, 03. Chaque notebook retrouve tout seul le dossier `Data/`, qu'il soit lancé depuis la racine ou depuis `notebooks/`.

## Métrique Kaggle

RMSE de la prime sur le test. Le notebook 03 calcule la même mesure en local, sur 5 plis par client.
