# Correction limitée du modèle initial

L'objectif reste un score public Kaggle strictement inférieur à **1365.500**. Aucun score atteignant ce seuil n'est confirmé.

Cette expérience conserve la gravité initiale **Gamma, alpha = 0,1** et corrige uniquement la fréquence :

```
p_corrige = sigmoid(logit(p_logistique_initiale) + poids × correction_XGBoost)
prime = p_corrige × gravite_Gamma_initiale
```

Le mécanisme utilise les marges brutes décrites dans la [documentation officielle XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html). Le correcteur est appris sur des marges logistiques calculées hors client dans trois plis internes. Les modèles externes et leurs prédictions excluent également les clients évalués. La gravité est toujours apprise uniquement sur les sinistres positifs.

Six correcteurs sont fixés avant la recherche : profondeurs 1, 2 ou 3 ; 80 ou 240 arbres ; taux 0,025 ; régularisation L2 100 ; poids minimum d'enfant 100. Quatre poids de correction sont comparés : 0,1, 0,25, 0,5 et 1. Le modèle initial sans correction est également candidat, soit **25 possibilités** au total.

La sélection utilise les cinq plis de développement habituels, puis son gagnant est fixé avant une deuxième partition en cinq plis. Le gagnant de la deuxième partition n'est pas utilisé pour remplacer ce choix. Un nouveau CSV n'est exporté que si le candidat améliore les deux partitions. Ces deux partitions réutilisent des données déjà explorées : elles ne constituent pas un test indépendant. Aucun seuil local ne garantit le score public demandé.

```powershell
.\.venv\Scripts\python.exe scripts/anchored_search.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_anchored_model.py
```

Les résultats et la provenance se trouvent dans `reports/anchored/`. Le notebook protégé, le modèle initial et les fichiers des recherches précédentes sont préservés.

## Résultats

| Évaluation | Référence initiale | Correction retenue |
| --- | ---: | ---: |
| Cinq plis de sélection | 550,341563 | 550,291854 |
| Deuxième partition, candidat déjà fixé | 550,341682 | 550,302453 |
| Diagnostic historique déjà exploré | 507,647017 | 507,555159 |

Le correcteur retenu utilise 240 arbres de profondeur 3, avec un poids de 0,5. Le gain sur la deuxième partition vaut 0,039229 RMSE ; son intervalle descriptif à 95 % est [-0,042368 ; 0,119764]. Il contient zéro. Il s'agit donc d'un challenger mesuré, pas d'une supériorité prouvée ni d'un score Kaggle confirmé.

Le réentraînement sans correction reproduit les 50 000 prédictions originales à moins de 8 × 10⁻¹⁰ par ligne. Avec correction, l'écart absolu moyen par rapport à l'original vaut environ 3,14 unités de prime. La gravité est inchangée.

Fichier d'essai : `deliverables/candidates/submission_anchored.csv`. Le modèle est sauvegardé dans `models/anchored.joblib` et rechargé dans un processus séparé pour vérifier les prédictions. Le fichier original gagnant reste disponible dans `deliverables/candidates/submission_reference.csv`.

## Résultat Kaggle transmis par l'utilisateur

Le 12 octobre 2026, l'utilisateur a transmis le résultat « Complete » : **1366.010**. Ce score est moins bon que **1365.943** pour la référence. Le challenger ancré n'est donc pas retenu. La provenance est conservée dans `reports/anchored/kaggle_result.json` ; le résultat n'a pas été vérifié indépendamment par le navigateur.
