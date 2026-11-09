# Résultats de la reprise XGBoost

Trois plis externes évaluent la sélection complète sur les 50 000 contrats, avec cinq plis internes et 60 essais par composante. Les données ont déjà été explorées : cette évaluation ne constitue pas un test neuf.

| Mesure externe | Référence initiale | Procédure de sélection |
| --- | ---: | ---: |
| RMSE | 542.060130 | 542.128072 |
| MSE | 293829.184497 | 293902.846967 |
| MAE | 190.861365 | 191.011443 |
| bias | -0.329742 | -0.458240 |

Gain RMSE : **-0.067943** ; intervalle descriptif bootstrap clients à 95 % : **[-0.215007 ; 0.065353]**. Recommandation : **original reference**.

Le choix final sur tous les clients est :

```json
{
  "frequency": {
    "config": {
      "trial": 21,
      "feature_set": "compact",
      "cleaned": true,
      "objective": "binary:logistic",
      "max_depth": 3,
      "learning_rate": 0.034553383920682385,
      "subsample": 0.9087300373134536,
      "colsample_bytree": 0.8099137041918788,
      "min_child_weight": 12.059789109228127,
      "reg_lambda": 0.671505419843041,
      "reg_alpha": 0.01,
      "max_cat_to_onehot": 8,
      "max_cat_threshold": 32
    },
    "trees": 126,
    "weight": 0.5,
    "calibrated": true
  },
  "severity": {
    "config": {
      "trial": 48,
      "feature_set": "compact",
      "cleaned": false,
      "objective": "reg:gamma",
      "max_depth": 3,
      "learning_rate": 0.029810526459648254,
      "subsample": 0.8368323959570091,
      "colsample_bytree": 0.7881979761754254,
      "min_child_weight": 32.575020953467444,
      "reg_lambda": 0.3130355746350659,
      "reg_alpha": 0.0,
      "max_cat_to_onehot": 8,
      "max_cat_threshold": 32
    },
    "trees": 98,
    "weight": 0.5,
    "calibrated": false
  }
}
```

Les catalogues détaillés des configurations retenues sont dans `reports/restart_xgboost/full_training/selection.json`. Les scores de cette sélection interne sont optimistes et ne remplacent pas les résultats externes ci-dessus.

Dans les trois recherches internes externes, les quinze configurations du top cinq de fréquence utilisent le panneau compact. La gravité sélectionnée et les poids varient davantage entre plis. Les configurations de panneaux utilisent des tirages de paramètres différents : cette observation n'est pas une ablation contrôlée ni une preuve causale de l'effet d'une variable.

Le challenger est `deliverables/candidates/submission_restart_xgboost.csv`. Le fichier original reste disponible et les anciennes expériences sont conservées. Le modèle sauvegardé reproduit les prédictions dans un processus séparé, octet pour octet. Le notebook `notebooks/04_reprise_xgboost.ipynb` est exécuté sans erreur. Les 30 tests passent.

Les importances de gain dans `reports/restart_xgboost/feature_importance.csv` sont relatives à chaque composante et ne mesurent pas des effets causaux.

**Kaggle : 1366.025 (résultat transmis par l’utilisateur, non vérifié indépendamment dans le navigateur).** Le seuil de 1365.500 n'est pas atteint. La référence initiale reste à 1365.943. Le modèle XGBoost n'est pas recommandé pour remplacer cette référence.
