# Recherches après le score public 1365.745

Le score public de départ transmis par l'utilisateur est **1365.745** (`submission_variable_models.csv`). La correction de fréquence obtient désormais **1365.657**, également transmis par l’utilisateur : c’est le nouveau meilleur score public observé, encore 0.157 au-dessus de 1365.500. Les recherches suivantes évaluent la recette fixe du modèle à 1365.745, réapprise dans chaque pli externe. Sa RMSE externe **541.971031** diffère de **541.942903** : cette dernière évaluait l'ancienne procédure de sélection, qui choisissait parfois XGBoost et d'autres poids dans chaque pli.

Chaque recherche utilise trois plis externes et cinq plis internes séparés par client. Catégories, imputations, courbes, corrections, pondérations auxiliaires et sélection sont apprises dans les données d'apprentissage. Le coût des sinistres extrêmes reste intact. Les données et le leaderboard ont déjà été consultés : ces évaluations restent diagnostiques, sans test indépendant nouveau.

| Recherche | RMSE sélection | RMSE recette fixe à 1365.745 | Gain | Export |
| --- | ---: | ---: | ---: | --- |
| smooth_severity | 542.000491010 | 541.971031162 | -0.029459849 | Aucun export |
| coverage_interactions | 541.986790685 | 541.971031162 | -0.015759524 | Aucun export |
| frequency_curves | 541.915570844 | 541.971031162 | 0.055460318 | deliverables\candidates\submission_frequency_curves.csv |

## Gravité : courbes et interactions

`refine_smooth_severity.py` teste 19 configurations : régularisation Ridge 1/3/10/30/100, nombre de nœuds 3/4/5/6, regroupement des catégories avec seuils 5/15/30/50, pénalisation relative des catégories, et corrections Gamma/Poisson à lien log. 96 choix avec mélanges, dont le meilleur modèle actuel. Sa sélection externe se dégrade ; aucun CSV n'est exporté.

`coverage_interactions.py` teste 12 corrections hiérarchiques : courbes de caractéristiques véhicule × couverture, courbes conducteur × usage, ou les deux ; deux pénalisations et deux échelles d'interaction. Les interactions sont centrées avec des moyennes d'apprentissage pondérées. 37 choix avec mélanges et option inchangée. La sélection externe se dégrade ; aucun CSV n'est exporté.

## Fréquence : correction régularisée des probabilités originales

`frequency_curves.py` ajuste `logit(p_nouveau) = logit(p_original) + poids × correction(X)`. La correction apprend une log-loss binomiale pénalisée, sans poids de classe ni repondération des observations. Le gradient analytique est vérifié numériquement. Les probabilités utilisées pour apprendre les corrections proviennent de trois sous-plis séparés par client, avec une copie exacte du modèle de fréquence original. L'utilisation des probabilités d'apprentissage sur les mêmes clients est ainsi évitée.

Neuf configurations : courbe d'âge du véhicule, courbe avec couverture/usage et interaction âge × couverture, ou panneau compact avec âge conducteur, bonus, prix et puissance ; pénalités 100/1000/10000. 28 choix avec corrections à 25/50/100 % et option inchangée. La gravité conserve exactement la recette du modèle à 1365.745. La sélection se fait sur la RMSE de prime ; log-loss et Brier sont aussi enregistrés.

```json
{
  "selected": {
    "MSE": 293672.4859229057,
    "RMSE": 541.9155708437484,
    "MAE": 191.09076225608067,
    "bias": 0.12389198183611065,
    "predicted_observed_ratio": 1.0011985369843899
  },
  "incumbent": {
    "MSE": 293732.59861827455,
    "RMSE": 541.9710311615138,
    "MAE": 190.73231784795638,
    "bias": -0.3664328533347433,
    "predicted_observed_ratio": 0.9964551109724102
  },
  "folds": [
    {
      "fold": 1,
      "choice": {
        "candidate": 3,
        "weight": 1.0,
        "selection_RMSE": 527.7324311089228,
        "incumbent_RMSE": 527.7564559031163
      },
      "selected": {
        "MSE": 324220.35772208276,
        "RMSE": 569.403510458166,
        "MAE": 193.08095271838437,
        "bias": -4.396378029411644,
        "predicted_observed_ratio": 0.95871547165261
      },
      "incumbent": {
        "MSE": 324307.19299520645,
        "RMSE": 569.4797564402148,
        "MAE": 192.9481387280579,
        "bias": -4.620264974492851,
        "predicted_observed_ratio": 0.9566130439566793
      }
    },
    {
      "fold": 2,
      "choice": {
        "candidate": 3,
        "weight": 1.0,
        "selection_RMSE": 553.5790083155395,
        "incumbent_RMSE": 553.639520274331
      },
      "selected": {
        "MSE": 268289.5818156117,
        "RMSE": 517.966776748868,
        "MAE": 191.47761889994132,
        "bias": 3.1705241039184995,
        "predicted_observed_ratio": 1.0310511866460585
      },
      "incumbent": {
        "MSE": 268364.0640068791,
        "RMSE": 518.0386703778775,
        "MAE": 191.00586515143374,
        "bias": 2.532467360124954,
        "predicted_observed_ratio": 1.024802245337641
      }
    },
    {
      "fold": 3,
      "choice": {
        "candidate": 3,
        "weight": 1.0,
        "selection_RMSE": 544.2979361088784,
        "incumbent_RMSE": 544.3730672652504
      },
      "selected": {
        "MSE": 288507.2083205645,
        "RMSE": 537.1286701718356,
        "MAE": 188.71357252138486,
        "bias": 1.597618292811701,
        "predicted_observed_ratio": 1.0157382486737379
      },
      "incumbent": {
        "MSE": 288526.2264766571,
        "RMSE": 537.1463734185097,
        "MAE": 188.2428002963117,
        "bias": 0.9885803535300961,
        "predicted_observed_ratio": 1.0097385736679607
      }
    }
  ],
  "gain": {
    "gain_RMSE": 0.05546031776543714,
    "interval_95": [
      0.0046495906212925325,
      0.11078638878312856
    ],
    "repetitions": 2000,
    "interpretation": "Descriptive paired client bootstrap conditional on fitted predictions; not full training uncertainty."
  },
  "note": "Previously explored clients; nested validation is diagnostic. Severity is exactly the fixed 1365.745 recipe.",
  "frequency": {
    "selected": {
      "log_loss": 0.2139301673711023,
      "brier": 0.054071440662871346
    },
    "incumbent": {
      "log_loss": 0.21404432055967237,
      "brier": 0.054085306146468504
    }
  },
  "final_choice": {
    "candidate": 3,
    "weight": 1.0,
    "selection_RMSE": 541.8759551601989,
    "incumbent_RMSE": 541.9278839363125
  },
  "export": {
    "file": "deliverables\\candidates\\submission_frequency_curves.csv",
    "sha256": "463eed0812b6e6de59dbe1f2218c234c134f4d198205deffa8a19299470b9803",
    "model_sha256": "cdbda24f51861834bc97a8666ca315debbf9e49aba5e5689c37295960ebffa07",
    "kaggle_score": 1365.657
  },
  "recommendation": "new user-reported public incumbent"
}
```

Les intervalles bootstrap sont descriptifs, appariés par client, conditionnels aux prédictions ajustées ; ils ne couvrent pas toute l'incertitude d'apprentissage. Aucun score Kaggle nouveau n'est inféré des résultats locaux. Le seuil strictement inférieur à 1365.500 reste non établi.

57 tests passent, le fichier du meilleur modèle est préservé, et le notebook original reste identique. Chaque éventuel export est reproduit octet pour octet dans un processus distinct et vérifié sur 50 000 lignes. Les preuves et performances par contrat sont dans les sous-dossiers de `reports/`.

```powershell
.\.venv\Scripts\python.exe scripts/refine_smooth_severity.py train
.\.venv\Scripts\python.exe scripts/coverage_interactions.py train
.\.venv\Scripts\python.exe scripts/frequency_curves.py train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/summarize_variable_followups.py
```

Les fichiers historiques sont conservés. L'accès automatisé à Kaggle reste bloqué par le réglage navigateur enregistré.
