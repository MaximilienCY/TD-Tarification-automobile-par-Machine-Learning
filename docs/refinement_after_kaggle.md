# Suivi après les résultats Kaggle décevants

## Résultats transmis par l'utilisateur

| Fichier | Score Kaggle | Décision |
| --- | ---: | --- |
| Référence initiale | 1365.943 | Référence conservée |
| submission_anchored.csv | 1366.010 | Écarté |
| submission_restart_xgboost.csv | 1366.025 | Écarté |
| submission_severity_refinement.csv | 1365.913 | Nouveau meilleur score public transmis |

Les nouveaux scores sont ceux transmis par l'utilisateur le 12 octobre 2026, sans vérification indépendante par le navigateur. Ils sont associés aux empreintes des fichiers locaux dans les reçus et le registre de soumissions. Le seuil demandé, strictement inférieur à 1365.500, n'est pas atteint.

## Deux expériences séparées

La fréquence et la gravité ne sont plus modifiées simultanément. Chaque recherche utilise les mêmes trois plis externes et cinq plis internes séparés par client, avec la référence originale parmi les candidats. Ces données ont déjà été explorées ; il ne s'agit pas d'un nouveau test indépendant. Le score Kaggle n'entre pas dans les ajustements ni dans le classement des paramètres, mais cette recherche succède à la consultation du leaderboard.

### Gravité avec fréquence originale inchangée

34 configurations : Gamma avec décalages 0/300/600 ; Poisson quasi-vraisemblable avec décalages 0/600 ; effets d'âge linéaires ou splines ; pondération uniforme ou par probabilité de sinistre ; régularisation Gamma supplémentaire ; corrections multiplicatives XGBoost limitées (profondeurs 1–3, 80/240 arbres, L2 100). Les mélanges avec Gamma prennent les poids 0/0,1/0,25/0,5/1. Le décalage de 600 est une hypothèse de modélisation suggérée par le support observé, pas une franchise contractuelle connue.

Les probabilités auxiliaires et gravités de référence sont calculées hors client sur trois plis internes supplémentaires. Les probabilités auxiliaires proviennent d'une logistique compacte. Sur les seuls sinistres, pondérer une perte quadratique de gravité par p(x) vise la partie dépendante du modèle de l'erreur de prime : la densité des sinistres est déjà proportionnelle à p(x), tandis que l'erreur de prime dépend de p(x)². Les pertes Gamma/Poisson n'égalent pas exactement cette perte ; leur sélection se fait donc sur la RMSE de prime.

Pour XGBoost, la cible est le ratio coût/gravité de référence hors client, et le poids est p × gravité², normalisé à moyenne 1. L'identité `p × s² × (y/s − ratio)² = p × (y − s × ratio)²` est testée. Les sinistres extrêmes sont conservés. Les catégories et transformations apprises restent dans l'apprentissage de chaque pli. [GammaRegressor et sample_weight](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html), [objectifs XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

### Fréquence avec Gamma originale inchangée

16 configurations : C=0,03/0,1/0,3/1, pondération uniforme ou par gravité auxiliaire², log-loss ou Brier. Elles réutilisent exactement la préparation originale (dont l'encodage cible croisé par client). La gravité des poids exclut chaque client par apprentissage croisé. La perte Brier avec lien logistique est optimisée localement par L-BFGS, initialisée à la solution logistique ; les gradients et la convergence sont testés. Ce sont des pertes propres pour la probabilité conditionnelle lorsque les poids sont des fonctions des covariables. Les poids estimés et la restriction de modèle ne garantissent pas une amélioration.

## Résultats externes

| Procédure | RMSE | Gain face à l'original | Intervalle descriptif bootstrap clients 95 % |
| --- | ---: | ---: | --- |
| Original | 542.060130 | 0.000000 | [0.000000, 0.000000] |
| Severity only | 542.055720 | 0.004410 | [-0.030733, 0.039481] |
| Frequency losses | 542.079618 | -0.019488 | [-0.084206, 0.047508] |

Le gain local de gravité est trop petit et trop incertain pour annoncer une supériorité générale. La nouvelle procédure de fréquence est écartée et n'exporte aucun CSV. Le modèle de gravité est désormais le meilleur score public transmis, à 1365.913 ; il devient le benchmark des recherches suivantes.

## Challenger exploratoire de gravité

Configuration finale sélectionnée :

```json
{
  "candidate": 26,
  "weight": 0.5,
  "config": {
    "kind": "xgb_ratio",
    "panel": "basic",
    "depth": 2,
    "trees": 240
  },
  "selection_RMSE": 541.9644365112739,
  "reference_RMSE": 541.981085620689
}
```

Le fichier `deliverables/candidates/submission_severity_refinement.csv` conserve exactement les probabilités originales. La gravité est un mélange à 50 % de Gamma originale et à 50 % d'une correction multiplicative XGBoost (240 arbres, profondeur 2), limitée à âge du véhicule, couverture et usage. Son score Kaggle est **1365.913**. Il est conservé comme nouvel incumbent public ; cela ne prouve pas une supériorité sur le classement privé.

Les 50 000 lignes, l'ordre et les index sont vérifiés ; le CSV est reproduit octet pour octet dans un nouveau processus, et le modèle sauvegardé est rechargé. Le notebook protégé est inchangé. 39 tests passent.

```powershell
.\.venv\Scripts\python.exe scripts/refine_severity.py train
.\.venv\Scripts\python.exe scripts/refine_frequency.py train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/refine_severity.py predict --output reports/severity_refinement/reproduced.csv
.\.venv\Scripts\python.exe scripts/summarize_refinement.py
```

Les résultats, choix de chaque pli et empreintes sont dans `reports/severity_refinement/` et `reports/frequency_refinement/`. Les scripts ne modifient ni le notebook original, ni les CSV précédents. Le cache dépend des données, des sources et des versions.
