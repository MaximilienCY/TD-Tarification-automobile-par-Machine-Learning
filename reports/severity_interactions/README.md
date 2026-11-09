# Controlled severity interaction experiment

**Decision: retain the current production severity model.** None of the five
challengers demonstrates a reliable reduction in final premium RMSE.

The experiment holds frequency predictions fixed and trains each severity
candidate only on positive claims in the training portion of each client fold.
Scoring uses all contracts in the held-out fold, including zero claims.
All candidates retain the current four severity features, Gamma objective,
400 depth-one trees, regularization and seed. Feature imputation is fitted
inside each severity training fold. The reproduced baseline agrees exactly
with the existing severity estimator.

## Candidates and findings

- Current severity: premium RMSE **541.6101**.
- Add power, principal-driver licence experience, speed and principal-driver age:
  **541.6353**.
- Added variables plus power × vehicle age: **541.6476**.
- Added variables plus power / (1 + principal-driver licence experience):
  **541.6496**. This ratio represents power relative to driving experience;
  it is not a power × experience product.
- Added variables plus speed × principal-driver age: **541.6353**.
- Added variables plus all three interactions: **541.6609**.

Each challenger wins on only one or two of the five folds. All paired-client
bootstrap 99% intervals for the RMSE difference contain zero. The intervals use
2,000 resamples and approximately account for screening five challengers, but
do not account for earlier research on this dataset.

On the prior 80/20 holdout, all interactions together improve RMSE from
535.2139 to 535.0414. That isolated result conflicts with the broader five-fold
comparison and is insufficient to promote the candidate. Positive-claim severity
RMSE improves for some candidates while final premium RMSE worsens; component
metrics alone should not select the global model.

The prespecified screening rule required an upper 99% difference bound below
zero, improvement in at least four folds, and improvement on the old holdout.
No candidate meets it. The deployed full-data model and submission are retained.

## Scope and reproducibility

Source: 50,000 training contracts, including 2,917 positive claims. Five
StratifiedGroupKFold folds by id_client, seed 42. Frozen frequency is the exact
OOF blend export from commit 682b3c5 of projet-simplifie, identified by SHA256
in results.json. It is shared unchanged by every candidate. It comes from the
source-branch runtime, rather than newly computed probabilities from the current
full-data production model. Its historical calibration and selection reused some
OOF labels. The secondary holdout has also been explored before. These are
controlled exploratory comparisons, not fresh external validation or Kaggle scores.

Run `.venv/bin/python experiments/severity_interactions.py` from the repository
root. The script restores and verifies the frequency export from the original
commit in local Git history; it does not contact Kaggle. Individual predictions
are local in outputs/severity_interactions/oof_predictions.csv. Aggregates,
fold results, uncertainty and configuration are in results.json and metrics.csv.
View comparison.png or experiments/severity_interactions.ipynb for the comparison.

This experiment does not establish that other interaction definitions,
hyperparameters, objectives or model families cannot help. It rejects promoting
these particular additions under the frozen current severity recipe.
