"""Controlled severity ablation; never replaces the production model.

Run from the repository root: .venv/bin/python experiments/severity_interactions.py
Frequency probabilities are frozen source-branch OOF exports. Their historical
calibration/model selection reused some labels: results are comparative research,
not a fresh external estimate of generalization.
"""
from pathlib import Path
import json
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'components/severity')]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor
from qualitative_features import QualitativeFeatures, QualitativeSeverityEstimator
import run as app

VARIANTS = ['baseline', 'main_effects', 'power_vehicle_age',
            'power_license_experience', 'speed_driver_age', 'all_interactions']


class InteractionFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, spec=None, variant='baseline'):
        self.spec = spec
        self.variant = variant

    def fit(self, X, y=None):
        self.base_ = QualitativeFeatures(**self.spec).fit(X)
        return self

    def transform(self, X):
        z = self.base_.transform(X)
        if self.variant == 'baseline':
            return z
        f = self.base_._clean(X)
        # Raw variables are retained alongside their products. Imputation is
        # learned only from positive claims in the current training fold.
        for col in ['din_vehicule', 'anciennete_permis1',
                    'vitesse_vehicule', 'age_conducteur1']:
            z[col] = f[col]
        if self.variant in ['power_vehicle_age', 'all_interactions']:
            z['power_x_vehicle_age'] = f.din_vehicule * f.anciennete_vehicule
        if self.variant in ['power_license_experience', 'all_interactions']:
            z['power_per_license_experience'] = f.din_vehicule / (1 + f.anciennete_permis1)
        if self.variant in ['speed_driver_age', 'all_interactions']:
            z['speed_x_principal_driver_age'] = f.vitesse_vehicule * f.age_conducteur1
        return z.replace([np.inf, -np.inf], np.nan)


def estimator(config, variant):
    return Pipeline([
        ('features', InteractionFeatures(config['spec'], variant)),
        ('encoding', ColumnTransformer([
            ('num', SimpleImputer(strategy='median', add_indicator=True,
                                  keep_empty_features=True), make_column_selector(dtype_include=np.number)),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 min_frequency=5, sparse_output=True, dtype=np.float32),
             make_column_selector(dtype_exclude=np.number))], sparse_threshold=1.)),
        ('xgb', XGBRegressor(objective=config['objective'], tree_method='hist',
                             n_jobs=2, random_state=2026, verbosity=0, **config['params']))])


def metrics(y, premium, severity):
    positive = y > 0
    return {'premium_rmse': float(np.sqrt(np.mean((y-premium)**2))),
            'premium_mae': float(np.mean(np.abs(y-premium))),
            'premium_predicted_observed_ratio': float(premium.sum()/y.sum()),
            'severity_rmse_positive': float(np.sqrt(np.mean((y[positive]-severity[positive])**2)))}


def main():
    out = ROOT/'reports/severity_interactions'; out.mkdir(parents=True, exist_ok=True)
    private = ROOT/'outputs/severity_interactions'; private.mkdir(parents=True, exist_ok=True)
    frame = app.read(ROOT/'Data/train.csv')
    cache = ROOT/'outputs/kaggle_diagnostic/frequence_melange_oof_train.csv'
    sources = json.loads((ROOT/'integration_sources.json').read_text())
    source_export = subprocess.run(
        ['git', '-c', 'safe.directory=*', 'show',
         sources['frequency_commit'] + ':Data/predictions/frequence_melange_oof_train.csv'],
        cwd=ROOT, check=True, capture_output=True).stdout
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.read_bytes() != source_export:
        raise ValueError('Cached frequency differs from the exact source branch export.')
    cache.write_bytes(source_export)
    frozen = pd.read_csv(cache)
    config = json.loads((ROOT/'components/severity/models/gravite_final.json').read_text())['configuration']
    assert frame['index'].equals(frozen['index'])
    y = frame.montant_sinistre.to_numpy(); p = frozen.proba_pred.to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    predictions = {v: np.zeros(len(frame)) for v in VARIANTS}
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for k, (a,b) in enumerate(folds.split(frame, frame.nombre_sinistres, frame.id_client), 1):
        assert (frozen.pli.iloc[b] == k).all()
        assert set(frame.id_client.iloc[a]).isdisjoint(frame.id_client.iloc[b])
        positive = a[y[a] > 0]
        for variant in VARIANTS:
            model = estimator(config, variant).fit(frame.iloc[positive], y[positive])
            predictions[variant][b] = np.maximum(model.predict(frame.iloc[b]), 1.)
            if k == 1 and variant == 'baseline':
                source = QualitativeSeverityEstimator(**config).fit(frame.iloc[positive], y[positive])
                np.testing.assert_allclose(predictions[variant][b], source.predict(frame.iloc[b]), rtol=0, atol=0)
        print(f'Severity experiment: fold {k}/5, six variants complete', flush=True)
    result = {}
    for name, sev in predictions.items():
        result[name] = metrics(y, p*sev, sev)
        result[name]['fold_rmse'] = [metrics(y[frozen.pli==k], (p*sev)[frozen.pli==k], sev[frozen.pli==k])['premium_rmse'] for k in range(1,6)]
        result[name]['delta_rmse'] = result[name]['premium_rmse'] - result['baseline']['premium_rmse']
        result[name]['folds_better_than_baseline'] = int(np.sum(np.array(result[name]['fold_rmse']) < result['baseline']['fold_rmse']))
    # Paired client bootstrap. 99% intervals account approximately for screening
    # five challengers (Bonferroni), but not for earlier historical exploration.
    sq = pd.DataFrame({v: (y-p*s)**2 for v,s in predictions.items()})
    sq['n'] = 1; sq['client'] = frame.id_client
    grouped = sq.groupby('client', sort=False).sum().to_numpy()
    rng = np.random.default_rng(42); differences = []
    for _ in range(2000):
        sums = grouped[rng.integers(0,len(grouped),len(grouped))].sum(axis=0)
        rmse = np.sqrt(sums[:-1]/sums[-1]); differences.append(rmse-rmse[0])
    bounds = np.quantile(differences,[.005,.995],axis=0)
    for i,v in enumerate(VARIANTS):
        result[v]['delta_client_bootstrap_99_interval'] = bounds[:,i].tolist()
    # Secondary check on the exact prior 80/20 population, using the existing
    # frequency holdout probabilities, never the full-data frequency predictions.
    a,b = next(GroupShuffleSplit(test_size=.2,random_state=2026).split(frame, groups=frame.id_client))
    holdout = pd.read_csv(ROOT/'outputs/validation_components.csv')
    assert frame['index'].iloc[b].reset_index(drop=True).equals(holdout['index'])
    positive = a[y[a]>0]
    for v in VARIANTS:
        model = estimator(config,v).fit(frame.iloc[positive],y[positive])
        sev = np.maximum(model.predict(frame.iloc[b]),1.)
        result[v]['prior_holdout_rmse'] = metrics(y[b],holdout.probabilite_sinistre.to_numpy()*sev,sev)['premium_rmse']
    eligible = [v for v in VARIANTS[1:] if result[v]['delta_client_bootstrap_99_interval'][1] < 0
                and result[v]['folds_better_than_baseline'] >= 4
                and result[v]['prior_holdout_rmse'] < result['baseline']['prior_holdout_rmse']]
    report = {'protocol': {'variants':VARIANTS, 'frequency':'frozen source-branch OOF blend',
               'frequency_cache_sha256':app.sha(ROOT/'outputs/kaggle_diagnostic/frequence_melange_oof_train.csv'),
               'severity_configuration':config, 'training_rows':len(frame),
               'positive_claims':int((y>0).sum()), 'group':'id_client', 'fold_seed':42,
               'baseline_source_parity_exact':True, 'production_model_changed':False,
               'limitations':'Historical frequency calibration and model selection reused some OOF labels. Secondary holdout was previously explored. No fresh external validation.'},
              'results':result, 'eligible_challengers':eligible,
              'decision':'Retain production model' if not eligible else 'Challenger merits further validation; production model retained'}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(result).T.to_csv(out/'metrics.csv')
    pd.DataFrame({'index':frame['index'],'fold':frozen.pli,'frequency':p,**predictions}).to_csv(private/'oof_predictions.csv',index=False)
    fig, ax = plt.subplots(figsize=(10,5))
    labels = VARIANTS[1:]; delta = [result[v]['delta_rmse'] for v in labels]
    low=[result[v]['delta_client_bootstrap_99_interval'][0] for v in labels]
    high=[result[v]['delta_client_bootstrap_99_interval'][1] for v in labels]
    ax.errorbar(delta,range(len(labels)),xerr=[np.array(delta)-low,np.array(high)-delta],fmt='o',color='#215e80',capsize=4)
    ax.set_yticks(range(len(labels)),labels); ax.axvline(0,color='gray',linestyle='--')
    ax.set(xlabel='Premium RMSE change versus current severity (€; lower is better)',
           ylabel='Severity candidate', title='Fixed frequency: severity interactions on five client folds')
    fig.text(.02,.015,'Source: 50,000 training contracts; paired client bootstrap, 99% intervals; historical OOF frequency.',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
