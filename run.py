"""Main entry point: python run.py train / predict / verify."""
from pathlib import Path
import argparse
import hashlib
import json
import importlib.metadata
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'components/severity')]

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, roc_auc_score, brier_score_loss
from pricing_model.frequency import FrequencyModel, SETTINGS, SEEDS
from pricing_model.global_model import GlobalModel
from qualitative_features import QualitativeSeverityEstimator

MODEL = ROOT / 'models/global_model.joblib'
MANIFEST = ROOT / 'models/global_model.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, dtype={'code_postal': 'string'})


def load():
    manifest = json.loads(MANIFEST.read_text())
    if sha(MODEL) != manifest['model_sha256']:
        raise ValueError('The combined model does not match its manifest.')
    return joblib.load(MODEL)


def write_predictions(model, frame, output, components=None):
    result = model.predict_components(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result[['index', 'prime_predite']].rename(columns={'prime_predite': 'pred'}).to_csv(output, index=False)
    if components:
        Path(components).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(components, index=False)
    print(f'{len(result):,} pure premiums written to {output}', flush=True)
    return result


def train(args):
    data = Path(args.data)
    frame = read(data / 'train.csv')
    test = read(data / 'test.csv')
    severity_dir = ROOT / 'components/severity'
    severity_manifest = json.loads((severity_dir / 'models/gravite_final.json').read_text())
    severity_path = severity_dir / severity_manifest['model_path']
    if sha(severity_path) != severity_manifest['model_sha256']:
        raise ValueError('The preserved severity artifact was modified.')
    expected_data = severity_manifest['data_sha256']
    for name in ['train.csv', 'test.csv']:
        expected = next(v for k, v in expected_data.items() if Path(k).name == name)
        if sha(data / name) != expected:
            raise ValueError(f'{name} differs from the preserved severity training source.')
    if not frame['index'].is_unique or frame['id_client'].isna().any():
        raise ValueError('Training indexes must be unique and client groups populated.')
    if not frame[['nombre_sinistres', 'montant_sinistre']].notna().all().all():
        raise ValueError('Missing training targets.')
    if not frame['nombre_sinistres'].isin([0, 1]).all():
        raise ValueError('Expected binary claim counts.')
    if not frame['nombre_sinistres'].gt(0).equals(frame['montant_sinistre'].gt(0)):
        raise ValueError('Claim occurrence and amount disagree.')
    fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2026).split(
        frame, groups=frame['id_client']))
    assert set(frame.iloc[fit]['id_client']).isdisjoint(set(frame.iloc[valid]['id_client']))
    severity = joblib.load(severity_path)
    frequency = FrequencyModel().fit(frame.iloc[fit], frame.iloc[fit]['nombre_sinistres'],
                                     frame.iloc[fit]['id_client'], threads=args.threads)
    model = GlobalModel(frequency, severity)
    validation = model.predict_components(frame.iloc[valid])
    p, g, premium = [validation[c].to_numpy() for c in ['probabilite_sinistre', 'gravite_predite', 'prime_predite']]
    amounts = frame.iloc[valid]['montant_sinistre'].to_numpy()
    claims = frame.iloc[valid]['nombre_sinistres'].to_numpy()
    positive = amounts > 0
    metrics = {
        'validation_contracts': len(valid), 'validation_positive_claims': int(positive.sum()),
        'training_contracts': len(fit), 'training_positive_claims': int(frame.iloc[fit]['nombre_sinistres'].sum()),
        'shared_clients': 0,
        'premium_rmse_euros': float(np.sqrt(mean_squared_error(amounts, premium))),
        'premium_mae_euros': float(mean_absolute_error(amounts, premium)),
        'premium_predicted_observed_ratio': float(premium.sum() / amounts.sum()),
        'constant_premium_rmse_euros': float(np.sqrt(mean_squared_error(amounts, np.full(len(valid), frame.iloc[fit]['montant_sinistre'].mean())))),
        'frequency_log_loss': float(log_loss(claims, p)),
        'frequency_auc': float(roc_auc_score(claims, p)),
        'frequency_brier': float(brier_score_loss(claims, p)),
        'severity_rmse_euros': float(np.sqrt(mean_squared_error(amounts[positive], g[positive]))),
        'severity_mae_euros': float(mean_absolute_error(amounts[positive], g[positive])),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    validation.assign(montant_observe=amounts).to_csv(output / 'validation_components.csv', index=False)
    # The held-out predictions above belong to the 80% evaluation fit only.
    # Freeze the recipes, then fit independent production components on all data.
    print('Production: fitting frequency on all training contracts', flush=True)
    frequency = FrequencyModel().fit(frame, frame['nombre_sinistres'],
                                     frame['id_client'], threads=args.threads)
    positive_fit = frame['montant_sinistre'].gt(0)
    print('Production: fitting the unchanged severity recipe on all positive claims', flush=True)
    severity = QualitativeSeverityEstimator(**severity_manifest['configuration']).fit(
        frame.loc[positive_fit], frame.loc[positive_fit, 'montant_sinistre'])
    model = GlobalModel(frequency, severity)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    reloaded = joblib.load(MODEL)
    np.testing.assert_allclose(model.predict(frame.iloc[valid].head(50)), reloaded.predict(frame.iloc[valid].head(50)), rtol=0, atol=0)
    changed = frame.iloc[valid].head(50).copy()
    changed[['nombre_sinistres', 'montant_sinistre']] = -999
    np.testing.assert_allclose(model.predict(frame.iloc[valid].head(50)), model.predict(changed), rtol=0, atol=0)
    results = ROOT / 'reports'; results.mkdir(exist_ok=True)
    (results / 'validation.json').write_text(json.dumps(metrics, indent=2) + '\n')
    sources = json.loads((ROOT / 'integration_sources.json').read_text())
    manifest = {
        'version': 'global-2026-11-09-full-data', 'status': 'current main model',
        'formula': 'P(claim | X) * E[claim amount | claim, X]',
        'sources': sources, 'model_path': str(MODEL.relative_to(ROOT)), 'model_sha256': sha(MODEL),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost-cpu', 'joblib']},
        'source_sha256': {str(path.relative_to(ROOT)): sha(path)
                          for path in [ROOT / 'run.py', *sorted((ROOT / 'src/pricing_model').glob('*.py'))]},
        'severity': {'model': 'four-variable XGBoost Gamma, original recipe refitted',
                     'source_artifact_sha256': sha(severity_path),
                     'configuration': severity_manifest['configuration'],
                     'features': severity_manifest['features'], 'refitted': True,
                     'fit_positive_claims': int(positive_fit.sum())},
        'frequency': {'model': '50% logistic E + 50% Platt-calibrated XGBoost ensemble',
                      'settings': SETTINGS, 'seeds': SEEDS, 'blend_weight': .5,
                      'fit_contracts': len(frame), 'calibration_contracts': frequency.calibration_rows_},
        'production_fit': {'training_contracts': len(frame),
                           'positive_claims': int(positive_fit.sum()),
                           'test_rows_used_for_fit': 0},
        'validation_scope': 'separate 80-percent fit; not the full-data production model',
        'split': {'method': 'GroupShuffleSplit', 'test_size': .2, 'random_state': 2026, 'group': 'id_client'},
        'validation': metrics, 'validation_previously_explored': True, 'external_validation': False,
        'reload_verified': True, 'prediction_target_invariance_verified': True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    write_predictions(reloaded, test, output / 'submission.csv', output / 'test_components.csv')
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train', help='Validate on the common 80/20 split, then refit both recipes on all training data and export.')
    fit.add_argument('--data', default=str(ROOT / 'Data'))
    fit.add_argument('--output', default=str(ROOT / 'outputs'))
    fit.add_argument('--threads', type=int, default=2)
    predict = sub.add_parser('predict', help='Predict pure premiums without retraining.')
    predict.add_argument('--input', required=True)
    predict.add_argument('--output', default=str(ROOT / 'outputs/submission.csv'))
    predict.add_argument('--components')
    sub.add_parser('verify', help='Check the combined model checksum and metadata.')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        write_predictions(load(), read(args.input), args.output, args.components)
    else:
        load()
        print('Combined model checksum and deserialization verified.')


if __name__ == '__main__':
    main()
