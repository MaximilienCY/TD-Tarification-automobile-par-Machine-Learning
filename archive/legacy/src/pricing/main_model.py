"""Recette principale figée : correction de fréquence et gravité Gamma/XGBoost/Ridge.

Les recherches historiques ne sont jamais relancées par ce point d'entrée.
Le score Kaggle reste associé au CSV soumis et au reçu historique, pas à un
réentraînement présumé identique sur une autre plateforme.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports/main'
MODEL_PATH = ROOT / 'models/main_model.joblib'
RELEASE_PATH = ROOT / 'deliverables/submission_frequency_curves.csv'
CHOICE = {'candidate': 3, 'weight': 1.0}
RELEASE_WINDOWS_SHA256 = '463eed0812b6e6de59dbe1f2218c234c134f4d198205deffa8a19299470b9803'
RECIPE = {
    'name': 'frequency_curves_1365_657',
    'frequency': {'base': 'logistique_C0.1_encodage_cible_hors_client',
                  'correction': {'panel': 'age_coverage', 'alpha': 100.0, 'weight': 1.0,
                                 'knots': 4, 'degree': 2, 'interaction_scale': 0.5}},
    'severity': {'gamma': {'alpha': 0.1, 'weight': 0.375},
                 'xgboost_ratio': {'depth': 2, 'trees': 240, 'learning_rate': 0.025,
                                   'min_child_weight': 100.0, 'reg_lambda': 100.0,
                                   'reg_alpha': 1.0, 'seed': 20261012, 'weight': 0.375},
                 'ridge_ratio': {'panel': 'all', 'alpha': 10.0, 'knots': 4,
                                 'degree': 2, 'rare_claims': 15, 'weight': 0.25}},
    'target': 'cout_observe_periode_non_annualise',
}


def libraries():
    scripts = str(ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import frequency_curves as pipeline
    assert pipeline.catalogue()[3] == {'panel': 'age_coverage', 'alpha': 100.0}
    assert pipeline.vm.catalogue()[12] == {'kind': 'ridge', 'panel': 'all', 'alpha': 10.0}
    return pipeline


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def release():
    """Vérifier le CSV évalué, en tolérant uniquement sa conversion CRLF/LF."""
    raw = RELEASE_PATH.read_bytes()
    windows = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    if hashlib.sha256(windows).hexdigest() != RELEASE_WINDOWS_SHA256:
        raise ValueError('Le contenu du CSV associé au score public a changé.')
    receipt = json.loads((REPORTS / 'evidence/kaggle_result.json').read_text())
    if receipt['sha256'] != RELEASE_WINDOWS_SHA256 or receipt['public_score'] != 1365.657:
        raise ValueError('Le reçu et la recette principale ne correspondent pas.')
    return pd.read_csv(RELEASE_PATH)


def components(model):
    """Objets effectivement ajustés, sans reconstituer leurs paramètres à la main."""
    variable = model.base
    prior = variable.incumbent
    original = prior.base
    return {'frequency_base': original.frequency, 'frequency_correction': model.correction,
            'gamma': original.base, 'xgboost': prior.members_[0],
            'ridge': variable.correction}


def details(model, frame):
    values = model.details(frame).copy()
    objects = components(model)
    base = objects['gamma'].predict(frame)
    tree = objects['xgboost'].predict(frame, base)
    ridge = objects['ridge'].predict(frame, base)
    np.testing.assert_allclose(values.montant_pred, .375 * base + .375 * tree + .25 * ridge,
                               rtol=1e-12, atol=1e-8)
    values['gamma'] = base
    values['xgboost_corrige'] = tree
    values['ridge_corrige'] = ridge
    values['correction_logit'] = objects['frequency_correction'].delta(frame)
    if (not np.isfinite(values.select_dtypes('number').to_numpy()).all()
            or not values.proba_pred.between(0, 1).all()
            or not values.montant_pred.gt(0).all()):
        raise ValueError('Prédictions hors domaine.')
    np.testing.assert_allclose(values.pred, values.proba_pred * values.montant_pred)
    return values


def load_model():
    libraries()  # Modules nécessaires au rechargement des classes joblib.
    if not MODEL_PATH.exists():
        raise FileNotFoundError('Modèle local absent : exécuter python run.py train.')
    return joblib.load(MODEL_PATH)


def fit_recipe(frame):
    return libraries().Premium(dict(CHOICE)).fit(frame)


def compare_to_release(values):
    expected = release()
    np.testing.assert_array_equal(values['index'], expected['index'])
    error = np.abs(values.pred.to_numpy() - expected.pred.to_numpy())
    return {'maximum_absolute_difference_eur': float(error.max()),
            'mean_absolute_difference_eur': float(error.mean()),
            'numeric_match_rtol_1e_8_atol_1e_6': bool(np.allclose(values.pred, expected.pred,
                                                               rtol=1e-8, atol=1e-6))}


def train_main():
    f = libraries()
    train, test, _, _ = f.r.load_data()
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print('Apprentissage de la recette figée sur 50 000 contrats et 2 917 sinistres.', flush=True)
    model = fit_recipe(train)
    objects = components(model)
    assert objects['gamma'].config == {'alpha': .1}
    assert objects['xgboost'].config['trees'] == 240
    assert objects['ridge'].config == {'kind': 'ridge', 'panel': 'all', 'alpha': 10.}
    joblib.dump(model, MODEL_PATH, compress=3)
    values = details(load_model(), test)
    values.to_csv(REPORTS / 'test_components.csv', index=False)
    values[['index', 'pred']].to_csv(REPORTS / 'recomputed_submission.csv', index=False)
    f.r.valid_submission(REPORTS / 'recomputed_submission.csv', test)
    comparison = compare_to_release(values)
    report = {'recipe': RECIPE, 'model_path': str(MODEL_PATH.relative_to(ROOT)),
              'model_sha256': sha256(MODEL_PATH), 'python': platform.python_version(),
              'versions': {name: f.r.package_version(name) for name in
                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'xgboost']},
              'data_sha256': {p: sha256(ROOT / 'Data' / p) for p in ['train.csv', 'test.csv']},
              'fitting_rows': len(train), 'positive_claims': int(train.nombre_sinistres.sum()),
              'test_rows': len(test), 'seconds': time.perf_counter() - started,
              'submitted_file': 'submission.csv', 'submitted_sha256_local': sha256(ROOT / 'submission.csv'),
              'submitted_sha256_lf': lf_sha256(RELEASE_PATH),
              'reported_sha256_crlf': RELEASE_WINDOWS_SHA256,
              'reported_public_score': 1365.657, 'score_source': 'recu historique utilisateur',
              'independently_kaggle_verified': False, 'local_reproduction': comparison,
              'note': 'Le score reste associé au CSV fourni. Le réentraînement est contrôlé numériquement ; les octets peuvent varier avec la plateforme.'}
    write_json(ROOT / 'reports/final_model.json', report)
    print(json.dumps(comparison, indent=2), flush=True)
    return model


PERMUTATION_GROUPS = {
    'Contrat et usage': ['type_contrat', 'utilisation', 'duree_contrat', 'anciennete_info', 'freq_paiement', 'paiement'],
    'Âge du véhicule': ['anciennete_vehicule'],
    'Conducteur principal et bonus': ['age_conducteur1', 'anciennete_permis1', 'bonus'],
    'Caractéristiques techniques et prix': ['prix_vehicule', 'din_vehicule', 'poids_vehicule', 'cylindre_vehicule', 'vitesse_vehicule', 'debut_vente_vehicule'],
    'Marque, modèle, énergie et type': ['marque_vehicule', 'modele_vehicule', 'essence_vehicule', 'type_vehicule'],
    'Second conducteur': ['conducteur2', 'age_conducteur2', 'anciennete_permis2'],
    'Département': ['code_postal'],
    'Fin de vente (contrôle inutilisé)': ['fin_vente_vehicule'],
}


def diagnostic_main(repeats=3):
    """Rejouer la recette figée hors client ; ce n'est pas une nouvelle sélection imbriquée."""
    f = libraries()
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    train, _, _, _ = f.r.load_data()
    REPORTS.mkdir(parents=True, exist_ok=True)
    parts, folds, importance = [], [], []
    for fold, (a, b) in enumerate(f.r.split_groups(train, 3, f.r.SEED + 23), 1):
        learn, evaluation = train.iloc[a], train.iloc[b]
        assert set(learn.id_client).isdisjoint(evaluation.id_client)
        print(f'Diagnostic de la recette figée : apprentissage du pli {fold}/3.', flush=True)
        model = fit_recipe(learn)
        v = details(model, evaluation)
        v['observed'] = evaluation.montant_sinistre.to_numpy()
        v['claim'] = evaluation.nombre_sinistres.to_numpy()
        v['fold'] = fold
        v['constant'] = learn.montant_sinistre.mean()
        parts.append(v)
        folds.append({'fold': fold, 'learn_clients': int(learn.id_client.nunique()),
                      'evaluation_clients': int(evaluation.id_client.nunique()), 'shared_clients': 0,
                      'learn_rows': len(learn), 'evaluation_rows': len(evaluation),
                      'selected_RMSE': f.r.score(v.observed, v.pred)['RMSE'],
                      'incumbent_RMSE': f.r.score(v.observed, v.incumbent)['RMSE']})
        base_sse = float(np.sum((v.observed.to_numpy() - v.pred.to_numpy()) ** 2))
        rng = np.random.default_rng(20261020 + fold)
        for name, columns in PERMUTATION_GROUPS.items():
            for repetition in range(repeats):
                x = evaluation.copy()
                permutation = rng.permutation(len(x))
                for column in columns:
                    x[column] = evaluation.iloc[permutation][column].to_numpy()
                prediction = model.details(x).pred.to_numpy()
                sse = float(np.sum((v.observed.to_numpy() - prediction) ** 2))
                importance.append({'group': name, 'fold': fold, 'repetition': repetition,
                                   'rows': len(x), 'SSE_increase': sse - base_sse})
        pd.DataFrame(folds).to_csv(REPORTS / 'diagnostic_folds.csv', index=False)
        print(f'Pli {fold}/3 terminé : RMSE {folds[-1]["selected_RMSE"]:.6f}.', flush=True)
    v = pd.concat(parts).set_index('index').loc[train['index']].reset_index()
    v.to_csv(REPORTS / 'diagnostic_oof.csv', index=False)
    result = {'scope': 'recette fixe réapprise ; clients exclus de chaque apprentissage ; données et paramètres déjà explorés',
              'selected': f.r.score(v.observed, v.pred), 'incumbent': f.r.score(v.observed, v.incumbent),
              'original_gamma': f.r.score(v.observed, v.reference),
              'constant': f.r.score(v.observed, v.constant), 'folds': folds,
              'gain': f.r.bootstrap_gain(train, v.pred.to_numpy(), v.incumbent.to_numpy()),
              'frequency': {'selected': {'log_loss': log_loss(v.claim, v.proba_pred),
                                         'Brier': brier_score_loss(v.claim, v.proba_pred),
                                         'AUC': roc_auc_score(v.claim, v.proba_pred)},
                            'base': {'log_loss': log_loss(v.claim, v.original_probability),
                                     'Brier': brier_score_loss(v.claim, v.original_probability),
                                     'AUC': roc_auc_score(v.claim, v.original_probability)}},
              'severity_claims': f.r.score(v.loc[v.claim.eq(1), 'observed'], v.loc[v.claim.eq(1), 'montant_pred'])}
    write_json(REPORTS / 'diagnostic.json', result)
    per_repeat = pd.DataFrame(importance).groupby(['group', 'repetition']).agg(
        SSE_increase=('SSE_increase', 'sum'), rows=('rows', 'sum')).reset_index()
    per_repeat['MSE_increase'] = per_repeat.SSE_increase / per_repeat.rows
    per_repeat.to_csv(REPORTS / 'permutation_repeats.csv', index=False)
    table = per_repeat.groupby('group').MSE_increase.agg(['mean', 'std']).sort_values('mean', ascending=False)
    table.to_csv(REPORTS / 'permutation_importance.csv')
    print('Diagnostic terminé :', result['selected'], flush=True)
    return result


def verify_main():
    f = libraries()
    train, test, _, _ = f.r.load_data()
    model = load_model()
    f.r.valid_submission(ROOT / 'submission.csv', test)
    if (ROOT / 'submission.csv').read_bytes() != RELEASE_PATH.read_bytes():
        raise ValueError('La soumission principale ne correspond pas au fichier public retenu.')
    values = details(model, test)
    comparison = compare_to_release(values)
    if not comparison['numeric_match_rtol_1e_8_atol_1e_6']:
        raise ValueError('Le réentraînement diverge du CSV évalué : consulter les écarts avant utilisation.')
    probe = test.iloc[:20].copy()
    expected = model.details(probe).pred.to_numpy()
    for name in ['id_client', 'id_vehicule', 'id_contrat', 'nombre_sinistres', 'montant_sinistre',
                 'sex_conducteur1', 'sex_conducteur2']:
        probe[name] = '__NE_DOIT_PAS_ENTRER__'
    np.testing.assert_allclose(expected, model.details(probe).pred)
    for column in ['type_contrat', 'utilisation', 'marque_vehicule', 'modele_vehicule']:
        probe[column] = probe[column].astype(object)
        probe.loc[probe.index[0], column] = '__NOUVEAU__'
    probe.loc[probe.index[1], ['poids_vehicule', 'cylindre_vehicule', 'anciennete_vehicule']] = np.nan
    assert np.isfinite(model.details(probe).pred).all()
    f.r.load_data()  # Contrôle du notebook protégé et des cibles.
    report = {'status': 'passed', 'rows': len(test), 'release_link_verified': True,
              'component_identity_verified': True, 'forbidden_inputs_invariance': True,
              'unknown_categories_and_missing_values': True, 'protected_notebook_content_unchanged': True,
              'reproduction': comparison, 'model_sha256': sha256(MODEL_PATH)}
    write_json(REPORTS / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='info', choices=['info', 'train', 'diagnostic', 'predict', 'verify'])
    parser.add_argument('--output', type=Path, help='Fichier de prédictions recalculées ; le CSV soumis reste conservé.')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage == 'info':
            print(json.dumps(RECIPE, ensure_ascii=False, indent=2))
        elif args.stage == 'train':
            train_main()
        elif args.stage == 'diagnostic':
            diagnostic_main()
        elif args.stage == 'verify':
            verify_main()
        elif args.stage == 'predict':
            f = libraries()
            _, test, _, _ = f.r.load_data()
            destination = args.output or REPORTS / 'submission_reproduite.csv'
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = details(load_model(), test)
            result[['index', 'pred']].to_csv(destination, index=False)
            f.r.valid_submission(destination, test)
            print(destination)


if __name__ == '__main__':
    main()
