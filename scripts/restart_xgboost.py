"""Client-separated XGBoost restart. Run with --help; no original artifacts are replaced."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor
from pricing.reference import RAW_FEATURES, NUM_COMPACT, LOW_CATS, ReferenceFrequence, GammaMetier
from pricing.experiment import load_data, digest, write_json, provenance, bootstrap_gain

OUT = ROOT / 'reports/restart_xgboost'
CACHE = ROOT / '.cache/restart_xgboost'
SEED = 20261012
WEIGHTS = [0., .25, .5, .75, 1.]
MEANINGS = dict(zip(RAW_FEATURES, [
    'Bonus-malus coefficient', 'Coverage level', 'Contract duration; unit unconfirmed',
    'Information tenure; unit unconfirmed', 'Payment frequency', 'Payment method', 'Vehicle usage',
    'Postal code', 'Presence of second driver', 'Main driver age', 'Second driver age (zero when absent)',
    'Main driver licence tenure', 'Second driver licence tenure (zero when absent)',
    'Vehicle age', 'Engine displacement', 'Engine power; unit unconfirmed', 'Fuel type', 'Vehicle make',
    'Vehicle model', 'Sale-start variable; reference date unconfirmed',
    'Sale-end variable; reference date unconfirmed', 'Vehicle speed specification', 'Vehicle type',
    'Vehicle price', 'Vehicle weight']))
MEANINGS.update(index='Submission row identifier', id_client='Client grouping identifier',
    id_vehicule='Vehicle identifier', id_contrat='Contract identifier',
    sex_conducteur1='Main driver sex; excluded', sex_conducteur2='Second driver sex; excluded',
    nombre_sinistres='Binary claim occurrence target', montant_sinistre='Observed cost target including zeros')


def study(train, test):
    OUT.mkdir(parents=True, exist_ok=True)
    profile, groups = [], []
    for c in train:
        s = train[c]; t = test[c] if c in test else None
        numeric = pd.api.types.is_numeric_dtype(s)
        record = dict(variable=c, meaning=MEANINGS.get(c, 'Undocumented'),
            role='predictor' if c in RAW_FEATURES else 'excluded / target / grouping',
            train_missing=int(s.isna().sum()), train_unique=int(s.nunique()),
            test_missing=int(t.isna().sum()) if t is not None else None,
            test_unseen=int((~t.isin(s.dropna()) & t.notna()).sum()) if t is not None and not numeric else None,
            minimum=float(s.min()) if numeric else None, maximum=float(s.max()) if numeric else None,
            train_test_KS=float(ks_2samp(s.dropna(), t.dropna()).statistic) if numeric and t is not None else None)
        profile.append(record)
        if c not in RAW_FEATURES:
            continue
        if numeric and s.nunique() > 12:
            bins = pd.qcut(s, 10, duplicates='drop').astype('string').fillna('__MISSING__')
        else:
            counts = s.value_counts(); keep = counts.head(30).index
            bins = s.astype('string').where(s.isin(keep), '__OTHER__').fillna('__MISSING__')
        v = train.assign(level=bins)
        for level, z in v.groupby('level', observed=True):
            positive = z.loc[z.nombre_sinistres.eq(1), 'montant_sinistre']
            groups.append(dict(variable=c, level=str(level), rows=len(z), claims=len(positive),
                claim_rate=float(z.nombre_sinistres.mean()), mean_cost=float(z.montant_sinistre.mean()),
                conditional_severity=float(positive.mean()) if len(positive) else None))
    pd.DataFrame(profile).to_csv(OUT / 'variable_dictionary.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT / 'variable_groups.csv', index=False)
    anomalies = dict(zero_weight=int(train.poids_vehicule.eq(0).sum()),
        zero_engine=int(train.cylindre_vehicule.eq(0).sum()),
        licence_exceeds_age1=int(train.anciennete_permis1.gt(train.age_conducteur1).sum()),
        licence_exceeds_age2=int((train.anciennete_permis2.gt(train.age_conducteur2) & train.age_conducteur2.gt(0)).sum()),
        largest_25_share_squared_cost=float(train.montant_sinistre.nlargest(25).pow(2).sum()/train.montant_sinistre.pow(2).sum()))
    write_json(OUT / 'data_audit.json', dict(train_rows=len(train), test_rows=len(test),
        clients=int(train.id_client.nunique()), claims=int(train.nombre_sinistres.sum()), anomalies=anomalies,
        note='Associations are descriptive, not causal. Duration/exposure and sale-date semantics unconfirmed.'))
    print('Variable audit:', anomalies, flush=True)


def features(X, feature_set, cleaned):
    """Stateless transformations only; learned categories live in each fitted model."""
    x = X[RAW_FEATURES].copy()
    second = x.conducteur2.eq('Yes')
    x['second_driver_present'] = second.astype(float)
    for col in ['poids_vehicule', 'cylindre_vehicule']:
        x[col + '_invalid'] = x[col].le(0).astype(float)
        if cleaned:
            x[col] = x[col].mask(x[col].le(0))
    for i in [1, 2]:
        c, age = f'anciennete_permis{i}', f'age_conducteur{i}'
        invalid = x[c].gt(x[age]) | x[c].lt(0)
        x[c + '_invalid'] = invalid.astype(float)
        if cleaned:
            x[c] = x[c].mask(invalid)
            if i == 2:
                x[c] = x[c].mask(~second)
                x[age] = x[age].mask(~second)
    for dest, src in [('log_bonus', 'bonus'), ('log_din', 'din_vehicule'),
                      ('log_poids', 'poids_vehicule'), ('log_cylindre', 'cylindre_vehicule')]:
        x[dest] = np.log1p(x[src].mask(x[src].lt(0)))
    flags = [c for c in x if c.endswith('_invalid')] + ['second_driver_present']
    if feature_set == 'compact':
        x = x[NUM_COMPACT + LOW_CATS + flags]
    elif feature_set == 'raw':
        x = x[RAW_FEATURES + flags]
    elif feature_set == 'engineered':
        s = x.code_postal.astype('string').str.strip().str.upper()
        valid = s.str.fullmatch(r'(?:0[1-9]|1[0-9]|2[1-9]|[3-8][0-9]|9[0-5]|2A|2B)[0-9]{3}', na=False)
        x['departement'] = s.str[:2].where(valid)
        dom = s.str.fullmatch(r'97[1-46][0-9]{2}', na=False)
        x.loc[dom, 'departement'] = s.loc[dom].str[:3]
        x['power_weight'] = x.din_vehicule / (x.poids_vehicule.where(x.poids_vehicule.gt(0)) / 1000)
        x['contract_usage'] = x.type_contrat.astype('string') + '|' + x.utilisation.astype('string')
        x['vehicle_age_maxi'] = x.anciennete_vehicule * x.type_contrat.eq('Maxi')
        x['log_price'] = np.log1p(x.prix_vehicule.clip(lower=0))
    else:
        raise ValueError(feature_set)
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
    return x


def catalogue(frequency, trials=60):
    rng = np.random.default_rng(SEED + (0 if frequency else 1))
    configs = []
    for i in range(trials):
        configs.append(dict(trial=i, feature_set=['compact', 'raw', 'engineered'][i % 3],
            cleaned=bool((i // 3) % 2), objective='binary:logistic' if frequency else ['reg:gamma', 'reg:squarederror'][(i // 6) % 2],
            max_depth=int(rng.integers(1, 6)), learning_rate=float(np.exp(rng.uniform(np.log(.01), np.log(.08)))),
            subsample=float(rng.uniform(.7, 1)), colsample_bytree=float(rng.uniform(.7, 1)),
            min_child_weight=float(np.exp(rng.uniform(np.log(1 if frequency else 5), np.log(100 if frequency else 200)))),
            reg_lambda=float(np.exp(rng.uniform(np.log(.1), np.log(300)))),
            reg_alpha=float(rng.choice([0., .01, .1, 1., 10.])), max_cat_to_onehot=8, max_cat_threshold=32))
    return configs


def split_groups(frame, folds, seed):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for a, b in cv.split(frame, frame.nombre_sinistres, frame.id_client):
        assert set(frame.iloc[a].id_client).isdisjoint(frame.iloc[b].id_client)
        yield a, b


class Component:
    def __init__(self, config, frequency, rounds=None):
        self.config, self.frequency, self.rounds = config, frequency, rounds

    def encode(self, X, fit=False):
        x = features(X, self.config['feature_set'], self.config['cleaned'])
        if fit:
            self.categories_ = {c: sorted(x[c].unique()) for c in x.select_dtypes(exclude='number')}
        for c, values in self.categories_.items():
            x[c] = pd.Categorical(x[c].where(x[c].isin(values)), categories=values)
        return x

    def fit(self, frame):
        # The early-stopping client holdout belongs entirely to this fit's training partition.
        if self.rounds is None:
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(frame, groups=frame.id_client))
            assert set(frame.iloc[a].id_client).isdisjoint(frame.iloc[b].id_client)
            learn, valid = frame.iloc[a], frame.iloc[b]
            if not self.frequency:
                learn = learn.loc[learn.nombre_sinistres.eq(1)]
                valid = valid.loc[valid.nombre_sinistres.eq(1)]
            self._fit(learn, valid, 3000)
            self.rounds_ = self.model_.best_iteration + 1
        else:
            self.rounds_ = int(self.rounds)
        learn = frame if self.frequency else frame.loc[frame.nombre_sinistres.eq(1)]
        self._fit(learn, None, self.rounds_)
        return self

    def _fit(self, learn, valid, rounds):
        self.scale_ = 1. if self.frequency else float(learn.montant_sinistre.mean())
        target = 'nombre_sinistres' if self.frequency else 'montant_sinistre'
        params = {k: v for k, v in self.config.items() if k not in ['trial', 'feature_set', 'cleaned']}
        params.update(n_estimators=rounds, tree_method='hist', enable_categorical=True,
            n_jobs=4, random_state=SEED, eval_metric='logloss' if self.frequency else 'rmse')
        if valid is not None:
            params['early_stopping_rounds'] = 80
        self.model_ = (XGBClassifier if self.frequency else XGBRegressor)(**params)
        x = self.encode(learn, fit=True)
        kwargs = {} if valid is None else dict(eval_set=[(self.encode(valid), valid[target].to_numpy()/self.scale_)], verbose=False)
        self.model_.fit(x, learn[target].to_numpy()/self.scale_, **kwargs)

    def predict(self, X):
        x = self.encode(X)
        values = self.model_.predict_proba(x)[:, 1] if self.frequency else self.model_.predict(x)*self.scale_
        values = np.clip(values, 1e-8, 1-1e-8) if self.frequency else np.maximum(values, 1e-8)
        assert np.isfinite(values).all()
        return values


def reference_components(learn, evaluation):
    f = ReferenceFrequence().fit(learn, learn.nombre_sinistres, learn.id_client)
    claims = learn.loc[learn.nombre_sinistres.eq(1)]
    s = GammaMetier({'alpha': .1}).fit(claims, claims.montant_sinistre, claims.id_client)
    return f.predict_proba(evaluation)[:, 1], s.predict(evaluation)


def cache_key(learn, evaluation, config):
    import importlib.metadata
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(json.dumps(dict(config=config, source=provenance(), xgboost=importlib.metadata.version('xgboost')), sort_keys=True).encode())
    for x in [learn, evaluation]:
        h.update(pd.util.hash_pandas_object(x, index=True).to_numpy().tobytes())
    return h.hexdigest()


def cached_component(learn, evaluation, config, frequency):
    path = CACHE / (cache_key(learn, evaluation, dict(config=config, frequency=frequency)) + '.npz')
    if path.exists():
        with np.load(path) as z:
            return z['prediction'], int(z['rounds'])
    m = Component(config, frequency).fit(learn)
    prediction = m.predict(evaluation)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=prediction, rounds=m.rounds_)
    return prediction, m.rounds_


def calibration(p, y):
    return LogisticRegression(C=1., max_iter=1000).fit(logit(np.clip(p, 1e-8, 1-1e-8)).reshape(-1, 1), y)


def calibrate(model, p):
    return model.predict_proba(logit(np.clip(p, 1e-8, 1-1e-8)).reshape(-1, 1))[:, 1]


def score(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    mse = float(np.mean((y-pred)**2))
    return dict(MSE=mse, RMSE=float(np.sqrt(mse)), MAE=float(np.mean(np.abs(y-pred))),
        bias=float(np.mean(pred-y)), predicted_observed_ratio=float(pred.sum()/y.sum()))


def selection(frame, label, trials=60):
    folder = OUT / label; folder.mkdir(parents=True, exist_ok=True)
    fc, sc = catalogue(True, trials), catalogue(False, trials)
    fp, sp = np.zeros((len(frame), trials)), np.zeros((len(frame), trials))
    refp, refs = np.zeros(len(frame)), np.zeros(len(frame))
    folds = np.zeros(len(frame), dtype=int)
    rounds_f, rounds_s = np.zeros((5, trials), dtype=int), np.zeros((5, trials), dtype=int)
    for fold, (a, b) in enumerate(split_groups(frame, 5, SEED + 11)):
        learn, evaluation = frame.iloc[a], frame.iloc[b]; folds[b] = fold
        key = CACHE / (cache_key(learn, evaluation, 'reference') + '.npz')
        if key.exists():
            with np.load(key) as z: refp[b], refs[b] = z['p'], z['s']
        else:
            refp[b], refs[b] = reference_components(learn, evaluation)
            CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(key, p=refp[b], s=refs[b])
        started = time.perf_counter()
        for frequency, configs, predictions, counts in [(True, fc, fp, rounds_f), (False, sc, sp, rounds_s)]:
            for j, c in enumerate(configs):
                predictions[b, j], counts[fold, j] = cached_component(learn, evaluation, c, frequency)
                if (j+1) % 10 == 0:
                    print(f'{label} fold {fold+1}/5 {"frequency" if frequency else "severity"} {j+1}/{trials}; {time.perf_counter()-started:.0f}s', flush=True)
    y = frame.nombre_sinistres.to_numpy(); cost = frame.montant_sinistre.to_numpy(); mask = y == 1
    fs = np.array([log_loss(y, fp[:, j]) for j in range(trials)])
    ss = np.array([mean_squared_error(cost[mask], sp[mask, j]) for j in range(trials)])
    topf, tops = np.argsort(fs)[:5], np.argsort(ss)[:5]
    pd.DataFrame([dict(**c, log_loss=float(fs[j]), Brier=float(brier_score_loss(y, fp[:, j])), median_trees=int(np.median(rounds_f[:, j]))) for j, c in enumerate(fc)]).sort_values('log_loss').to_csv(folder/'frequency_trials.csv', index=False)
    pd.DataFrame([dict(**c, RMSE=float(np.sqrt(ss[j])), median_trees=int(np.median(rounds_s[:, j]))) for j, c in enumerate(sc)]).sort_values('RMSE').to_csv(folder/'severity_trials.csv', index=False)
    # Cross-fit calibration on inner OOF predictions. Outer evaluation is never used.
    candidates_f = [(dict(kind='reference'), refp)]
    for j in topf:
        candidates_f.append((dict(kind='xgb', trial=int(j), calibrated=False), fp[:, j]))
        cp = np.zeros(len(frame))
        for fold in range(5):
            a, b = folds != fold, folds == fold
            cp[b] = calibrate(calibration(fp[a, j], y[a]), fp[b, j])
        candidates_f.append((dict(kind='xgb', trial=int(j), calibrated=True), cp))
    candidates_s = [(dict(kind='reference'), refs)] + [(dict(kind='xgb', trial=int(j)), sp[:, j]) for j in tops]
    # Deduplicate zero weights to avoid ranking repeated copies of the reference.
    choices_f = [(dict(kind='reference', weight=0.), refp)]
    for spec, p in candidates_f[1:]:
        choices_f.extend((dict(**spec, weight=w), w*p+(1-w)*refp) for w in WEIGHTS[1:])
    choices_s = [(dict(kind='reference', weight=0.), refs)]
    for spec, s in candidates_s[1:]:
        choices_s.extend((dict(**spec, weight=w), w*s+(1-w)*refs) for w in WEIGHTS[1:])
    rankings = []
    for fi, (f, p) in enumerate(choices_f):
        for si, (s, v) in enumerate(choices_s):
            rankings.append(dict(fi=fi, si=si, RMSE=float(np.sqrt(np.mean((cost-p*v)**2)))))
    ranking = pd.DataFrame(rankings).sort_values(['RMSE', 'fi', 'si'])
    best = ranking.iloc[0]; f, _ = choices_f[int(best.fi)]; s, _ = choices_s[int(best.si)]
    selected = dict(frequency=f, severity=s, selection_RMSE=float(best.RMSE), reference_RMSE=score(cost, refp*refs)['RMSE'])
    ranking.to_csv(folder/'pair_ranking.csv', index=False)
    write_json(folder/'selection.json', dict(**selected, frequency_catalogue=fc, severity_catalogue=sc,
        note='Inner selection scores are optimistic. Cross-fitted calibration uses inner OOF predictions; only outer folds evaluate the complete selection procedure.'))
    np.savez_compressed(folder/'oof.npz', p=fp, s=sp, refp=refp, refs=refs, folds=folds, ids=frame['index'].to_numpy())
    # Calibration deployed later is learned on all training-only OOF predictions.
    cal = calibration(fp[:, f['trial']], y) if f.get('calibrated') else None
    return selected, cal


class Premium:
    def __init__(self, choice, calibrator=None, trials=60):
        self.choice, self.calibrator, self.trials = choice, calibrator, trials

    def fit(self, learn):
        self.reference_f = ReferenceFrequence().fit(learn, learn.nombre_sinistres, learn.id_client)
        claims = learn.loc[learn.nombre_sinistres.eq(1)]
        self.reference_s = GammaMetier({'alpha': .1}).fit(claims, claims.montant_sinistre, claims.id_client)
        for name, frequency in [('frequency', True), ('severity', False)]:
            spec = self.choice[name]
            if spec['kind'] == 'xgb':
                c = catalogue(frequency, self.trials)[spec['trial']]
                setattr(self, name+'_model', Component(c, frequency).fit(learn))
        return self

    def details(self, X):
        rp = self.reference_f.predict_proba(X)[:, 1]; rs = self.reference_s.predict(X)
        p, s = rp.copy(), rs.copy()
        for name in ['frequency', 'severity']:
            spec = self.choice[name]
            if spec['kind'] == 'xgb':
                v = getattr(self, name+'_model').predict(X)
                if name == 'frequency':
                    if spec.get('calibrated'): v = calibrate(self.calibrator, v)
                    p = spec['weight']*v+(1-spec['weight'])*rp
                else: s = spec['weight']*v+(1-spec['weight'])*rs
        assert ((p >= 0) & (p <= 1)).all() and (s > 0).all() and np.isfinite(p*s).all()
        return pd.DataFrame(dict(index=X['index'].to_numpy(), proba_pred=p, montant_pred=s, pred=p*s, reference=rp*rs))


def nested(train, trials=60):
    parts, rows = [], []
    for fold, (a, b) in enumerate(split_groups(train, 3, SEED + 23), 1):
        learn, evaluation = train.iloc[a], train.iloc[b]
        choice, cal = selection(learn, f'outer_{fold}', trials)
        m = Premium(choice, cal, trials).fit(learn); v = m.details(evaluation)
        v['client'] = evaluation.id_client.to_numpy(); v['claim'] = evaluation.nombre_sinistres.to_numpy()
        v['observed'] = evaluation.montant_sinistre.to_numpy(); v['type_contrat'] = evaluation.type_contrat.to_numpy(); v['fold'] = fold
        parts.append(v); rows.append(dict(fold=fold, rows=len(v), choice=choice,
            selected=score(v.observed, v.pred), reference=score(v.observed, v.reference)))
        write_json(OUT/f'outer_{fold}/evaluation.json', rows[-1])
        print('Outer evaluation', rows[-1], flush=True)
    v = pd.concat(parts).set_index('index').loc[train['index']].reset_index()
    v.to_csv(OUT/'nested_predictions.csv', index=False)
    gain = bootstrap_gain(train.reset_index(drop=True), v.pred.to_numpy(), v.reference.to_numpy())
    result = dict(selected=score(v.observed, v.pred), reference=score(v.observed, v.reference), gain=gain,
        frequency=dict(log_loss=float(log_loss(v.claim, v.proba_pred)), Brier=float(brier_score_loss(v.claim, v.proba_pred)), AUC=float(roc_auc_score(v.claim, v.proba_pred))),
        severity=score(v.loc[v.claim.eq(1), 'observed'], v.loc[v.claim.eq(1), 'montant_pred']), folds=rows,
        stable_improvement=bool(gain['interval_95'][0] > 0),
        caveat='All data have been explored previously. Nested evaluation protects this search from outer label leakage but does not erase prior researcher adaptation.')
    write_json(OUT/'nested_results.json', result)
    v.assign(bin=pd.qcut(v.proba_pred, 10, duplicates='drop')).groupby('bin', observed=True).agg(n=('claim','size'), predicted=('proba_pred','mean'), observed=('claim','mean')).to_csv(OUT/'calibration.csv')
    pd.DataFrame([dict(type_contrat=c, n=len(z), **score(z.observed, z.pred), reference_RMSE=score(z.observed,z.reference)['RMSE']) for c,z in v.groupby('type_contrat')]).to_csv(OUT/'contract_performance.csv', index=False)
    return result


def finalize(train, test, results, trials=60):
    choice, cal = selection(train, 'full_training', trials)
    m = Premium(choice, cal, trials).fit(train)
    model_path = ROOT/'models/restart_xgboost.joblib'; model_path.parent.mkdir(exist_ok=True)
    joblib.dump(m, model_path)
    v = joblib.load(model_path).details(test)
    candidate = ROOT/'deliverables/candidates/submission_restart_xgboost.csv'
    v[['index','pred']].to_csv(candidate, index=False)
    v.to_csv(OUT/'test_components.csv', index=False)
    reference = ROOT/'deliverables/candidates/submission_reference.csv'
    old = pd.read_csv(reference)
    assert np.array_equal(v['index'], old['index'])
    reproduction = float(np.max(np.abs(v.reference.to_numpy()-old.pred.to_numpy())))
    assert reproduction < 1e-6
    valid_submission(candidate, test)
    write_json(OUT/'final_model.json', dict(choice=choice, challenger=str(candidate.relative_to(ROOT)),
        challenger_sha256=digest(candidate), model_sha256=digest(model_path), reference_reproduction_max_abs=reproduction,
        production_recommendation='challenger' if results['stable_improvement'] else 'original reference',
        target=1365.500, kaggle_score=None, submission_state='saved Kaggle browser permission unresolved',
        protected_notebook_sha256=digest(ROOT/"Calcul de la prime d'assurance.ipynb"), provenance=provenance()))
    print('Final candidate', candidate, 'SHA256', digest(candidate), flush=True)


def valid_submission(path, test):
    x = pd.read_csv(path)
    assert list(x.columns) == ['index','pred'] and len(x) == len(test)
    assert x['index'].is_unique and np.array_equal(x['index'], test['index'])
    assert np.isfinite(x.pred).all() and x.pred.ge(0).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit','train','all','predict'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    train, test, _, _ = load_data(); OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage in ['audit', 'all']: study(train, test)
        if args.stage in ['train', 'all']:
            result = nested(train)
            finalize(train, test, result)
        if args.stage == 'predict':
            if args.output is None: parser.error('predict requires --output')
            model = joblib.load(ROOT/'models/restart_xgboost.joblib')
            model.details(test)[['index','pred']].to_csv(args.output, index=False)
            valid_submission(args.output, test)


if __name__ == '__main__':
    # Stable pickle module, including prediction in a separate Python process.
    from restart_xgboost import main
    main()
