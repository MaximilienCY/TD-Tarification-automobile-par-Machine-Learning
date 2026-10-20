"""Cached grouped validation, selection and reproducible final fitting."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from .models import candidates, make_frequency, make_severity, PremiumModel

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports'
CACHE = ROOT / '.cache' / 'predictions'
PROTECTED = ROOT / "Calcul de la prime d'assurance.ipynb"
PROTECTED_HASH = '76248bdb51400fe399c670a9b71092cabe041cc13d3839b83846ac1499938429'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load_data():
    assert digest(PROTECTED) == PROTECTED_HASH, 'Protected notebook changed'
    train = pd.read_csv(ROOT/'Data/train.csv', dtype={'code_postal': str})
    test = pd.read_csv(ROOT/'Data/test.csv', dtype={'code_postal': str})
    assert train['index'].is_unique and test['index'].is_unique
    assert set(train['index']).isdisjoint(test['index'])
    assert set(train.nombre_sinistres.unique()) == {0, 1}
    assert ((train.nombre_sinistres == 0) == (train.montant_sinistre == 0)).all()
    assert train.montant_sinistre.ge(0).all()
    a,b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(train, groups=train.id_client))
    dev, historical = train.iloc[a].copy(), train.iloc[b].copy()
    assert set(dev.id_client).isdisjoint(historical.id_client)
    return train, test, dev, historical


def provenance():
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'lightgbm', 'joblib']
    return dict(data={n: digest(ROOT/'Data'/n) for n in ['train.csv','test.csv']},
                source={p.name: digest(p) for p in sorted((ROOT/'src/pricing').glob('*.py'))},
                versions={p:importlib.metadata.version(p) for p in packages}, python=platform.python_version())


def predict_component(learn, evaluation, config, frequency):
    component = 'frequency' if frequency else 'severity'
    fitting = learn if frequency else learn.loc[learn.nombre_sinistres.eq(1)]
    # Hash the entire input, groups, targets and source; IDs alone are insufficient.
    key = hashlib.sha256(json.dumps({'config':config, 'component':component,
                                    'provenance':provenance()}, sort_keys=True).encode())
    for frame in [fitting, evaluation]:
        key.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    path = CACHE / (key.hexdigest()+'.npy')
    if path.exists():
        values = np.load(path, allow_pickle=False)
    else:
        started = time.perf_counter()
        target = fitting.nombre_sinistres if frequency else fitting.montant_sinistre
        model = (make_frequency if frequency else make_severity)(config).fit(fitting, target, fitting.id_client)
        values = model.predict_proba(evaluation)[:,1] if frequency else model.predict(evaluation)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, values, allow_pickle=False)
        print(f'{component}: {config["name"]}, {len(fitting)} rows, {time.perf_counter()-started:.1f}s', flush=True)
    assert values.shape == (len(evaluation),) and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all() if frequency else (values > 0).all()
    return values


def oof_predictions(frame, fconfigs, gconfigs, folds=5, seed=20261009, label='search'):
    p = np.full((len(frame),len(fconfigs)), np.nan)
    s = np.full((len(frame),len(gconfigs)), np.nan)
    fold_id = np.full(len(frame), -1)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k,(a,b) in enumerate(cv.split(frame,frame.nombre_sinistres,groups=frame.id_client)):
        learn, evaluation = frame.iloc[a],frame.iloc[b]
        assert set(learn.id_client).isdisjoint(evaluation.id_client)
        for j,c in enumerate(fconfigs):
            p[b,j] = predict_component(learn,evaluation,c,True)
        for j,c in enumerate(gconfigs):
            s[b,j] = predict_component(learn,evaluation,c,False)
        fold_id[b] = k
        write_json(REPORTS/'progress.json', {'stage':label,'fold':k+1,'folds':folds})
        print(f'{label}: fold {k+1}/{folds} complete', flush=True)
    assert np.isfinite(p).all() and np.isfinite(s).all() and (fold_id >= 0).all()
    return p,s,fold_id


def combinations(matrix, configs, weights):
    columns, descriptions = [], []
    for i, c in enumerate(configs):
        for w in ([1.] if i == 0 else weights):
            vector = np.zeros(len(configs));vector[i] += w;vector[0] += 1-w
            columns.append(matrix @ vector)
            descriptions.append(dict(name=c['name'] if w == 1 else f'{w:g}*{c["name"]}+{1-w:g}*reference',
                                     weights=vector.tolist()))
    return np.column_stack(columns), descriptions


def select_pair(p,s,y,fconfigs,gconfigs):
    # Discrete prespecified blends, avoiding unconstrained weights on rare large losses.
    fp,fd = combinations(p,fconfigs,[.25,.5,.75,1.])
    gs,gd = combinations(s,gconfigs,[.5,1.])
    rows=[]
    for i, f in enumerate(fd):
        loss = np.mean((fp[:,i,None]*gs - np.asarray(y)[:,None])**2,axis=0)
        for j,g in enumerate(gd):
            rows.append(dict(frequency=f['name'],severity=g['name'],MSE=float(loss[j]),
                             RMSE=float(np.sqrt(loss[j])),fi=i,gi=j))
    ranking = pd.DataFrame(rows).sort_values(['MSE','frequency','severity']).reset_index(drop=True)
    r = ranking.iloc[0]
    choice = dict(frequency_weights=fd[int(r.fi)]['weights'],severity_weights=gd[int(r.gi)]['weights'],
                  frequency=r.frequency,severity=r.severity,MSE=float(r.MSE),RMSE=float(r.RMSE))
    return choice,ranking


def selected_prediction(p,s,choice):
    return (p @ choice['frequency_weights']) * (s @ choice['severity_weights'])


def old_optimized(p,s,fc,gc):
    j = next(i for i,c in enumerate(fc) if c['name']=='hist_previous')
    k = next(i for i,c in enumerate(gc) if c['name']=='gamma_basic_a0.01')
    return (.75*p[:,0]+.25*p[:,j])*s[:,k]


def metrics(y,p):
    mse = float(mean_squared_error(y,p))
    return dict(MSE=mse,RMSE=np.sqrt(mse),MAE=float(np.mean(np.abs(y-p))),
                bias=float(np.mean(p-y)),predicted_observed_ratio=float(np.sum(p)/np.sum(y)))


def bootstrap_gain(frame, new, reference, reps=2000):
    e = pd.DataFrame({'client':frame.id_client,'new':(frame.montant_sinistre-new)**2,
                      'reference':(frame.montant_sinistre-reference)**2,'n':1})
    sums = e.groupby('client')[['new','reference','n']].sum().to_numpy()
    rng=np.random.default_rng(701); values=[]
    for start in range(0,reps,25):
        z = sums[rng.integers(len(sums),size=(min(25,reps-start),len(sums)))].sum(axis=1)
        values.extend(np.sqrt(z[:,1]/z[:,2])-np.sqrt(z[:,0]/z[:,2]))
    return dict(gain_RMSE=metrics(frame.montant_sinistre,reference)['RMSE']-metrics(frame.montant_sinistre,new)['RMSE'],
                interval_95=np.quantile(values,[.025,.975]).tolist(),repetitions=reps,
                interpretation='Descriptive paired client bootstrap conditional on fitted predictions; not full training uncertainty.')


def search(dev,fc,gc):
    p,s,folds = oof_predictions(dev,fc,gc)
    choice,rank = select_pair(p,s,dev.montant_sinistre,fc,gc)
    REPORTS.mkdir(exist_ok=True)
    rank.to_csv(REPORTS/'search_pairs.csv',index=False)
    np.savez_compressed(REPORTS/'search_oof.npz',ids=dev['index'],p=p,s=s,fold=folds)
    write_json(REPORTS/'selection.json',dict(choice=choice,frequency_configs=fc,severity_configs=gc,
                                           provenance=provenance(),note='OOF scores used for selection are optimistic.'))
    scores=[]
    for j,c in enumerate(fc):
        scores.append(dict(model=c['name'],log_loss=log_loss(dev.nombre_sinistres,p[:,j]),
                           Brier=brier_score_loss(dev.nombre_sinistres,p[:,j]),AUC=roc_auc_score(dev.nombre_sinistres,p[:,j])))
    pd.DataFrame(scores).sort_values('log_loss').to_csv(REPORTS/'frequency_cv.csv',index=False)
    mask=dev.nombre_sinistres.eq(1)
    pd.DataFrame([dict(model=c['name'],**metrics(dev.loc[mask,'montant_sinistre'],s[mask,j]))
                  for j,c in enumerate(gc)]).sort_values('RMSE').to_csv(REPORTS/'severity_cv.csv',index=False)
    print(rank.head(10).to_string(index=False),flush=True)


def nested(dev,fc,gc):
    result=np.full(len(dev),np.nan); reference=result.copy(); previous=result.copy(); constant=result.copy()
    rows=[]
    outer=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261010)
    for k,(a,b) in enumerate(outer.split(dev,dev.nombre_sinistres,groups=dev.id_client),1):
        learn,evaluation=dev.iloc[a],dev.iloc[b]
        assert set(learn.id_client).isdisjoint(evaluation.id_client)
        p,s,_=oof_predictions(learn,fc,gc,folds=3,seed=3200+k,label=f'nested outer {k}')
        choice,_=select_pair(p,s,learn.montant_sinistre,fc,gc)
        # Fit only selected and benchmark components on outer training.
        fw=np.array(choice['frequency_weights']);gw=np.array(choice['severity_weights'])
        needed_f=set(np.flatnonzero(fw))|{0,next(i for i,c in enumerate(fc) if c['name']=='hist_previous')}
        needed_g=set(np.flatnonzero(gw))|{0,next(i for i,c in enumerate(gc) if c['name']=='gamma_basic_a0.01')}
        ep=np.zeros((len(evaluation),len(fc)));es=np.zeros((len(evaluation),len(gc)))
        for j in needed_f:ep[:,j]=predict_component(learn,evaluation,fc[j],True)
        for j in needed_g:es[:,j]=predict_component(learn,evaluation,gc[j],False)
        result[b]=selected_prediction(ep,es,choice)
        reference[b]=ep[:,0]*es[:,0];previous[b]=old_optimized(ep,es,fc,gc)
        constant[b]=learn.montant_sinistre.mean()
        rows.append(dict(fold=k,frequency=choice['frequency'],severity=choice['severity'],
                         **metrics(evaluation.montant_sinistre,result[b]),
                         reference_RMSE=metrics(evaluation.montant_sinistre,reference[b])['RMSE'],
                         previous_RMSE=metrics(evaluation.montant_sinistre,previous[b])['RMSE']))
        pd.DataFrame(rows).to_csv(REPORTS/'nested_folds.csv',index=False)
        print('NESTED',rows[-1],flush=True)
    df=pd.DataFrame({'index':dev['index'],'id_client':dev.id_client,'cost':dev.montant_sinistre,
                     'selected':result,'reference':reference,'previous':previous,'constant':constant})
    df.to_csv(REPORTS/'nested_predictions.csv',index=False)
    scores={name:metrics(dev.montant_sinistre,df[name]) for name in ['selected','reference','previous','constant']}
    write_json(REPORTS/'nested_validation.json',dict(scores=scores,
        gain_vs_reference=bootstrap_gain(dev,result,reference),gain_vs_previous=bootstrap_gain(dev,result,previous),
        caveat='Outer folds exclude their clients from all inner selection. Earlier human exploration of this development set remains a limitation.'))
    print(json.dumps(scores,indent=2),flush=True)


def finalize(train,test,dev,historical):
    saved=json.loads((REPORTS/'selection.json').read_text(encoding='utf-8'))
    if saved['provenance'] != provenance():
        raise ValueError('Source/environment/data changed after search; rerun search.')
    fc,gc,choice=saved['frequency_configs'],saved['severity_configs'],saved['choice']
    model=PremiumModel(fc,gc,choice['frequency_weights'],choice['severity_weights'])
    model.fit(dev)
    detail=model.predict_details(historical)
    validation=metrics(historical.montant_sinistre,detail.pred)
    validation.update(frequency_log_loss=log_loss(historical.nombre_sinistres,detail.proba_pred),
                      frequency_Brier=brier_score_loss(historical.nombre_sinistres,detail.proba_pred),
                      frequency_AUC=roc_auc_score(historical.nombre_sinistres,detail.proba_pred))
    write_json(REPORTS/'historical_diagnostic.json',dict(scores=validation,
        note='Previously examined holdout; not used for model/weight selection in this run.'))
    detail.insert(0,'index',historical['index'].to_numpy())
    detail['cost']=historical.montant_sinistre.to_numpy()
    detail['claim']=historical.nombre_sinistres.to_numpy()
    detail.to_csv(REPORTS/'historical_predictions.csv',index=False)
    model.fit(train)
    details=model.predict_details(test)
    details.insert(0,'index',test['index'].to_numpy())
    submission=details[['index','pred']]
    validate_submission(submission,test)
    submission.to_csv(ROOT/'submission.csv',index=False)
    details.to_csv(REPORTS/'test_components.csv',index=False)
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'models/premium.joblib',compress=3)
    reloaded=joblib.load(ROOT/'models/premium.joblib')
    np.testing.assert_allclose(reloaded.predict(test.iloc[:50]),details.pred.iloc[:50],rtol=1e-12)
    write_json(REPORTS/'final_model.json',dict(selection=saved,training_rows=len(train),
        severity_rows=int(train.nombre_sinistres.sum()),submission_rows=len(submission),
        submission_sha256=digest(ROOT/'submission.csv'),protected_notebook_sha256=digest(PROTECTED),
        historical_diagnostic=validation,kaggle_score=None))
    print('Final submission:',ROOT/'submission.csv',validation,flush=True)


def validate_submission(submission,test):
    assert list(submission.columns)==['index','pred']
    assert len(submission)==len(test) and submission['index'].is_unique
    np.testing.assert_array_equal(submission['index'],test['index'])
    assert np.isfinite(submission.pred).all() and submission.pred.ge(0).all()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['search','nested','finalize','all','predict'])
    args=parser.parse_args()
    threadpool_limits(4)
    warnings.filterwarnings('error',category=ConvergenceWarning)
    train,test,dev,historical=load_data()
    fc,gc=candidates()
    REPORTS.mkdir(exist_ok=True)
    if args.stage in ('search','all'):search(dev,fc,gc)
    if args.stage in ('nested','all'):nested(dev,fc,gc)
    if args.stage in ('finalize','all'):finalize(train,test,dev,historical)
    if args.stage=='predict':
        model=joblib.load(ROOT/'models/premium.joblib')
        sub=pd.DataFrame({'index':test['index'],'pred':model.predict(test)})
        validate_submission(sub,test)
        sub.to_csv(ROOT/'submission.csv',index=False)
