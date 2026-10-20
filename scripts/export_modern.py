"""Fit/export the shortlisted EBM/XGBoost candidates, or reproduce their predictions."""
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import advanced_search as engine
from modern_search import ModernCandidate


def predict(bundle,X):
    d=bundle['incumbent'].predict_details(X)
    p=d.proba_pred.to_numpy(copy=True);s=d.montant_pred.to_numpy(copy=True)
    for key,base in [('frequency',p),('severity',s)]:
        w=bundle['choice'][key]['weight']
        if w:base[:]=(1-w)*base+w*bundle[key].predict(X)
    assert np.isfinite(p*s).all() and (p>=0).all() and (p<=1).all() and (s>0).all()
    return pd.DataFrame({'index':X['index'].to_numpy(),'proba_pred':p,'montant_pred':s,'pred':p*s})


def main():
    train,test,dev,hist=engine.load_data()
    out=engine.ROOT/'reports/modern';folder=engine.ROOT/'deliverables/candidates'
    folder.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1 and sys.argv[1]=='predict':
        for path in sorted((engine.ROOT/'models').glob('modern_*.joblib')):
            d=predict(joblib.load(path),test)
            d[['index','pred']].to_csv(folder/f'submission_{path.stem}.csv',index=False)
        return
    selection=json.loads((out/'selection.json').read_text())
    confirmation=json.loads((out/'confirmation.json').read_text())
    engine.Candidate=ModernCandidate;engine.CACHE=engine.ROOT/'.cache/modern'
    baseline=joblib.load(engine.ROOT/'models/premium.joblib')
    p0h,s0h=engine.incumbent(dev,hist)
    entries=[]
    sample=pd.read_csv(engine.ROOT/'Data/prime_pred_sandbox.csv')
    assert np.array_equal(sample['index'],test['index'])
    for j,ch in enumerate(selection['choices'],1):
        bundle=dict(incumbent=baseline,choice=ch)
        for key in ['frequency','severity']:
            frequency=key=='frequency';subset=train if frequency else train.loc[train.nombre_sinistres.eq(1)]
            y=subset.nombre_sinistres if frequency else subset.montant_sinistre
            bundle[key]=ModernCandidate(ch[key]['config'],frequency).fit(subset,y) if ch[key]['weight'] else None
        d=predict(bundle,test);path=engine.ROOT/f'models/modern_{j}.joblib'
        joblib.dump(bundle,path)
        np.testing.assert_array_equal(predict(joblib.load(path),test.iloc[:100])['pred'],d['pred'].iloc[:100])
        filename=folder/f'submission_modern_{j}.csv'
        d[['index','pred']].to_csv(filename,index=False)
        d.to_csv(engine.CACHE/f'test_components_{j}.csv',index=False)
        ph,sh=engine.choice_components(dev,hist,ch,p0h,s0h)
        entries.append(dict(candidate=j,file=str(filename.relative_to(engine.ROOT)),sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            choice=ch,historical_diagnostic=engine.metrics(hist.montant_sinistre,ph*sh),kaggle_score=None))
    engine.write_json(out/'exports.json',dict(entries=entries,confirmation=confirmation,
        incumbent_historical=engine.metrics(hist.montant_sinistre,p0h*s0h),provenance=engine.provenance(),
        submission_status='Blocked by saved Kaggle browser permission. No new Kaggle score observed.'))
    print(json.dumps(entries,indent=2),flush=True)


if __name__=='__main__':
    with engine.threadpool_limits(limits=4):main()
