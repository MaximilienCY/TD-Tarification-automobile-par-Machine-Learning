"""Local GPU TabPFN-v2 severity benchmark (separate environment).

Built with TabPFN (Prior Labs). Uses only positive claims for severity and the
frozen incumbent's out-of-fold frequency probabilities. No cloud inference.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ['TABPFN_MODEL_CACHE_DIR']=str(ROOT/'.cache/tabpfn-models')
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import json
import hashlib
import importlib.metadata
import sys
import gc
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
OUT=ROOT/'reports/foundation'
CACHE=ROOT/'.cache/foundation'
WEIGHTS=ROOT/'.cache/tabpfn-models/tabpfn-v2-regressor.ckpt'
VERSIONS={k:importlib.metadata.version(k) for k in ['torch','tabpfn','numpy','pandas','scikit-learn']}
FEATURES={
    'core':['anciennete_vehicule','type_contrat','utilisation'],
    'compact':['anciennete_vehicule','type_contrat','utilisation','prix_vehicule',
               'age_conducteur1','bonus','din_vehicule','conducteur2'],
}


def prediction(learn,evaluation,features):
    key=hashlib.sha256(Path(__file__).read_bytes())
    key.update(json.dumps(VERSIONS,sort_keys=True).encode())
    key.update(hashlib.sha256(WEIGHTS.read_bytes()).digest())
    for frame in [learn,evaluation]:key.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    key.update(json.dumps(features).encode())
    filename=CACHE/(key.hexdigest()+'.npy')
    if filename.exists():return np.load(filename)
    x=learn[features].copy();z=evaluation[features].copy()
    cats=[]
    for i,col in enumerate(features):
        if not pd.api.types.is_numeric_dtype(x[col]):
            cats.append(i);labels={v:j for j,v in enumerate(sorted(x[col].dropna().unique()))}
            x[col]=x[col].map(labels).astype(float);z[col]=z[col].map(labels).astype(float)
    m=TabPFNRegressor(n_estimators=4,device='cuda',random_state=42,n_jobs=4,
        categorical_features_indices=cats,memory_saving_mode=True,model_path=WEIGHTS)
    m.fit(x,learn.montant_sinistre/1000)
    values=[]
    for start in range(0,len(z),1024):
        values.append(m.predict(z.iloc[start:start+1024])*1000)
    p=np.maximum(np.concatenate(values),1e-8)
    assert np.isfinite(p).all()
    np.save(filename,p)
    del m;gc.collect();torch.cuda.empty_cache()
    return p


def main():
    assert torch.cuda.is_available(),'CUDA required for this benchmark'
    assert hashlib.sha256((ROOT/"Calcul de la prime d'assurance.ipynb").read_bytes()).hexdigest()=='76248bdb51400fe399c670a9b71092cabe041cc13d3839b83846ac1499938429'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'Data/train.csv',dtype={'code_postal':str})
    old=np.load(ROOT/'reports/search_oof.npz');meta=json.loads((ROOT/'reports/selection.json').read_text())
    assert hashlib.sha256((ROOT/'Data/train.csv').read_bytes()).hexdigest()==meta['provenance']['data']['train.csv']
    dev=train.set_index('index',drop=False).loc[old['ids']]
    assert np.array_equal(dev['index'],old['ids'])
    p0=old['p']@meta['choice']['frequency_weights'];s0=old['s']@meta['choice']['severity_weights']
    y=dev.montant_sinistre.to_numpy();rows=[]
    for label,features in FEATURES.items():
        severity=np.zeros(len(dev))
        for fold in range(5):
            a=np.flatnonzero(old['fold']!=fold);b=np.flatnonzero(old['fold']==fold)
            learn=dev.iloc[a];learn=learn.loc[learn.nombre_sinistres.eq(1)]
            assert set(learn.id_client).isdisjoint(dev.iloc[b].id_client)
            severity[b]=prediction(learn,dev.iloc[b],features)
            print(f'{label} fold {fold+1}/5 complete',flush=True)
        np.save(CACHE/f'oof_{label}.npy',severity)
        for w in [.25,.5,.75,1.]:
            pred=p0*((1-w)*s0+w*severity)
            rows.append(dict(features=label,weight=w,RMSE=float(np.sqrt(np.mean((y-pred)**2)))))
    ranking=pd.DataFrame(rows).sort_values('RMSE')
    ranking.to_csv(OUT/'ranking.csv',index=False)
    report=dict(baseline_RMSE=float(np.sqrt(np.mean((y-p0*s0)**2))),results=ranking.to_dict('records'),
        model='TabPFN-v2 regressor, 4 estimators, local CUDA',
        versions=VERSIONS,weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Selection on reused development data; not independent validation. Built with TabPFN by Prior Labs.')
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
