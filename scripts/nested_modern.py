"""Repeat modern-model selection inside every outer client fold.

The incumbent recipe is fixed from earlier development. This accounts for the
new 525-pair search, not all historical human exploration of the dataset.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import advanced_search as engine
from modern_search import ModernCandidate,catalogue
from pricing.experiment import predict_component


def baseline(learn,evaluation):
    meta=json.loads((engine.ROOT/'reports/selection.json').read_text())
    vals=[]
    for frequency,key in [(True,'frequency'),(False,'severity')]:
        v=np.zeros(len(evaluation))
        for c,w in zip(meta[key+'_configs'],meta['choice'][key+'_weights']):
            if w:v+=w*predict_component(learn,evaluation,c,frequency)
        vals.append(v)
    return vals


def main():
    _,_,dev,_=engine.load_data();fc,gc=catalogue()
    engine.Candidate=ModernCandidate;engine.CACHE=engine.ROOT/'.cache/modern'
    out=engine.ROOT/'reports/modern'
    outer=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261010)
    result=np.zeros(len(dev));reference=np.zeros(len(dev));rows=[]
    for k,(a,b) in enumerate(outer.split(dev,dev.nombre_sinistres,dev.id_client),1):
        learn,evaluation=dev.iloc[a],dev.iloc[b]
        assert set(learn.id_client).isdisjoint(evaluation.id_client)
        p0=np.zeros(len(learn));s0=p0.copy();p=np.zeros((len(learn),len(fc)));s=np.zeros((len(learn),len(gc)))
        inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=3200+k)
        for n,(ia,ib) in enumerate(inner.split(learn,learn.nombre_sinistres,learn.id_client),1):
            training,validation=learn.iloc[ia],learn.iloc[ib]
            assert set(training.id_client).isdisjoint(validation.id_client)
            p0[ib],s0[ib]=baseline(training,validation)
            for frequency,configs,values in [(True,fc,p),(False,gc,s)]:
                for j,c in enumerate(configs):values[ib,j]=engine.cached(training,validation,c,frequency)
            print(f'nested outer {k}/3 inner {n}/3 complete',flush=True)
        fp,fd=engine.blend_columns(p0,p,fc);gs,gd=engine.blend_columns(s0,s,gc)
        best=(float('inf'),0,0);y=learn.montant_sinistre.to_numpy()
        for i in range(fp.shape[1]):
            losses=((fp[:,i,None]*gs-y[:,None])**2).mean(axis=0);j=int(np.argmin(losses))
            if losses[j]<best[0]:best=(float(losses[j]),i,j)
        choice=dict(frequency=fd[best[1]],severity=gd[best[2]])
        ep,es=baseline(learn,evaluation);reference[b]=ep*es
        cp,cs=engine.choice_components(learn,evaluation,choice,ep,es);result[b]=cp*cs
        rows.append(dict(fold=k,choice=choice,selected=engine.metrics(evaluation.montant_sinistre,result[b]),incumbent=engine.metrics(evaluation.montant_sinistre,reference[b])))
        engine.write_json(out/'nested_progress.json',rows)
        print(f'nested outer {k} selected {rows[-1]["selected"]["RMSE"]:.6f}, incumbent {rows[-1]["incumbent"]["RMSE"]:.6f}',flush=True)
    report=dict(selected=engine.metrics(dev.montant_sinistre,result),incumbent=engine.metrics(dev.montant_sinistre,reference),
        gain=engine.bootstrap_gain(dev,result,reference),folds=rows,
        caveat='Nested selection of the new 525-pair catalogue relative to a fixed incumbent; earlier human exploration of these development rows remains a limitation.')
    engine.write_json(out/'nested_validation.json',report)
    np.savez_compressed(engine.CACHE/'nested_predictions.npz',selected=result,incumbent=reference)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    with engine.threadpool_limits(limits=4):main()
