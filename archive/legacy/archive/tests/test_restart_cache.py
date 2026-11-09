"""Scheduling optimization must preserve exact statistical cache identities."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import restart_xgboost as pipeline
from restart_cache import install


def test_optimized_keys_match_canonical_keys(monkeypatch):
    train,test,_,_=pipeline.load_data()
    a,b=train.iloc[:200],test.iloc[:100]
    configurations=['reference',{'config':pipeline.catalogue(True)[0],'frequency':True},
                    {'config':pipeline.catalogue(False)[11],'frequency':False}]
    original_key,original_selection=pipeline.cache_key,pipeline.selection
    expected=[original_key(a,b,c) for c in configurations]
    monkeypatch.setattr(pipeline,'cache_key',original_key)
    monkeypatch.setattr(pipeline,'selection',original_selection)
    install()
    assert [pipeline.cache_key(a,b,c) for c in configurations] == expected
    install()  # Installing twice does not wrap selection again.
    assert [pipeline.cache_key(a,b,c) for c in configurations] == expected
