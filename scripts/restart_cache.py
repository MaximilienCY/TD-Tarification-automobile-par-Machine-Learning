"""Reuse immutable partition digests within a selection; preserve canonical cache keys."""
import hashlib
import importlib.metadata
import json
import weakref
import pandas as pd
import restart_xgboost as pipeline


def install():
    if getattr(pipeline.selection,'optimized_cache',False):
        return
    original_selection = pipeline.selection
    frames = {}
    state = {}

    def refresh():
        frames.clear()
        state['code'] = pipeline.Path(pipeline.__file__).read_bytes()
        state['source'] = pipeline.provenance()
        state['xgboost'] = importlib.metadata.version('xgboost')

    def key(learn,evaluation,config):
        h = hashlib.sha256(state['code'])
        h.update(json.dumps(dict(config=config,source=state['source'],xgboost=state['xgboost']),sort_keys=True).encode())
        for x in [learn,evaluation]:
            item = frames.get(id(x))
            if item is None or item[0]() is not x:
                item = (weakref.ref(x),pd.util.hash_pandas_object(x,index=True).to_numpy().tobytes())
                frames[id(x)] = item
            h.update(item[1])
        return h.hexdigest()

    def selection(frame,label,trials=60):
        # All pipeline transformations copy inputs, so these frames are immutable
        # for the duration of a selection. Refresh code/data provenance each call.
        refresh()
        return original_selection(frame,label,trials)

    selection.optimized_cache = True
    refresh()
    pipeline.cache_key = key
    pipeline.selection = selection
