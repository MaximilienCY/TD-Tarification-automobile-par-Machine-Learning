"""Precompute independent training partitions on three processes, then evaluate/export.

Equivalent statistical procedure to restart_xgboost.py all. Only scheduling differs.
Never run concurrently with another restart command sharing this cache.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from concurrent.futures import ProcessPoolExecutor
import restart_xgboost as pipeline
from restart_cache import install
from threadpoolctl import threadpool_limits


def precompute(partition):
    install()
    train, _, _, _ = pipeline.load_data()
    with threadpool_limits(limits=4):
        if partition == 4:
            pipeline.selection(train, 'full_training')
        else:
            a, _ = list(pipeline.split_groups(train,3,pipeline.SEED+23))[partition-1]
            pipeline.selection(train.iloc[a],f'outer_{partition}')
    return partition


if __name__ == '__main__':
    install()
    train,test,_,_=pipeline.load_data()
    pipeline.study(train,test)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for done in pool.map(precompute,[1,2,3,4]):
            print('Partition search complete:',done,flush=True)
    with threadpool_limits(limits=4):
        result=pipeline.nested(train)
        pipeline.finalize(train,test,result)
