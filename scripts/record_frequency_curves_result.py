"""Record the exact Kaggle result supplied by the user; no network verification."""
import json
import frequency_curves as f
ROOT=f.ROOT;OUT=f.OUT
path=ROOT/'deliverables/candidates/submission_frequency_curves.csv'
sha=f.r.digest(path)
assert sha=='463eed0812b6e6de59dbe1f2218c234c134f4d198205deffa8a19299470b9803'
receipt=dict(file=str(path.relative_to(ROOT)),sha256=sha,public_score=1365.657,
    source='User supplied Kaggle Complete public score in this chat',independently_browser_verified=False,
    local_date='2026-10-12',previous_best_public_score=1365.745,improvement=.088,target=1365.5,target_reached=False)
f.r.write_json(OUT/'kaggle_result.json',receipt)
result=json.loads((OUT/'results.json').read_text());result['export']['kaggle_score']=1365.657
result['recommendation']='new user-reported public incumbent';f.r.write_json(OUT/'results.json',result)
print(json.dumps(receipt,indent=2))
