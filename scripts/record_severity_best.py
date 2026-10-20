"""Record the user's reported new public incumbent, keeping verification provenance."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
receipt=dict(file='deliverables/candidates/submission_severity_refinement.csv',public_score=1365.913,
    sha256='2874eeb918f4fd40ec773baf000a3cf743da4a036e230c0fb204a0447a57c1ee',
    source='User supplied Kaggle Complete public score in this chat',independently_browser_verified=False,
    local_date='2026-10-12',previous_best_public_score=1365.943,improvement=.030,target=1365.500,target_reached=False)
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
save(ROOT/'reports/severity_refinement/kaggle_result.json',receipt)
path=ROOT/'reports/severity_refinement/results.json';result=json.loads(path.read_text())
result['export']['kaggle_score']=receipt['public_score'];result['recommendation']='new public incumbent reported by user; local gain remains uncertain';result['score_source']=receipt['source'];save(path,result)
path=ROOT/'reports/kaggle_submissions.json';registry=json.loads(path.read_text())
for z in registry['submissions']:
    if z.get('sha256')==receipt['sha256']:z.update(public_score=receipt['public_score'],status='complete_user_reported',score_source=receipt['source'],independently_browser_verified=False)
registry['best_user_reported_public_score']=receipt['public_score'];registry['best_user_reported_file']=receipt['file'];save(path,registry)
path=ROOT/'reports/submission_queue.json';queue=json.loads(path.read_text())
queue['public_incumbent']=receipt;queue['best_user_reported_public_score']=receipt['public_score'];queue['state']='improving_severity_public_incumbent'
if 'exploratory_candidate' in queue:queue['exploratory_candidate'].update(kaggle_score=receipt['public_score'],status='complete_user_reported')
save(path,queue)
print(json.dumps(receipt,indent=2))
