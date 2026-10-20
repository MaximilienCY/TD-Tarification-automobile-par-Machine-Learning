"""Record the user's reported result without claiming independent browser verification."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/restart_xgboost'
receipt=dict(file='deliverables/candidates/submission_restart_xgboost.csv',public_score=1366.025,
    sha256='a7a7069658a3954983b3c4bc2a943dc71e76f9a632dd8855cc8ad7c7bc817941',
    source='User supplied Kaggle Complete result in this chat',independently_browser_verified=False,
    local_date='2026-10-12',reference_score=1365.943,worse_than_reference_by=.082)
(R/'kaggle_result.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
registry_path=ROOT/'reports/kaggle_submissions.json'
registry=json.loads(registry_path.read_text())
for entry in registry['submissions']:
    if entry.get('sha256')=='a7a7069658a3954983b3c4bc2a943dc71e76f9a632dd8855cc8ad7c7bc817941':
        entry.update(public_score=1366.025,status='complete_user_reported',score_source=receipt['source'],independently_browser_verified=False)
registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False),encoding='utf-8')
final_path=R/'final_model.json'
final=json.loads(final_path.read_text());final.update(kaggle_score=1366.025,submission_state='complete_user_reported',score_source=receipt['source'])
final_path.write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf-8')
queue_path=ROOT/'reports/submission_queue.json'
queue=json.loads(queue_path.read_text())
if queue['next_submission'].get('sha256')==receipt['sha256']:
    queue['next_submission'].update(kaggle_score=1366.025,status='complete_user_reported')
    queue['state']='researching_replacement_after_worse_user_reported_score'
queue_path.write_text(json.dumps(queue,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(receipt,indent=2))

anchored=dict(file='deliverables/candidates/submission_anchored.csv',public_score=1366.010,
    sha256='81f621aaca358abdd64311bb2a9349becacf2756c95ea53dc56b08a7ca56aa70',
    source='User supplied Kaggle Complete result in this chat',independently_browser_verified=False,
    local_date='2026-10-12',reference_score=1365.943,worse_than_reference_by=.067)
anchor_dir=ROOT/'reports/anchored'
(anchor_dir/'kaggle_result.json').write_text(json.dumps(anchored,indent=2),encoding='utf-8')
anchor_path=anchor_dir/'results.json';anchor_report=json.loads(anchor_path.read_text())
anchor_report['export']['kaggle_score']=anchored['public_score'];anchor_report['score_source']=anchored['source']
anchor_path.write_text(json.dumps(anchor_report,indent=2,ensure_ascii=False),encoding='utf-8')
for entry in registry['submissions']:
    if entry.get('sha256')==anchored['sha256']:
        entry.update(public_score=anchored['public_score'],status='complete_user_reported',score_source=anchored['source'],independently_browser_verified=False)
registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(anchored,indent=2))
