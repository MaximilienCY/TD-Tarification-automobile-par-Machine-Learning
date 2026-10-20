"""Execute an evidence notebook for the post-1365.745 searches."""
from pathlib import Path
import os,sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython');os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter')
cells=[
('md',"# Recherches après 1365.745\n\nLe meilleur score Kaggle transmis par l'utilisateur est 1365.745. Les cellules lisent les résultats des recherches locales. Le modèle original et toutes les soumissions précédentes sont conservés. Les clients et le leaderboard ont déjà été étudiés : les résultats restent diagnostiques."),
('code',"from pathlib import Path\nimport json\nimport pandas as pd\nROOT=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'run.py').exists())\nnames=['smooth_severity','coverage_interactions','frequency_curves']\nresults={n:json.loads((ROOT/'reports'/n/'results.json').read_text()) for n in names}\npd.DataFrame([{'search':n,'RMSE':r['selected']['RMSE'],'incumbent_RMSE':r['incumbent']['RMSE'],'gain':r['gain']['gain_RMSE'],'export':r['export']} for n,r in results.items()])"),
('md',"La recette fixe du modèle à 1365.745 est réapprise par pli. Sa RMSE de 541.971031 diffère de la RMSE 541.942903 de la précédente procédure, qui sélectionnait des configurations différentes selon les plis. Les courbes de gravité affinées et les interactions couverture/usage dégradent cette référence fixe et n'exportent aucun CSV."),
('md',"La recherche de fréquence apprend une correction de logit des probabilités originales, avec splines et pénalisation L2 forte. Les probabilités auxiliaires sont calculées hors client dans trois sous-plis supplémentaires ; aucune probabilité ajustée sur son propre client n'est utilisée pour apprendre la correction. Pas de poids de classe ni de repondération des observations. La gravité du modèle à 1365.745 reste exactement inchangée. Trois plis externes évaluent toute la sélection dans cinq plis internes."),
('code',"r=results['frequency_curves']\npd.DataFrame([{'fold':x['fold'],'RMSE':x['selected']['RMSE'],'incumbent_RMSE':x['incumbent']['RMSE'],'choice':x['choice']} for x in r['folds']])"),
('code',"r['gain'],r['frequency']"),
('code',"pd.read_csv(ROOT/'reports/frequency_curves/contract_performance.csv')"),
('code',"s=json.loads((ROOT/'reports/frequency_curves/full_training/selection.json').read_text())\nprint('Poids :',r['final_choice']['weight'])\ns['catalogue'][r['final_choice']['candidate']] if r['final_choice']['candidate']>=0 else 'Incumbent inchangé'"),
('code',"verification=json.loads((ROOT/'reports/frequency_curves/verification.json').read_text())\nassert verification['status']=='verified'\nverification"),
('md',"La sélection minimise la RMSE de prime. Les intervalles bootstrap clients sont descriptifs, conditionnels aux prédictions ajustées. Une amélioration locale ne garantit pas une amélioration Kaggle ; aucun nouveau score public n'est inventé. Les exports, lorsqu'ils existent, sont reproduits octet pour octet dans un processus distinct. Commandes et résultats : `docs/variable_followups.md`."),
]
nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(v) if k=='md' else nbf.v4.new_code_cell(v) for k,v in cells])
nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
dest=ROOT/'notebooks/07_recherches_apres_1365_745.ipynb';nbf.write(nb,dest);print(dest)
