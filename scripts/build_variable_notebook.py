"""Execute the expanded-variable analysis from saved, verified results."""
from pathlib import Path
import sys,os
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython');os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter')
cells=[
('md',"# Réanalyse des variables et modèles de gravité\n\nLes cellules lisent les résultats déjà calculés ; elles ne relancent pas la recherche. Les sinistres extrêmes sont conservés et le notebook original reste inchangé. Les associations étudiées après consultation des données sont descriptives, pas causales."),
('code',"from pathlib import Path\nimport json\nimport pandas as pd\nROOT=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'run.py').exists())\nR=ROOT/'reports/variable_models'\nr=json.loads((R/'results.json').read_text())\nreceipt=json.loads((R/'kaggle_result.json').read_text())\nreceipt"),
('code',"pd.read_csv(ROOT/'reports/restart_xgboost/variable_dictionary.csv')"),
('code',"g=pd.read_csv(ROOT/'reports/variable_reanalysis/stable_descriptive_severity_groups.csv')\ng[['variable','level','claims','weighted_severity_relative_residual','top25_claims','severity_residual_without_top25_diagnostic']].head(15)"),
('md',"Les 25 coûts les plus élevés sont retirés uniquement dans la colonne diagnostique d'influence. Leur retrait réduit fortement certains effets apparents. Ils restent dans tous les entraînements. Catégories rares, permis incohérents et mesures nulles demandent une forte régularisation. L'absence du second conducteur est représentée séparément de ses mesures manquantes."),
('md',"20 configurations, 61 choix : courbes splines Ridge sur familles véhicule/conducteur/contrat/ensemble avec quatre régularisations ; quatre XGBoost sur toutes les variables ; mélanges à 25/50/100 % avec le modèle précédent, qui reste candidat seul. Trois plis externes et cinq plis internes séparés par client. Toutes les transformations et tous les ajustements sont appris dans les plis d'apprentissage. Les données ont déjà été explorées ; cette évaluation n'est pas un nouveau test indépendant."),
('code',"pd.DataFrame({name:r[name] for name in ['selected','incumbent']})"),
('code',"pd.DataFrame([{'fold':x['fold'],'selected_RMSE':x['selected']['RMSE'],'incumbent_RMSE':x['incumbent']['RMSE'],'choice':x['choice']} for x in r['folds']])"),
('code',"r['gain']"),
('code',"s=json.loads((R/'full_training/selection.json').read_text())\nprint('Poids :',r['final_choice']['weight'])\ns['catalogue'][r['final_choice']['candidate']]"),
('code',"v=json.loads((R/'verification.json').read_text())\nassert v['frequency_unchanged'] and v['fresh_process_byte_identical']\nv"),
('md',"La gravité finale mélange 25 % du modèle Ridge sur variables étendues et 75 % du précédent meilleur modèle. La fréquence originale reste inchangée. Le score public transmis est 1365.745, contre 1365.913 auparavant. L'intervalle local inclut zéro ; le score public ne prouve pas une supériorité sur le test privé. L'objectif strictement inférieur à 1365.500 reste non atteint. Commandes et détails : `docs/variable_reanalysis.md`."),
]
nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(v) if k=='md' else nbf.v4.new_code_cell(v) for k,v in cells])
nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
dest=ROOT/'notebooks/06_reanalyse_variables.ipynb';nbf.write(nb,dest);print(dest)
