"""Execute a notebook summarizing the saved successor evaluation."""
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter')
nb=nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell("# Successeur de gravité\n\nLe score public précédent, transmis par l'utilisateur, est 1365.913. Ce notebook lit les résultats de `scripts/improve_severity.py train` et les contrôles reproductibles. Le nouveau score Kaggle reste inconnu. Les données ont déjà été explorées ; l'évaluation locale est diagnostique."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nROOT=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'run.py').exists())\nR=ROOT/'reports/severity_successor'\nr=json.loads((R/'results.json').read_text())\npd.DataFrame({name:r[name] for name in ['selected','incumbent','original']})"),
    nbf.v4.new_markdown_cell("93 choix prédéfinis : 21 corrections, deux moyennes, quatre poids et Gamma seule. Trois plis externes séparés par client évaluent la sélection sur cinq plis internes. Probabilités originales inchangées, corrections entraînées seulement sur les sinistres, pondérations auxiliaires calculées hors client, sinistres extrêmes conservés."),
    nbf.v4.new_code_cell("pd.DataFrame([{'fold':x['fold'],'selected_RMSE':x['selected']['RMSE'],'incumbent_RMSE':x['incumbent']['RMSE'],'choice':x['choice']} for x in r['folds']])"),
    nbf.v4.new_code_cell("r['gain_vs_incumbent']"),
    nbf.v4.new_markdown_cell("L'intervalle bootstrap inclut zéro : le gain n'est pas établi avec précision. La recette fixe du modèle à 1365.913 est réapprise par pli. Cette comparaison diffère de l'ancienne recherche qui sélectionnait un modèle de gravité par pli."),
    nbf.v4.new_code_cell("selection=json.loads((R/'full_training/selection.json').read_text())\nconfig=selection['catalogue'][r['final_choice']['candidate']]\nprint('Poids de correction :',r['final_choice']['weight'])\nconfig"),
    nbf.v4.new_code_cell("verification=json.loads((R/'verification.json').read_text())\nassert verification['status']=='verified'\nassert verification['frequency_unchanged'] and verification['fresh_process_byte_identical']\nassert verification['full_refit_incumbent_exact']\nverification"),
    nbf.v4.new_code_cell("pd.read_csv(ROOT/r['export']['file']).describe()"),
    nbf.v4.new_markdown_cell("Export : `deliverables/candidates/submission_severity_successor.csv`, 50 000 lignes index,pred. L'ancien meilleur CSV et le notebook original sont préservés. Aucun score Kaggle nouveau n'est revendiqué. Commandes et détails : `docs/severity_successor.md`."),
])
nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
path=ROOT/'notebooks/05_successeur_gravite.ipynb';nbf.write(nb,path)
print(path)
