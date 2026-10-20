"""Execute a fresh explanatory notebook from the restart's saved evidence."""
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND'] = 'Agg'
os.environ['IPYTHONDIR'] = str(ROOT/'.cache/ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT/'.cache/jupyter')
cells = [
('md', "# Reprise complète — fréquence × gravité avec XGBoost\n\nL'objectif est la RMSE de prime, avec deux composantes distinctes. Le notebook original est conservé. Les cellules suivantes lisent les résultats reproductibles de `scripts/restart_xgboost.py all` ; elles ne relancent pas la recherche coûteuse. Les associations descriptives ne sont pas des effets causaux."),
('code', "from pathlib import Path\nimport json, sys\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p/'run.py').exists())\nR = ROOT/'reports/restart_xgboost'\nsys.path.insert(0,str(ROOT/'src'))\nfrom pricing.experiment import load_data\ntrain,test,_,_=load_data()\naudit=json.loads((R/'data_audit.json').read_text())\naudit"),
('md', "## Dictionnaire et qualité des variables\n\nLes unités non documentées restent signalées. Les identifiants et les sexes sont exclus. Les zéros du second conducteur représentent son absence. Poids/cylindrée nuls et permis plus anciens que le conducteur sont comparés sous deux traitements : valeurs brutes avec indicateurs, ou valeurs manquantes avec indicateurs. Aucune suppression des sinistres extrêmes."),
('code', "dictionary=pd.read_csv(R/'variable_dictionary.csv')\npd.set_option('display.max_rows',40)\ndictionary"),
('code', "groups=pd.read_csv(R/'variable_groups.csv')\ncontract=groups[groups.variable.eq('type_contrat')].set_index('level')\nfig,ax=plt.subplots(1,2,figsize=(11,4))\ncontract.claim_rate.plot.bar(ax=ax[0],title='Probabilité descriptive par contrat')\ncontract.conditional_severity.plot.bar(ax=ax[1],title='Coût moyen parmi les sinistres')\nplt.tight_layout();plt.show()\ncontract"),
('code', "claims=train.loc[train.nombre_sinistres.eq(1),'montant_sinistre']\nfig,ax=plt.subplots(figsize=(8,4))\nax.hist(claims,bins=np.geomspace(claims.min(),claims.max(),45));ax.set_xscale('log')\nax.set(xlabel='Montant (échelle logarithmique)',ylabel='Effectif',title='Gravité : forte asymétrie et grandes valeurs conservées')\nplt.show()\nclaims.describe(percentiles=[.5,.9,.99])"),
('md', "## Protocole\n\nTrois plis externes séparés par client évaluent toute la sélection. Chacun contient cinq plis internes, 60 configurations de fréquence et 60 de gravité. Chaque ajustement choisit son nombre d'arbres sur un sous-ensemble de clients de son apprentissage, puis réapprend avec ce nombre d'arbres. Les catégories sont apprises localement.\n\nFréquence : `binary:logistic`, sans repondération des classes. Gravité : `reg:gamma` ou `reg:squarederror`, uniquement sur les sinistres positifs. La calibration logistique est croisée sur les prédictions internes hors pli ; son éventuel optimisme est inclus dans l'évaluation externe. Les cinq meilleures configurations de chaque composante sont comparées, puis leurs mélanges avec la référence sont sélectionnés sur la RMSE de prime.\n\nLes données ayant déjà été explorées, ces plis ne constituent pas un nouveau jeu indépendant. La recherche sur l'ensemble du train ne mesure pas la généralisation."),
('code', "results=json.loads((R/'nested_results.json').read_text())\npd.DataFrame({'Sélection XGBoost':results['selected'],'Référence initiale':results['reference']})"),
('code', "pd.DataFrame([{'pli':z['fold'],'RMSE sélection':z['selected']['RMSE'],'RMSE référence':z['reference']['RMSE']} for z in results['folds']])"),
('code', "print('Gain RMSE et intervalle bootstrap clients :',results['gain'])\nprint('Fréquence :',results['frequency'])\nprint('Gravité conditionnelle :',results['severity'])\npd.read_csv(R/'contract_performance.csv')"),
('code', "cal=pd.read_csv(R/'calibration.csv')\nfig,ax=plt.subplots(figsize=(5,5));ax.plot(cal.predicted,cal.observed,'o-')\nupper=max(cal.predicted.max(),cal.observed.max());ax.plot([0,upper],[0,upper],'--',color='gray')\nax.set(xlabel='Probabilité prédite',ylabel='Taux observé',title='Calibration sur les plis externes');plt.show()\ncal"),
('md', "## Sélection finale et fichier Kaggle\n\nLa recherche finale est ajustée sur tous les clients après l'évaluation externe. Si l'intervalle du gain inclut zéro, la référence reste recommandée pour la production ; le nouveau fichier est identifié comme challenger. Un score Kaggle inférieur à 1365.500 doit être constaté après soumission et ne se déduit pas du score local."),
('code', "final=json.loads((R/'final_model.json').read_text())\nprint('Choix final :',final['choice'])\nprint('Recommandation :',final['production_recommendation'])\nprint('Score Kaggle :',final['kaggle_score'])\nsubmission=pd.read_csv(ROOT/final['challenger'])\nassert list(submission)==['index','pred'] and len(submission)==50000\nassert np.array_equal(submission['index'],test['index'])\nassert np.isfinite(submission.pred).all() and submission.pred.ge(0).all()\nsubmission.head()"),
]
nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(v) if k=='md' else nbf.v4.new_code_cell(v) for k,v in cells])
nb.metadata.kernelspec = {'display_name':'Python 3 (pricing)','language':'python','name':'python3'}
dest = ROOT/'notebooks/04_reprise_xgboost.ipynb'
nbf.write(nb,dest)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,km=km,timeout=180,startup_timeout=30,resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
nbf.write(nb,dest)
print('Executed',dest)
