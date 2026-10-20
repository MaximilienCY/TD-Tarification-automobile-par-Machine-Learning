"""Build and execute the three compact teaching notebooks from validated reports."""
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg'
os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter')

INIT='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT / 'src'))
from pricing.experiment import load_data
train, test, development, historique = load_data()
R = ROOT / 'reports'
selection = json.loads((R / 'selection.json').read_text(encoding='utf-8'))
plt.rcParams.update({'figure.figsize': (9, 4), 'axes.spines.top': False, 'axes.spines.right': False})
'''


def save(name,cells):
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(text) if kind=='md' else nbf.v4.new_code_cell(text)
                                 for kind,text in cells])
    nb.metadata.kernelspec={'display_name':'Python 3 (pricing)','language':'python','name':'python3'}
    dest=ROOT/'notebooks'/name
    nbf.write(nb,dest)
    print('Executing',dest,flush=True)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(nb,km=km,timeout=180,startup_timeout=30,resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    nbf.write(nb,dest)
    print('Executed',dest,flush=True)


save('01_frequence.ipynb',[
('md', '# Modèle de fréquence — probabilité de sinistre\n\nLa cible vaut 1 si le contrat a connu un sinistre. Avec environ 5,8 % de positifs, une accuracy élevée ne démontre pas une bonne tarification. Nous évaluons des probabilités avec log-loss, score de Brier, AUC et calibration.\n\nLes résultats ci-dessous proviennent de `python run.py search`. Le code réutilisable est dans `src/pricing/`.'),
('code',INIT),
('code',"pd.DataFrame({'effectif': train.nombre_sinistres.value_counts(), 'proportion': train.nombre_sinistres.value_counts(normalize=True)})"),
('md','## Préparation et absence de fuite\n\nLes cibles, index et identifiants ne sont pas des prédicteurs. Le sexe est exclu. Poids/cylindrée impossibles deviennent manquants. Les zéros du second conducteur sont structurels. Les variables dérivées décrivent département, logarithmes, puissance/poids et interactions contrat/usage.\n\nImputation, standardisation, catégories et encodages sont appris uniquement sur le pli d’apprentissage. L’encodage cible de référence est lui-même croisé par client. Aucun client n’est présent dans deux partitions opposées.'),
('code',"assert set(development.id_client).isdisjoint(historique.id_client)\nprint(f'{len(development):,} contrats de développement ; {len(historique):,} de diagnostic historique.')\npd.read_csv(R / 'frequency_cv.csv').round(6)"),
('md','## Modèles comparés\n\nRégression logistique régularisée, splines et interactions ; boosting par histogrammes de référence ; CatBoost avec variables catégorielles ; LightGBM avec arbres régularisés. Aucune pondération artificielle des classes ne modifie la prévalence cible. Les poids de mélange sont choisis sur la MSE de prime, pas sur l’AUC seule.'),
('code',"[(c['name'],w) for c,w in zip(selection['frequency_configs'],selection['choice']['frequency_weights']) if w]"),
('code',"v=pd.read_csv(R/'historical_predictions.csv')\nv['decile']=pd.qcut(v.proba_pred,10,duplicates='drop')\nc=v.groupby('decile',observed=True).agg(predicted=('proba_pred','mean'),observed=('claim','mean'),n=('claim','size'))\nfig,ax=plt.subplots();ax.plot(c.predicted,c.observed,'o-');m=max(c.predicted.max(),c.observed.max());ax.plot([0,m],[0,m],'--',color='gray');ax.set(xlabel='Probabilité moyenne prédite',ylabel='Taux observé',title='Calibration — diagnostic historique déjà examiné');plt.show()\nc"),
('md','## Lecture\n\nLe diagnostic historique n’est pas une nouvelle validation indépendante. La validation imbriquée de la prime dans le troisième notebook évalue toute la procédure de sélection. Des probabilités bien calibrées doivent être multipliées par les coûts conditionnels pour juger la tarification finale.'),
])

save('02_gravite.ipynb',[
('md','# Modèle de gravité — coût conditionnel\n\nLe modèle est entraîné uniquement sur les sinistres positifs. Nous cherchons une **espérance de coût**, car la prime finale est évaluée par une erreur quadratique. Une médiane ou une exponentielle naïve de log-coût ne remplit pas automatiquement cet objectif.'),
('code',INIT),
('code',"claims=train.loc[train.nombre_sinistres.eq(1)]\nclaims.montant_sinistre.describe(percentiles=[.5,.9,.95,.99])"),
('code',"fig,ax=plt.subplots();ax.hist(claims.montant_sinistre,bins=60,color='#157a6e');ax.set(xlabel='Coût du sinistre',ylabel='Effectif',yscale='log',title='Asymétrie et sinistres rares de coût élevé');plt.show()"),
('md','## Modèles et régularisation\n\nRéférence Gamma à lien logarithmique avec ancienneté du véhicule, contrat et usage. Alternatives : Gamma enrichi avec splines ; régression de Poisson à lien log pour une moyenne positive ; Ridge sur le coût ; CatBoost et LightGBM. Les modèles flexibles sont régularisés car environ 2 900 observations positives sont disponibles.\n\nLes sinistres extrêmes sont conservés dans l’apprentissage et l’évaluation : leur suppression réduirait artificiellement le risque estimé. Les prédictions de coût sont positives. Les plis sont construits sur tous les contrats et partagés avec la fréquence, puis seuls les sinistres du pli d’apprentissage entraînent la gravité.'),
('code',"pd.read_csv(R/'severity_cv.csv').round(4)"),
('code',"[(c,w) for c,w in zip(selection['severity_configs'],selection['choice']['severity_weights']) if w]"),
('md','## Pourquoi ne pas choisir uniquement la meilleure RMSE de gravité ?\n\nUne amélioration sur les seuls sinistrés peut détériorer la prime sur l’ensemble des contrats. Le couple et ses mélanges sont donc sélectionnés par `mean((probabilité × coût_conditionnel − coût_réel)**2)` sur les prédictions hors pli.'),
('code',"v=pd.read_csv(R/'historical_predictions.csv');positive=v.loc[v.claim.eq(1)]\nprint('RMSE conditionnelle historique :',np.sqrt(np.mean((positive.montant_pred-positive.cost)**2)))\npositive[['cost','montant_pred']].describe()"),
])

save('03_tarification.ipynb',[
('md','# Tarification commune et validation\n\n**Prime pure = probabilité de sinistre × coût attendu conditionnel.** Ce document réunit les deux composantes et distingue recherche, validation imbriquée, diagnostic historique et score Kaggle.'),
('code',INIT),
('md','## Protocole\n\n1. Partition par client : 80 % développement et 20 % diagnostic historique déjà connu.\n2. Recherche sur cinq plis groupés, identiques pour fréquence et gravité.\n3. Mélanges convexes prédéfinis et choix sur MSE de prime hors pli.\n4. Validation imbriquée : trois plis externes ; dans chaque apprentissage externe, la recherche entière est répétée sur trois plis internes.\n5. Réentraînement final sur tous les contrats / tous les sinistres positifs.\n\nMSE et RMSE ordonnent les modèles de la même manière. Les scores de recherche sont optimistes après sélection. Le développement a déjà été exploré avant cette étude ; la validation imbriquée ne supprime pas cette limite historique.'),
('code',"pd.read_csv(R/'search_pairs.csv').head(12)"),
('code',"nested=json.loads((R/'nested_validation.json').read_text(encoding='utf-8'))\npd.DataFrame(nested['scores']).T"),
('code',"pd.read_csv(R/'nested_folds.csv')"),
('code',"pd.DataFrame({k:nested[k] for k in ['gain_vs_reference','gain_vs_previous']}).T"),
('md','## Incertitude et grands sinistres\n\nLe bootstrap rééchantillonne les clients, pas les lignes indépendamment. L’intervalle est descriptif et conditionnel aux prédictions déjà ajustées. Un intervalle qui contient zéro ne prouve pas la supériorité du nouveau modèle.'),
('code',"n=pd.read_csv(R/'nested_predictions.csv');n['sq_error']=(n.cost-n.selected)**2\npos=n.loc[n.cost.gt(0)].sort_values('cost',ascending=False);k=int(np.ceil(.01*len(pos)))\nprint(f'Les {k} plus gros sinistres représentent {pos.head(k).sq_error.sum()/n.sq_error.sum():.1%} de l’erreur quadratique.')\nfig,ax=plt.subplots();pd.DataFrame(nested['scores']).T.RMSE.plot.barh(ax=ax,color='#225ea8');ax.set(xlabel='RMSE',title='Comparaison sur les mêmes plis externes');plt.show()"),
('md','## Export et reproductibilité\n\nLa soumission respecte le fichier officiel `prime_pred_sandbox.csv`. Les prédictions détaillées conservent séparément fréquence, gravité et produit. Le modèle sauvegardé se recharge sans exécuter un notebook. Les empreintes de données, code et versions sont enregistrées dans les rapports.'),
('code',"sub=pd.read_csv(ROOT/'submission.csv');sample=pd.read_csv(ROOT/'Data/prime_pred_sandbox.csv')\nassert list(sub.columns)==list(sample.columns)==['index','pred']\nnp.testing.assert_array_equal(sub['index'],sample['index'])\nassert np.isfinite(sub.pred).all() and sub.pred.ge(0).all()\nprint('Soumission valide :',len(sub),'lignes')\nsub.head()"),
('code',"kaggle_path=R/'kaggle_submissions.json'\nif kaggle_path.exists():\n    display(pd.DataFrame(json.loads(kaggle_path.read_text(encoding='utf-8'))['submissions']))\nelse:\n    print('Score Kaggle à consulter après soumission. Aucun score local ne le remplace.')"),
])
