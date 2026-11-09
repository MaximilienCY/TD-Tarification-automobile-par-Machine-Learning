"""Construire le notebook pédagogique du modèle principal, sans relancer la recherche."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text):cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md(r'''
# MODÈLE PRINCIPAL — tarification automobile

**Recette retenue : `frequency_curves_1365_657`.**

Ce notebook est le point de lecture principal. Il explique les données, les variables,
les modèles réellement utilisés, leur apprentissage, la validation, leurs faiblesses
et la soumission. Les versions précédentes sont conservées dans `archive/`.

Le **score public communiqué est 1365,657**, associé à un CSV précis dans le reçu
historique. Il n'a pas été vérifié à nouveau sur Kaggle. La **RMSE locale historique
est 541,915571 €**, sur d'autres observations : ces deux nombres ne sont pas
interchangeables. Le modèle est destiné à l'exercice, pas à un tarif commercial
annuel dont l'exposition et les frais seraient documentés.

Environnement : Python 3.12 et `requirements-main.txt`. Les cellules utilisent le
code Python partagé du projet ; elles ne chargent jamais le code d'un autre notebook.
Les sections « code exact » affichent les sources des objets effectivement utilisés.
''')
code('''
from pathlib import Path
import sys,json,inspect,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,Code
from threadpoolctl import threadpool_limits
roots=[Path.cwd(),*Path.cwd().parents]
ROOT=next(p for p in roots if (p/'Data/train.csv').exists() and (p/'src/pricing').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from pricing import main_model as main
from pricing import reference as reference_code
f=main.libraries()
threadpool_limits(limits=4)
OUT=ROOT/'reports/main';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
def figure(fig,name):
    fig.tight_layout();fig.savefig(FIG/name,dpi=150,bbox_inches='tight');display(fig);plt.close(fig)
def scores(y,p):return f.r.score(np.asarray(y),np.asarray(p))
''')

md(r'''
## 1. Cadrage : que prédit-on ?

Pour un contrat décrit par X :

\[\widehat{\mathrm{prime}}(X)=\widehat p(X)\times\widehat s(X).\]

`p` est la probabilité d'au moins un sinistre ; `s` est le coût moyen **conditionnel
à un sinistre**. Le coût réel vaut zéro sans sinistre. La fréquence apprend sur
tous les contrats ; la gravité seulement sur les sinistres positifs.

Nous prédisons une gravité conditionnelle pour **chaque prospect**, car son accident
futur n'est pas connu. Aucun seuil de classification ne décide qui paiera une
prime nulle. La MSE de prime sert à sélectionner ; sa racine, la RMSE, exprime
l'erreur en euros. `pred` dans le CSV est une prime, jamais une RMSE.
''')
code('''
train,test,dev,historique=f.r.load_data()
claims=train.loc[train.nombre_sinistres.eq(1)]
etat=pd.DataFrame([
    {'jeu':'Train','contrats':len(train),'clients':train.id_client.nunique(),'sinistres':len(claims),'cibles_connues':True},
    {'jeu':'Test prospects','contrats':len(test),'clients':np.nan,'sinistres':np.nan,'cibles_connues':False}])
display(etat)
print(f'Taux de sinistre : {len(claims)/len(train):.3%}.')
print('Le test ne contient pas les identifiants clients : on ne peut pas mesurer leur recouvrement avec le train.')
print('Index du test :',test['index'].min(),'à',test['index'].max())
display(claims.montant_sinistre.describe(percentiles=[.5,.9,.95,.99]).to_frame('coût conditionnel (€)'))
assert main.sha256(ROOT/'Data/train.csv')=='8b5ca8a59c214269056ed1720de33619c10968549c383ec350898a659b97bfc5'
assert main.sha256(ROOT/'Data/test.csv')=='68cdf87201974688d754d846c2d8ff33c5aa2f9c011e84fba92018186cd8f569'
''')
md('''
## 2. Comprendre la cible et les données

La majorité des contrats n'a aucun coût. Parmi les sinistrés, les montants sont
très asymétriques. Leur moyenne dépasse la médiane et quelques grands sinistres
pèsent fortement dans une perte quadratique. Nous ne les supprimons pas et ne
remplaçons pas la moyenne conditionnelle par une médiane issue d'un logarithme.

Le minimum positif observé n'est pas une preuve de franchise contractuelle connue.
Les durées ne sont pas annualisées. Les variables de vente et de permis comportent
des valeurs dont le sens ou la cohérence doivent être documentés.
''')
code('''
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
counts=train.nombre_sinistres.value_counts().reindex([0,1])
axes[0].bar(['Sans sinistre','Sinistre'],counts,color=['#64748b','#2563eb'])
axes[0].set(title='Répartition de la cible de fréquence',ylabel='Nombre de contrats')
positive=claims.montant_sinistre.to_numpy()
axes[1].hist(positive,bins=np.geomspace(positive.min(),positive.max(),45),color='#2563eb')
axes[1].set(xscale='log',xlabel='Montant positif (€), échelle logarithmique',ylabel='Nombre de sinistres',title='Distribution de la gravité — tous les sinistres')
figure(fig,'01_cibles.png')
audit=[]
for col in reference_code.RAW_FEATURES:
    a,b=train[col],test[col]
    audit.append({'variable':col,'type':str(a.dtype),'manquants_train':int(a.isna().sum()),
        'manquants_test':int(b.isna().sum()),'modalites_train':int(a.nunique()),
        'nouvelles_modalites_test':int((b.notna() & ~b.isin(a.dropna())).sum()) if not pd.api.types.is_numeric_dtype(a) else np.nan})
audit=pd.DataFrame(audit);audit.to_csv(OUT/'audit_variables.csv',index=False);display(audit)
anomalies=pd.DataFrame({
    'poids_nul':[int(train.poids_vehicule.eq(0).sum())],
    'cylindree_nulle':[int(train.cylindre_vehicule.eq(0).sum())],
    'permis1_superieur_age1':[int(train.anciennete_permis1.gt(train.age_conducteur1).sum())],
    'age_second_conducteur_nul':[int(train.age_conducteur2.eq(0).sum())]})
display(anomalies)
''')
md('''
### Corrélations : diagnostic, pas règle automatique de sélection

Les corrélations de Spearman comparent les rangs des variables quantitatives. Elles
peuvent révéler la redondance entre caractéristiques de véhicule ou entre âge et
permis. Elles ne décrivent pas correctement toutes les catégories, interactions,
non-linéarités ou effets conditionnels. La cible de gravité est étudiée seulement
sur les sinistrés. Ces graphiques ne déclenchent aucun nouveau réglage du modèle.
''')
code('''
numeriques=[c for c in reference_code.RAW_FEATURES if pd.api.types.is_numeric_dtype(train[c])]
corr=train[numeriques].corr(method='spearman')
corr.to_csv(OUT/'correlations_spearman.csv')
fig,ax=plt.subplots(figsize=(11,9))
im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
ax.set_xticks(range(len(corr)),corr.columns,rotation=70,ha='right');ax.set_yticks(range(len(corr)),corr.index)
ax.set_title('Corrélations de Spearman — prédicteurs quantitatifs du train')
fig.colorbar(im,ax=ax,label='Coefficient de Spearman');figure(fig,'02_correlations.png')
association=claims[numeriques+['montant_sinistre']].corr(method='spearman')['montant_sinistre'].drop('montant_sinistre').sort_values()
display(association.to_frame('association avec la gravité, sinistrés seulement'))
''')

md(r'''
## 3. Architecture exacte du modèle principal

**Fréquence :** `p = sigmoid(logit(p0) + correction_age_couverture_usage)`.

**Gravité :** `s = 0,375 × Gamma + 0,375 × Gamma corrigé par XGBoost + 0,25 × Gamma corrigé par Ridge`.

**Prime :** `p × s`.

Les poids de gravité résultent de deux mélanges successifs : 50/50 Gamma–XGBoost,
puis 75 % de ce mélange et 25 % de Ridge. Le modèle n'est donc ni une régression
linéaire ordinaire, ni un Gamma pur à trois variables. Le Gamma reste son ancre.
''')
code('''
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(13,4.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#f8fafc',edgecolor='#64748b'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#475569','lw':1.4})
box(.015,.39,.17,.22,'Caractéristiques du contrat\\nCibles et identifiants exclus')
box(.255,.69,.37,.22,'Logistique C = 0,1\\n+ correction de logit par courbes\\nâge véhicule × contrat, usage')
box(.255,.08,.37,.33,'Gravité conditionnelle\\n37,5 % Gamma à lien log\\n37,5 % correction XGBoost du Gamma\\n25 % correction Ridge du Gamma')
box(.72,.39,.25,.22,'Prime pure = p × s\\nExport : index,pred')
arrow((.185,.55),(.255,.8));arrow((.185,.45),(.255,.25))
arrow((.625,.8),(.75,.61));arrow((.625,.25),(.75,.39))
ax.text(.67,.74,'p',fontsize=13);ax.text(.67,.25,'s',fontsize=13)
figure(fig,'03_architecture.png')
display(pd.json_normalize(main.RECIPE))
''')

md('''
### Variables conservées : distinguer chaque composante

« Trois variables » ne décrit que le Gamma et la correction XGBoost de base.
La logistique et la Ridge utilisent davantage d'informations. Le département
intervient dans la fréquence et la Ridge, sans entrer dans le Gamma. Être utilisé
ne signifie pas qu'un champ apporte à lui seul un gain causal ou statistiquement net.
La fin de vente n'est utilisée par aucune des composantes retenues.
''')
code('''
base_frequency=set(reference_code.RAW_FEATURES)-set(reference_code.REDONDANTES)
basic={'anciennete_vehicule','type_contrat','utilisation'}
ridge_raw=set(reference_code.RAW_FEATURES)-{'fin_vente_vehicule'}
rows=[]
for col in reference_code.RAW_FEATURES:
    rows.append({'champ':col,'logistique_base':col in base_frequency,'correction_frequence':col in basic,
        'Gamma':col in basic,'correction_XGBoost':col in basic,'correction_Ridge':col in ridge_raw,
        'note':'transformé en département' if col=='code_postal' else 'inutilisé' if col=='fin_vente_vehicule' else ''})
usage=pd.DataFrame(rows);usage.to_csv(OUT/'variables_par_composante.csv',index=False);display(usage)
print('Exclus : index, id_client, id_vehicule, id_contrat, sexes et cibles.')
print('Ridge : les variables dérivées comprennent absence du second conducteur, anomalies de permis, zéros techniques et écart de vente/âge.')
''')

md(r'''
## 4. Comment la fréquence apprend-elle ?

La base est une logistique régularisée L2 (`C = 0,1`). Imputation hiérarchique,
logarithmes `log1p`, standardisation, catégories rares et encodages de cible sont
appris dans l'apprentissage autorisé. Marque, modèle et département sont encodés
par taux lissé (`lissage = 100`) ; les valeurs d'apprentissage excluent le client
concerné, et leur prior exclut aussi son pli.

La correction finale utilise des splines quadratiques de l'âge du véhicule à
quatre nœuds uniformes, des catégories contrat/usage et leurs interactions avec
l'âge du véhicule. Les interactions sont multipliées par 0,5. Elle minimise la
somme des log-loss binomiales avec pénalisation L2 **alpha = 100**, intercept
compris. Son poids final est 1 : toute la correction est ajoutée au logit.

Pour apprendre cette correction, ses probabilités de base proviennent de trois
sous-plis séparés par client. Pour prédire, la base est réapprise sur tout
l'apprentissage autorisé. Aucun poids de classe ne force un équilibre artificiel.
''')
code('''
# Code exact des principales étapes d'encodage et de la correction retenue.
for obj in [reference_code.encoder_hors_clients,reference_code.apprendre_frequence,
            reference_code.ReferenceFrequence,f.loss_gradient,f.Correction]:
    display(Markdown('**'+obj.__name__+'**'));display(Code(inspect.getsource(obj),language='python'))
''')

md(r'''
## 5. Comment la gravité apprend-elle ?

**Gamma :** lien log, `alpha = 0,1`, type de contrat, ancienneté du véhicule,
utilisation. Il prédit une moyenne positive et optimise sa déviance régularisée.

**XGBoost :** 240 arbres de profondeur 2, taux d'apprentissage 0,025,
`min_child_weight = 100`, L2 = 100, L1 = 1, seed 20261012. Il apprend le ratio
coût/Gamma hors client, à partir des mêmes trois variables. Son montant corrigé
est Gamma multiplié par ce ratio prédit, avec protection de positivité.

**Ridge :** alpha = 10, panneau complet véhicule/conducteur/contrat/département,
splines quadratiques à quatre nœuds, regroupement des catégories à moins de
15 sinistres. Il apprend coût/Gamma hors client − 1 et corrige le Gamma par ce ratio.

Les probabilités et Gamma auxiliaires sont appris hors client dans trois plis.
La probabilité auxiliaire de gravité vient d'une logistique compacte ; elle n'est
pas la fréquence finale corrigée. Le poids des corrections est proportionnel à
`p_aux × Gamma_aux²`, normalisé à moyenne 1. L'identité
`p×s²×(y/s−r)² = p×(y−s×r)²` explique l'échelle de cette pondération.

Tous les grands sinistres sont conservés. Les planchers numériques positifs ne
sont ni des franchises connues ni une suppression des coûts extrêmes.
''')
code('''
for obj in [reference_code.GammaMetier,f.previous.moments,f.old.Correction,
            f.vm.features,f.vm.Correction]:
    display(Markdown('**'+obj.__module__+'.'+obj.__name__+'**'));display(Code(inspect.getsource(obj),language='python'))
''')

md('''
## 6. Réentraîner et inspecter les objets réellement utilisés

Les paramètres retenus sont figés. Cette cellule recharge le modèle principal
local ; s'il manque, elle réapprend uniquement cette recette, sans relancer les
anciennes recherches. La fréquence utilise 50 000 contrats, les gravités 2 917
sinistres. Le test ne fournit aucune cible d'apprentissage.

Le CSV soumis est conservé séparément des prédictions recalculées. Une différence
de plateforme peut modifier les derniers chiffres flottants ; le contrôle compare
chaque ligne numériquement et distingue cela d'une identité d'octets.
''')
code('''
if not main.MODEL_PATH.exists():main.train_main()
model=main.load_model();objects=main.components(model)
manifest=json.loads((ROOT/'reports/final_model.json').read_text())
display(pd.DataFrame({'composante':list(objects),'classe':[type(x).__name__ for x in objects.values()]}))
display(pd.DataFrame([manifest['versions']]))
display(pd.DataFrame([manifest['local_reproduction']]))
print('Gamma :',objects['gamma'].config)
print('XGBoost :',objects['xgboost'].config)
print('Ridge :',objects['ridge'].config)
''')

md('''
### Coefficients : ce qu'ils permettent d'expliquer

Un coefficient logistique agit sur les log-odds ; celui du Gamma sur le log du
montant. Les coefficients portent sur les **entrées transformées**, pas toujours
sur une unité brute. Les coefficients des splines décrivent ensemble une courbe.
Pour comparer deux modalités Gamma, le facteur relatif est exp(beta A − beta B).
La Ridge corrige un ratio de coût. Ces associations sont conditionnelles au modèle
et ne sont pas des effets causaux. Leurs valeurs absolues ne sont pas directement
comparables entre variables différemment encodées ou mises à l'échelle.
''')
code('''
def coefficients(names,values,filename):
    t=pd.DataFrame({'entree_preparee':list(names),'coefficient':np.asarray(values).ravel()})
    t.to_csv(OUT/filename,index=False);return t
gamma=coefficients(objects['gamma'].prep_.get_feature_names_out(),objects['gamma'].model_.coef_,'coefficients_gamma.csv')
display(gamma)
base=coefficients(objects['frequency_base'].params_['colonnes'],objects['frequency_base'].model_.coef_,'coefficients_logistique.csv')
display(base.reindex(base.coefficient.abs().sort_values(ascending=False).index).head(20))
correction=objects['frequency_correction'];names=list(correction.prep_.get_feature_names_out())
names+=['0.5*'+names[i]+'*'+names[j] for i in correction.numeric_ for j in correction.coverage_]
delta=coefficients(['intercept']+names,correction.coef_,'coefficients_correction_frequence.csv')
display(delta)
ridge=coefficients(objects['ridge'].prep_.get_feature_names_out(),objects['ridge'].model_.coef_,'coefficients_ridge.csv')
display(ridge.reindex(ridge.coefficient.abs().sort_values(ascending=False).index).head(20))
print('Tous les coefficients sont exportés, même ceux qui ne figurent pas parmi les vingt premiers.')
''')

md('''
### Exemple calculé de bout en bout

Chaque ligne ci-dessous donne la probabilité de base, sa correction, les trois
montants, leur mélange et la prime obtenue. Le produit n'est pas un montant
d'indemnisation promis à un client : c'est un coût attendu avant l'accident.
''')
code('''
examples=main.details(model,test.iloc[:5])
display(examples[['index','original_probability','correction_logit','proba_pred','gamma','xgboost_corrige','ridge_corrige','montant_pred','pred']])
np.testing.assert_allclose(examples.montant_pred,.375*examples.gamma+.375*examples.xgboost_corrige+.25*examples.ridge_corrige)
np.testing.assert_allclose(examples.pred,examples.proba_pred*examples.montant_pred)
''')

md('''
## 7. Validation : clients séparés, sélection et diagnostic distingués

La recherche historique comparait neuf corrections et trois poids, plus l'option
inchangée : 28 choix. Elle utilisait cinq plis internes et trois plis externes par
client. La gravité était maintenue fixe. Les trois apprentissages externes ont
choisi `age_coverage`, alpha 100, poids 1.

Le diagnostic courant **réapprend la recette fixe** sur ces trois plis externes.
Il ne relance pas les 28 choix internes. Il contrôle la reproduction et permet
des analyses hors apprentissage, mais ne constitue pas une nouvelle sélection
imbriquée indépendante. Données et leaderboard ont déjà influencé des choix
humains antérieurs ; aucune technique ne supprime cette exposition passée.
''')
code('''
if not (OUT/'diagnostic_oof.csv').exists():main.diagnostic_main()
oof=pd.read_csv(OUT/'diagnostic_oof.csv');diagnostic=json.loads((OUT/'diagnostic.json').read_text())
np.testing.assert_array_equal(oof['index'],train['index'])
comparison=pd.DataFrame({name:diagnostic[name] for name in ['selected','incumbent','original_gamma','constant']}).T
display(comparison)
display(pd.DataFrame(diagnostic['frequency']).T)
display(pd.DataFrame([diagnostic['severity_claims']],index=['gravite_sur_sinistres']))
display(pd.DataFrame(diagnostic['folds']))
display(pd.DataFrame([diagnostic['gain']]))
print('Nombre de clients communs à apprentissage et validation de chaque pli : zéro.')
''')

md('''
### Calibration des probabilités et des primes

Nous regroupons les prédictions hors apprentissage par décile. Une bonne
calibration rapproche la moyenne prédite de la fréquence ou du coût moyen
observé. Le découpage en déciles est descriptif ; les points fluctuent et ne
prouvent pas une calibration exacte pour chaque client.
''')
code('''
fig,axes=plt.subplots(1,2,figsize=(12,4.3))
for ax,col,target,title,xlabel,ylabel in [
    (axes[0],'proba_pred','claim','Calibration de la fréquence','Probabilité moyenne prédite','Taux de sinistre observé'),
    (axes[1],'pred','observed','Calibration de la prime','Prime moyenne prédite (€)','Coût moyen observé (€)')]:
    z=oof.copy();z['decile']=pd.qcut(z[col],10,labels=False,duplicates='drop')
    cal=z.groupby('decile').agg(prediction=(col,'mean'),observation=(target,'mean'),contrats=('index','size'))
    cal.to_csv(OUT/('calibration_'+col+'.csv'))
    ax.plot(cal.prediction,cal.observation,'o-',label='Déciles hors apprentissage')
    lim=max(cal.prediction.max(),cal.observation.max())*1.05
    ax.plot([0,lim],[0,lim],'--',color='grey',label='Égalité prédiction / observation')
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.legend(fontsize=8)
figure(fig,'04_calibration.png')
''')

md('''
### Erreurs, gros sinistres et segments

La RMSE et la MAE ne privilégient pas les mêmes erreurs. Une baisse de RMSE peut
coexister avec une hausse de MAE. Les coûts extrêmes sont analysés, jamais retirés
de l'apprentissage. Les résultats par contrat affichent leurs effectifs : une
petite différence dans un groupe peu sinistré ne doit pas être surinterprétée.
''')
code('''
e=oof.copy();e['erreur_carree']=(e.observed-e.pred)**2;e['residu']=e.observed-e.pred
claim_errors=e.loc[e.claim.eq(1)].sort_values('observed',ascending=False)
top_n=int(np.ceil(.01*len(claim_errors)))
tail_share=claim_errors.head(top_n).erreur_carree.sum()/e.erreur_carree.sum()
print(f'Les {top_n} plus gros sinistres (1 % des sinistrés) représentent {tail_share:.1%} de l’erreur quadratique.')
display(claim_errors[['index','observed','montant_pred','pred','residu']].head(12))
segment_rows=[]
for field in ['type_contrat','utilisation','conducteur2']:
    for level,idx in train.groupby(field,dropna=False).groups.items():
        z=oof.loc[idx]
        segment_rows.append({'variable':field,'modalite':str(level),'contrats':len(z),'sinistres':int(z.claim.sum()),
            'RMSE_principal':scores(z.observed,z.pred)['RMSE'],'RMSE_precedent':scores(z.observed,z.incumbent)['RMSE'],
            'MAE_principal':scores(z.observed,z.pred)['MAE'],'biais_EUR':float((z.pred-z.observed).mean())})
segments=pd.DataFrame(segment_rows);segments.to_csv(OUT/'performances_segments.csv',index=False);display(segments)
fig,axes=plt.subplots(1,2,figsize=(12,4.3))
z=e.loc[e.claim.eq(1)]
axes[0].scatter(z.observed,z.montant_pred,s=9,alpha=.25)
lim=max(z.observed.max(),z.montant_pred.max());axes[0].plot([500,lim],[500,lim],'--',color='grey')
axes[0].set(xscale='log',yscale='log',xlabel='Montant observé du sinistre (€), log',ylabel='Gravité conditionnelle prédite (€), log',title='Gravité : clients sinistrés hors apprentissage')
q=segments.loc[segments.variable.eq('type_contrat')];x=np.arange(len(q))
axes[1].bar(x-.18,q.RMSE_precedent,width=.36,label='Recette précédente')
axes[1].bar(x+.18,q.RMSE_principal,width=.36,label='Modèle principal')
axes[1].set_xticks(x,q.modalite);axes[1].set(ylabel='RMSE de prime (€)',title='Erreurs par type de contrat');axes[1].legend()
figure(fig,'05_erreurs_segments.png')
''')

md('''
## 8. Importance des groupes de variables sur des clients exclus

Dans chaque pli, nous permutons ensemble les colonnes d'un groupe, trois fois,
sans réapprendre le modèle. La hausse de MSE de prime mesure la sensibilité à cette
perturbation. Une hausse positive indique une information utile dans ce diagnostic.
L'écart-type représente seulement les trois permutations, pas une incertitude
complète d'apprentissage. Une importance faible n'établit pas l'inutilité d'un
champ : les variables peuvent être redondantes. Les permutations peuvent aussi
créer des véhicules ou contrats peu réalistes. La fin de vente sert de contrôle
négatif puisqu'elle n'est utilisée par aucune composante.
''')
code('''
importance=pd.read_csv(OUT/'permutation_importance.csv').sort_values('mean')
display(importance.sort_values('mean',ascending=False))
fig,ax=plt.subplots(figsize=(10,5.3))
ax.barh(importance.group,importance['mean'],xerr=importance['std'].fillna(0),color='#2563eb',capsize=3)
ax.axvline(0,color='grey',lw=1)
ax.set(xlabel='Hausse de MSE de prime après permutation (€²)',ylabel='Groupe de prédicteurs',title='Importance par perturbation — clients hors apprentissage')
figure(fig,'06_importance.png')
''')

md('''
### Courbes du modèle pour un profil fixé

Nous faisons varier l'âge du véhicule et le type de contrat pour le profil du
premier prospect, en gardant ses autres valeurs. Ces courbes illustrent le calcul
du modèle ; elles ne sont ni des effets causaux ni la moyenne de tous les assurés.
La plage d'âge reste dans le domaine observé entre les quantiles 1 % et 99 %.
''')
code('''
ages=np.linspace(train.anciennete_vehicule.quantile(.01),train.anciennete_vehicule.quantile(.99),90)
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for contract in sorted(train.type_contrat.unique()):
    profile=pd.concat([test.iloc[[0]]]*len(ages),ignore_index=True)
    profile['anciennete_vehicule']=ages;profile['type_contrat']=contract
    prediction=model.details(profile)
    axes[0].plot(ages,100*prediction.proba_pred,label=contract)
    axes[1].plot(ages,prediction.montant_pred,label=contract)
axes[0].set(xlabel='Ancienneté du véhicule (unité du fichier)',ylabel='Probabilité prédite (%)',title='Fréquence selon âge et couverture — profil fixé')
axes[1].set(xlabel='Ancienneté du véhicule (unité du fichier)',ylabel='Gravité conditionnelle prédite (€)',title='Gravité selon âge et couverture — profil fixé')
for ax in axes:ax.legend(title='Type de contrat')
figure(fig,'07_courbes_profil.png')
''')

md('''
## 9. Matrices de confusion : fréquence uniquement

Une matrice de confusion exige un seuil. Le seuil 0,50 peut prédire « sans
sinistre » pour presque tout le monde, avec une exactitude trompeuse dans un jeu
à 5,834 % de positifs. Le second seuil est le taux global d'apprentissage, présenté
uniquement pour comprendre le compromis : il n'est pas sélectionné comme décision
optimale. **Ces seuils ne servent jamais à calculer la soumission.** La gravité est
une régression et n'a pas de matrice de confusion pertinente pour ses euros.
''')
code('''
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,threshold in zip(axes,[.5,float(train.nombre_sinistres.mean())]):
    matrix=confusion_matrix(oof.claim,oof.proba_pred.ge(threshold),labels=[0,1])
    ConfusionMatrixDisplay(matrix,display_labels=['Sans sinistre','Sinistre']).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d')
    ax.set(title=f'Fréquence hors apprentissage — seuil {threshold:.4f}',xlabel='Classe prédite',ylabel='Classe observée')
figure(fig,'08_confusion.png')
''')

md('''
## 10. Forces, faiblesses et ce qu'il faudrait améliorer

**Forces :** décomposition fréquence × gravité cohérente ; plis et encodages par
client ; ancre Gamma et corrections régularisées ; coûts extrêmes conservés ;
prédictions positives ; recette et fichier évalué identifiés explicitement.

**Faiblesses :** gain très faible ; peu de sinistres pour de nombreuses catégories ;
forte influence de quelques gros coûts ; modèle composite moins simple à expliquer
qu'un GLM ; MAE plus élevée malgré une baisse de RMSE ; certains segments légèrement
moins bons. Les données et le leaderboard ont déjà été étudiés : l'intervalle
bootstrap descriptif ne couvre pas toute l'incertitude ni les choix humains.

Le Gamma a toujours trois variables, mais **le modèle complet en utilise davantage**.
Le département est présent dans la fréquence et la Ridge. La base historique lit
deux caractères pour le département ; avant un usage sur d'autres zones, notamment
les DOM, cette convention devra être harmonisée avec le traitement métier récent.

**Priorités :** documenter exposition, dates et unités ; réserver une nouvelle
cohorte finale ; disposer de plus de sinistres ; suivre calibration, valeurs
manquantes et erreurs par segment. Le modèle prédit une moyenne, pas une indemnité
garantie ni un intervalle fiable pour chaque assuré.

L'écart entre score public et RMSE locale ne permet pas d'identifier une cause
sans les coûts du test et son échantillonnage. Le score privé n'est pas connu et
la cible historique strictement inférieure à 1365,500 n'est pas atteinte.
''')

md('''
## 11. Code d'assemblage et contrôles de livraison

Les objets suivants montrent l'ordre exact des apprentissages et des mélanges.
Le CSV principal doit être celui du dernier modèle, pas un export d'une ancienne
recherche. Le contrôle distingue l'empreinte brute, le contenu normalisé LF et
l'empreinte CRLF du reçu Windows. Les chiffres ne sont pas modifiés pour faire
coïncider artificiellement ces empreintes.
''')
code('''
for obj in [f.previous.Premium,f.old.Premium,f.vm.Premium,f.Premium]:
    display(Markdown('**'+obj.__module__+'.'+obj.__name__+'**'));display(Code(inspect.getsource(obj),language='python'))
verification=main.verify_main()
submitted=main.release();display(submitted.head())
summary={'recipe':main.RECIPE,'diagnostic':diagnostic,'tail_top_1pct_claims_SSE_share':float(tail_share),
    'permutation_importance':importance.to_dict(orient='records'),'verification':verification,
    'main_notebook':'modele_principal.ipynb','archives':'archive/'}
main.write_json(OUT/'analyse.json',summary)
''')

md('''
## MODÈLE FINAL RETENU

**Fréquence : logistique C = 0,1 + correction de logit âge/couverture/usage,
alpha = 100, poids = 1.**

**Gravité : 37,5 % Gamma alpha = 0,1 + 37,5 % correction XGBoost (240 arbres,
profondeur 2) + 25 % correction Ridge sur variables étendues, alpha = 10.**

**Soumission : `submission.csv`, 50 000 lignes `index,pred`.
Modèle local : `models/main_model.joblib`.**

Le score public rapporté de 1365,657 reste associé au fichier soumis. La recherche
historique gagne environ 0,05546 € de RMSE locale ; ce progrès marginal n'établit
pas une supériorité universelle ou sur le leaderboard privé.

Pour reproduire : `python run.py train`, `python run.py diagnostic`,
`python run.py verify`, puis `python run.py predict --output reports/main/submission_reproduite.csv`.
Les anciennes recherches sont accessibles dans `archive/` et ne sont plus le modèle actif.

Lecture détaillée : [analyse du modèle](docs/modele_principal.md).
Références : [Gamma](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html),
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[splines](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
''')

nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3.12 — modèle principal','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
dest=ROOT/'modele_principal.ipynb';nbf.write(nb,dest);print(dest,len(cells),'cellules')
