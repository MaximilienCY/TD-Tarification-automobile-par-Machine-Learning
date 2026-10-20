"""Reference models migrated from the original notebooks, without execution side effects."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression, GammaRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
SEED = 42
RAW_FEATURES=['bonus','type_contrat','duree_contrat','anciennete_info','freq_paiement','paiement','utilisation',
    'code_postal','conducteur2','age_conducteur1','age_conducteur2','anciennete_permis1','anciennete_permis2',
    'anciennete_vehicule','cylindre_vehicule','din_vehicule','essence_vehicule','marque_vehicule','modele_vehicule',
    'debut_vente_vehicule','fin_vente_vehicule','vitesse_vehicule','type_vehicule','prix_vehicule','poids_vehicule']
LOW_CATS=['type_contrat','freq_paiement','paiement','utilisation','conducteur2','essence_vehicule','type_vehicule']
NUM_NATIVE=['bonus','duree_contrat','anciennete_info','age_conducteur1','age_conducteur2','anciennete_permis1',
    'anciennete_permis2','anciennete_vehicule','cylindre_vehicule','din_vehicule','prix_vehicule','poids_vehicule',
    'vitesse_vehicule','debut_vente_vehicule','fin_vente_vehicule','puissance_par_tonne']
NUM_COMPACT=['log_bonus','duree_contrat','anciennete_info','age_conducteur1','anciennete_permis2',
    'anciennete_vehicule','log_din','log_poids','log_cylindre']

def preparer_metier(frame):
    needed=['bonus','type_contrat','utilisation','anciennete_vehicule','age_conducteur1',
        'poids_vehicule','din_vehicule','cylindre_vehicule','prix_vehicule','code_postal']
    absent=sorted(set(needed)-set(frame.columns))
    if absent:raise ValueError(f'Colonnes nécessaires absentes : {absent}')
    x=frame.loc[:,[c for c in RAW_FEATURES if c in frame]].copy()
    for col in ['poids_vehicule','cylindre_vehicule']:
        x[col]=x[col].mask(x[col]<=0)
    # Zéros du second conducteur conservés : son absence est une information structurelle.
    s=x.code_postal.astype('string').str.strip().str.upper()
    x['departement']=s.str[:2].where(s.str.fullmatch(r'(?:0[1-9]|1[0-9]|2[1-9]|[3-8][0-9]|9[0-5]|2A|2B)[0-9]{3}',na=False))
    dom=s.str.fullmatch(r'97[1-46][0-9]{2}',na=False)
    x.loc[dom,'departement']=s.loc[dom].str[:3]
    for dest,src in [('log_bonus','bonus'),('log_din','din_vehicule'),('log_poids','poids_vehicule'),
                     ('log_cylindre','cylindre_vehicule'),('log_prix','prix_vehicule')]:
        x[dest]=np.log1p(x[src].mask(x[src]<0))
    x['puissance_par_tonne']=x.din_vehicule/(x.poids_vehicule/1000)
    x['contrat_usage']=x.type_contrat.astype('string').fillna('__NA__')+'|'+x.utilisation.astype('string').fillna('__NA__')
    x['age_maxi']=x.anciennete_vehicule*(x.type_contrat=='Maxi')
    x['prix_maxi']=x.log_prix*(x.type_contrat=='Maxi')
    for col in x.select_dtypes(include=['object','string']).columns:
        x[col]=x[col].astype(object).where(x[col].notna(),np.nan)
    return x

def construire_preparation(numeriques,categorielles,splines=(),min_frequency=50):
    splines=list(splines)
    lineaires=[c for c in numeriques if c not in splines]
    parts=[]
    if lineaires:
        parts.append(('lineaire',Pipeline([('imputation',SimpleImputer(strategy='median',add_indicator=True)),
            ('echelle',StandardScaler())]),lineaires))
    if splines:
        parts.append(('courbes',Pipeline([('imputation',SimpleImputer(strategy='median')),
            ('splines',SplineTransformer(n_knots=4,degree=2,knots='quantile',include_bias=False,extrapolation='constant')),
            ('echelle',StandardScaler())]),splines))
    if categorielles:
        parts.append(('categories',Pipeline([('imputation',SimpleImputer(strategy='constant',fill_value='__NA__')),
            ('encodage',OneHotEncoder(min_frequency=min_frequency,handle_unknown='infrequent_if_exist',sparse_output=False))]),list(categorielles)))
    return ColumnTransformer(parts,remainder='drop')
VARIABLES_FREQUENCE=RAW_FEATURES
REDONDANTES = ['debut_vente_vehicule', 'fin_vente_vehicule', 'anciennete_permis1',
               'age_conducteur2', 'prix_vehicule', 'vitesse_vehicule']
A_IMPUTER = ['poids_vehicule', 'cylindre_vehicule', 'anciennete_vehicule']
QUEUE_LOURDE = ['bonus', 'din_vehicule', 'poids_vehicule', 'cylindre_vehicule']
CAT_FAIBLE = ['type_contrat', 'freq_paiement', 'paiement', 'utilisation',
              'conducteur2', 'essence_vehicule', 'type_vehicule']
CAT_FORTE = ['marque_vehicule', 'modele_vehicule', 'departement']
SEUIL_RARE, LISSAGE, C_FREQUENCE = 50, 100, 0.1

def nettoyer_frequence(frame):
    manque = sorted(set(VARIABLES_FREQUENCE) - set(frame.columns))
    if manque:
        raise ValueError(f'Colonnes de fréquence absentes : {manque}')
    # Sélection explicite : les cibles présentes dans train ne peuvent pas entrer.
    x = frame.loc[:, VARIABLES_FREQUENCE].copy()
    for col in ['poids_vehicule', 'cylindre_vehicule']:
        x[col] = x[col].replace(0, np.nan)
    x['departement'] = x['code_postal'].astype('string').str[:2]
    x = x.drop(columns='code_postal')
    for col in CAT_FAIBLE + CAT_FORTE:
        x[col] = x[col].astype(object).where(x[col].notna(), '__MANQUANT__')
    return x

def taux_lisse(x, y, prior):
    stats = y.groupby(x).agg(['sum', 'count'])
    return (stats['sum'] + LISSAGE * prior) / (stats['count'] + LISSAGE)

def encoder_hors_clients(x, y, groupes):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    encodage = pd.DataFrame(np.nan, index=x.index, columns=CAT_FORTE)
    audit = []
    for k, (a, b) in enumerate(cv.split(x, y, groups=groupes), 1):
        assert set(groupes.iloc[a]).isdisjoint(groupes.iloc[b])
        prior = float(y.iloc[a].mean())
        for col in CAT_FORTE:
            mapping = taux_lisse(x.iloc[a][col], y.iloc[a], prior)
            encodage.loc[x.index[b], col] = x.iloc[b][col].map(mapping).fillna(prior).to_numpy()
        audit.append({'pli': k, 'contrats_apprentissage': len(a),
            'contrats_encodés': len(b), 'clients_communs': 0, 'prior_autres_plis': prior})
    assert np.isfinite(encodage.to_numpy()).all()
    return encodage, pd.DataFrame(audit)

def transformer_frequence(frame, p, encodage=None):
    x = nettoyer_frequence(frame).drop(columns=REDONDANTES)
    for col in A_IMPUTER:
        x[col] = (x[col].fillna(x.modele_vehicule.map(p[f'med_modele_{col}']))
            .fillna(x.marque_vehicule.map(p[f'med_marque_{col}']))
            .fillna(p[f'med_globale_{col}']))
    for col in QUEUE_LOURDE:
        if (x[col] < 0).any():
            raise ValueError(f'Valeur négative incompatible avec log1p : {col}')
        x[col] = np.log1p(x[col])
    for col in CAT_FORTE:
        if encodage is not None:
            assert encodage.index.equals(x.index)
            x[col] = encodage[col]
        else:
            x[col] = x[col].map(p[f'taux_{col}']).fillna(p['taux_moyen'])
    for col in CAT_FAIBLE:
        x[col] = x[col].where(x[col].isin(p[f'frequentes_{col}']), 'Autre')
    x = pd.get_dummies(x, columns=CAT_FAIBLE, dtype=int)
    x = x.drop(columns=[f'{col}_{p[f"reference_{col}"]}' for col in CAT_FAIBLE], errors='ignore')
    if 'colonnes' in p:
        x = x.reindex(columns=p['colonnes'], fill_value=0)
    return x.astype(float)

def standardiser_frequence(x, p):
    x = x.copy()
    cols = p['colonnes_num']
    x[cols] = (x[cols] - p['moyennes']) / p['ecarts_types']
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Préparation de fréquence non finie.')
    return x

def apprendre_frequence(frame, y, groupes):
    assert frame.index.is_unique and frame.index.equals(y.index)
    assert frame.index.equals(groupes.index)
    x = nettoyer_frequence(frame)
    p = {'taux_moyen': float(y.mean())}
    for col in A_IMPUTER:
        p[f'med_modele_{col}'] = x.groupby('modele_vehicule')[col].median()
        p[f'med_marque_{col}'] = x.groupby('marque_vehicule')[col].median()
        p[f'med_globale_{col}'] = float(x[col].median())
    for col in CAT_FAIBLE:
        counts = x[col].value_counts()
        p[f'frequentes_{col}'] = counts[counts >= SEUIL_RARE].index
        p[f'reference_{col}'] = counts.index[0]
    for col in CAT_FORTE:
        p[f'taux_{col}'] = taux_lisse(x[col], y, p['taux_moyen'])
    oof, audit = encoder_hors_clients(x, y, groupes)
    z = transformer_frequence(frame, p, encodage=oof)
    p['colonnes'] = z.columns
    p['colonnes_num'] = [col for col in z if z[col].nunique() > 2]
    p['moyennes'] = z[p['colonnes_num']].mean()
    p['ecarts_types'] = z[p['colonnes_num']].std().replace(0, 1)
    return p, standardiser_frequence(z, p), audit

def preparer_frequence(frame, p):
    # Validation / test : aucun y, toutes les statistiques viennent de p.
    return standardiser_frequence(transformer_frequence(frame, p), p)
class ReferenceFrequence:
    def fit(self,X,y,groups):
        y=pd.Series(np.asarray(y),index=X.index)
        groups=pd.Series(np.asarray(groups),index=X.index)
        self.params_,z,self.audit_=apprendre_frequence(X,y,groups)
        self.model_=LogisticRegression(C=.1,max_iter=3000)
        self.model_.fit(z,y)
        return self
    def predict_proba(self,X):return self.model_.predict_proba(preparer_frequence(X,self.params_))

class LogistiqueMetier:
    def __init__(self,config):self.config=config
    def fit(self,X,y,groups=None):
        c=self.config
        nums=NUM_COMPACT.copy()
        cats=LOW_CATS.copy()
        if c.get('prix'):nums+=['log_prix','puissance_par_tonne']
        if c.get('interaction'):nums+=['age_maxi','prix_maxi'];cats+=['contrat_usage']
        if c.get('geographie'):cats+=['departement']
        if c.get('vehicules'):cats+=['marque_vehicule','modele_vehicule']
        self.prep_=construire_preparation(nums,cats,['anciennete_vehicule','age_conducteur1','duree_contrat'] if c.get('splines') else [],c.get('rare',50))
        z=self.prep_.fit_transform(preparer_metier(X))
        self.model_=LogisticRegression(C=c['C'],max_iter=4000,tol=1e-5)
        self.model_.fit(z,y)
        return self
    def predict_proba(self,X):return self.model_.predict_proba(self.prep_.transform(preparer_metier(X)))

class BoostingMetier:
    def __init__(self,config):self.config=config
    def _frame(self,X,fit=False):
        x=preparer_metier(X)
        cats=LOW_CATS+['departement','marque_vehicule','modele_vehicule','contrat_usage']
        x=x.loc[:,NUM_NATIVE+cats].copy()
        if fit:self.categories_={}
        for col in cats:
            s=x[col].fillna('__NA__').astype(str)
            if fit:
                vc=s.value_counts()
                self.categories_[col]=list(vc[vc>=self.config.get('rare',100)].index[:200])
            keep=self.categories_[col]
            s=s.where(s.isin(keep),'__RARE__')
            x[col]=pd.Categorical(s,categories=keep+['__RARE__'])
        return x
    def fit(self,X,y,groups=None):
        c=self.config
        self.model_=HistGradientBoostingClassifier(learning_rate=c['lr'],max_leaf_nodes=c['leaves'],
            max_iter=c['iterations'],min_samples_leaf=c['min_leaf'],l2_regularization=c['l2'],
            categorical_features='from_dtype',early_stopping=False,random_state=42)
        self.model_.fit(self._frame(X,fit=True),y)
        return self
    def predict_proba(self,X):return self.model_.predict_proba(self._frame(X))

def logit_safe(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

class FrequenceCalibree:
    def __init__(self,config):self.config=config
    def fit(self,X,y,groups):
        y=np.asarray(y);groups=np.asarray(groups)
        oof=np.full(len(X),np.nan)
        cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=1803)
        for a,b in cv.split(X,y,groups=groups):
            assert set(groups[a]).isdisjoint(groups[b])
            m=BoostingMetier(self.config['base']).fit(X.iloc[a],y[a],groups[a])
            oof[b]=m.predict_proba(X.iloc[b])[:,1]
        assert np.isfinite(oof).all()
        self.calibration_=LogisticRegression(C=10,max_iter=1000)
        self.calibration_.fit(logit_safe(oof),y)
        self.base_=BoostingMetier(self.config['base']).fit(X,y,groups)
        return self
    def predict_proba(self,X):return self.calibration_.predict_proba(logit_safe(self.base_.predict_proba(X)[:,1]))

class MelangeFrequence:
    def __init__(self,config):self.config=config
    def fit(self,X,y,groups):
        self.models_=[creer_frequence(c).fit(X,y,groups) for c in self.config['membres']]
        return self
    def predict_proba(self,X):
        w=self.config['poids']
        p=w*self.models_[0].predict_proba(X)[:,1]+(1-w)*self.models_[1].predict_proba(X)[:,1]
        return np.column_stack([1-p,p])


class GammaMetier:
    def __init__(self,config):self.config=config
    def _spec(self):
        c=self.config
        nums=['anciennete_vehicule'];cats=['type_contrat','utilisation']
        if c.get('compact'):nums+=['log_prix','age_conducteur1','log_din'];cats+=['conducteur2','type_vehicule']
        if c.get('interaction'):nums+=['age_maxi','prix_maxi'];cats+=['contrat_usage']
        if c.get('geo'):cats+=['departement']
        if c.get('vehicules'):cats+=['marque_vehicule','modele_vehicule']
        if c.get('target'):nums+=['cout_modele_lisse','cout_marque_lisse']
        splines=['anciennete_vehicule','log_prix','age_conducteur1'] if c.get('splines') else []
        return nums,cats,splines
    def _mappings(self,X,y,prior):
        maps={}
        for col in ['modele_vehicule','marque_vehicule']:
            stats=pd.Series(np.asarray(y),index=X.index).groupby(X[col].fillna('__NA__')).agg(['sum','count'])
            maps[col]=(stats['sum']+self.config['lissage']*prior)/(stats['count']+self.config['lissage'])
        return maps
    def _encode(self,X,maps,prior):
        out=pd.DataFrame(index=X.index)
        for col,dest in [('modele_vehicule','cout_modele_lisse'),('marque_vehicule','cout_marque_lisse')]:
            out[dest]=X[col].fillna('__NA__').map(maps[col]).fillna(prior)
        return out
    def fit(self,X,y,groups):
        y=np.asarray(y,dtype=float);groups=np.asarray(groups)
        assert (y>0).all() and len(X)==len(groups)
        x=preparer_metier(X)
        if self.config.get('target'):
            self.prior_=float(y.mean());self.maps_=self._mappings(x,y,self.prior_)
            enc=pd.DataFrame(np.nan,index=x.index,columns=['cout_modele_lisse','cout_marque_lisse'])
            inner=GroupKFold(n_splits=3,shuffle=True,random_state=2303)
            for a,b in inner.split(x,y,groups=groups):
                assert set(groups[a]).isdisjoint(groups[b])
                prior=float(y[a].mean());maps=self._mappings(x.iloc[a],y[a],prior)
                enc.loc[x.index[b]]=self._encode(x.iloc[b],maps,prior).to_numpy()
            x=x.join(enc)
        nums,cats,splines=self._spec()
        self.prep_=construire_preparation(nums,cats,splines,self.config.get('rare',20))
        z=self.prep_.fit_transform(x)
        self.model_=GammaRegressor(alpha=self.config['alpha'],solver='lbfgs',max_iter=5000,tol=1e-5)
        self.model_.fit(z,y)
        return self
    def predict(self,X):
        x=preparer_metier(X)
        if self.config.get('target'):x=x.join(self._encode(x,self.maps_,self.prior_))
        p=self.model_.predict(self.prep_.transform(x))
        if not np.isfinite(p).all() or not (p>0).all():raise ValueError('Montants Gamma invalides.')
        return p


