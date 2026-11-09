"""Document the observed model history for a 25-30 minute presentation.

No training or prediction changes. Historical evidence is read from Git and
archived reports. The JSON feeds an editable presentation and its oral guide.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/presentation_histoire"
BUILD = ROOT / "tmp/presentation_histoire"
OUT.mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)
PROOFS = OUT / "sources_historiques"
PROOFS.mkdir(exist_ok=True)

sources = {
    "td": {"path": "../td_data_science.pdf", "label": "Énoncé du TD, pages 1 à 4"},
    "roadmap": {"path": "deliverables/presentation_histoire/assets/roadmap_reference.png", "label": "Roadmap de projet fournie par l’utilisateur"},
    "first": {"path": "Git 0a3a4a5 : modele_gravite.ipynb", "label": "Première gravité, 9 octobre 2026"},
    "gamma": {"path": "Git 5310163 : output/gravite/configuration_gravite.json", "label": "Sélection Gamma à trois variables"},
    "friend": {"path": "Git 167019d : frequence_regression_logisitique.ipynb", "label": "Notebook original de fréquence du partenaire"},
    "audit": {"path": "Git 566e1b5 : output/frequence/analyse/resume_analyse_frequence.json", "label": "Audit des plis d’encodage du partenaire"},
    "merge": {"path": "Git 566e1b5 : output/prime/configuration_prime.json", "label": "Première combinaison logistique × Gamma"},
    "v4": {"path": "Git 566e1b5 : output/optimisation/prime/configuration_finale.json", "label": "Optimisation de 391 paires"},
    "broad": {"path": "archive/docs/results.md", "label": "Recherche CatBoost, LightGBM et Gamma"},
    "modern": {"path": "archive/docs/research_round2.md", "label": "EBM, XGBoost et benchmark TabPFN"},
    "anchor": {"path": "archive/docs/anchored_model.md", "label": "Correction limitée de fréquence"},
    "restart": {"path": "archive/docs/restart_results.md", "label": "Reprise XGBoost complète"},
    "refine": {"path": "archive/docs/refinement_after_kaggle.md", "label": "Recentrage sur la gravité et les pondérations"},
    "bag": {"path": "archive/docs/severity_successor.md", "label": "Successeur à trois graines, finalement écarté"},
    "ridge": {"path": "archive/docs/variable_reanalysis.md", "label": "Réanalyse des résidus et correction Ridge"},
    "follow": {"path": "archive/docs/variable_followups.md", "label": "Courbes, interactions et correction finale de fréquence"},
    "registry": {"path": "reports/kaggle_submissions.json", "label": "Registre des soumissions et statut des scores"},
    "main": {"path": "docs/modele_principal.md", "label": "Analyse du modèle principal actuel"},
    "diag": {"path": "reports/main/diagnostic.json", "label": "Diagnostic réexécuté de la recette fixe"},
    "analysis": {"path": "reports/main/analyse.json", "label": "Importance des variables, queue et vérifications"},
    "vars": {"path": "reports/main/variables_par_composante.csv", "label": "Inventaire exact des variables par composante"},
    "model": {"path": "reports/final_model.json", "label": "Paramètres et provenance du modèle final"},
    "skgamma": {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html", "label": "Documentation officielle GammaRegressor"},
    "skenc": {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html", "label": "Documentation officielle sur le cross-fitting d’encodage cible"},
    "skcv": {"url": "https://scikit-learn.org/stable/modules/cross_validation.html", "label": "Documentation officielle sur la validation croisée"},
    "skspline": {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html", "label": "Documentation officielle SplineTransformer"},
    "xgb": {"url": "https://xgboost.readthedocs.io/en/stable/parameter.html", "label": "Documentation officielle des paramètres XGBoost"},
}


def git_text(rev, path):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "show", f"{rev}:{path}"], cwd=ROOT, text=True)


proof_specs = [
    ("0a3a4a5", "output/gravite/comparaison_cv.csv", "premiere_gravite_cv.csv"),
    ("0a3a4a5", "output/gravite/scores_validation.csv", "premiere_gravite_validation.csv"),
    ("5310163", "output/gravite/selection_variables_gamma.csv", "selection_variables_gamma.csv"),
    ("5310163", "output/gravite/configuration_gravite.json", "gamma_3_variables.json"),
    ("566e1b5", "output/frequence/analyse/resume_analyse_frequence.json", "audit_frequence_partenaire.json"),
    ("566e1b5", "output/prime/configuration_prime.json", "premiere_combinaison.json"),
    ("566e1b5", "output/optimisation/prime/configuration_finale.json", "premiere_optimisation_391_paires.json"),
]
proof_manifest = []
for rev, path, name in proof_specs:
    value = git_text(rev, path)
    (PROOFS / name).write_text(value, encoding="utf-8")
    proof_manifest.append({"commit": rev, "source_path": path, "copy": name,
                           "sha256_utf8": hashlib.sha256(value.encode()).hexdigest()})

slides = []


def add(title, stage, kind="columns", *, lead="", left=None, right=None, rows=None,
        chart=None, formula=None, takeaway="", notes="", refs=(), seconds=50,
        appendix=False, **extra):
    slides.append(dict(number=len(slides)+1, title=title, stage=stage, kind=kind,
                       lead=lead, left=left or [], right=right or [], rows=rows or [],
                       chart=chart, formula=formula, takeaway=takeaway, notes=notes,
                       refs=list(refs), seconds=0 if appendix else seconds,
                       appendix=appendix, **extra))


add("L’histoire de notre modèle de tarification", "Introduction", "cover",
    lead="Comprendre les données, comparer les essais et justifier le modèle retenu",
    formula="Prime pure = probabilité de sinistre × montant moyen si sinistre",
    takeaway="Du premier Gamma au meilleur score public observé : 1 365,657",
    notes="Nous présentons une démarche de data science et pas seulement un algorithme final. Au départ, nous travaillions sur la gravité tandis que notre partenaire construisait la fréquence. Nous avons ensuite assemblé les deux modèles et comparé plusieurs améliorations. Le support raconte aussi les tentatives qui n’ont pas donné le résultat attendu. Le terme meilleur désigne ici le meilleur score public observé parmi nos essais documentés. Il ne prouve ni un optimum universel, ni une supériorité sur le classement privé. Les chiffres présentés viennent des notebooks, de l’historique Git et des rapports conservés. Les annexes permettent de revenir sur chaque variable et sur les réglages précis pendant les questions.",
    refs=("td", "registry"), seconds=35)

add("La roadmap et les retours de décision", "Plan", "roadmap",
    lead="Le projet suit cinq étapes, avec des retours quand les résultats remettent un choix en cause",
    steps=["Cadrage", "Analyse des données", "Traitement", "Modélisation", "Validation"],
    returns=["Revoir une variable : département, prix, modèle automobile", "Limiter une correction quand la complexité n’apporte pas de gain", "Revenir à la référence après un résultat Kaggle moins bon"],
    takeaway="Chaque itération relie une observation, une hypothèse, une expérience et une décision",
    notes="La roadmap fournie nous sert de plan. Le cadrage fixe ce que nous prédisons et la métrique. L’analyse décrit les cibles, les anomalies et les relations. Le traitement traduit ces observations en transformations réutilisables. La modélisation compare les familles et les variables. La validation mesure l’erreur et la cohérence métier. Les flèches de retour sont essentielles à notre histoire. Nous avons par exemple retiré le département du premier Gamma, puis réintroduit la géographie dans une autre composante. Nous avons aussi abandonné plusieurs modèles plus complexes et conservé une référence stable. Les reprises de décision ne sont pas des contradictions : elles répondent à une nouvelle hypothèse testée dans un contexte différent.",
    refs=("roadmap", "gamma", "ridge", "follow"), seconds=45)

add("Le cadrage actuariel", "Cadrage", "formula",
    lead="Le coût réel est nul sans sinistre et positif lorsqu’un sinistre est observé",
    formula="E[S | X] = P(S > 0 | X) × E[S | S > 0, X]",
    left=[("Fréquence", "Apprendre une probabilité sur tous les contrats"),
          ("Gravité", "Apprendre un montant sur les seuls sinistres positifs")],
    right=[("Critère final", "RMSE de prime sur tous les contrats"),
           ("Périmètre", "Coût attendu de la période, sans annualisation supposée")],
    takeaway="La prime pure ne contient ni frais, ni marge, ni taxes",
    notes="La fréquence répond à la question : quelle est la probabilité d’avoir au moins un sinistre ? Dans nos données, nombre_sinistres ne prend que zéro ou un. La gravité répond à une autre question : quel montant moyen prévoir si un sinistre survient ? Nous ne savons pas, pour les prospects, qui aura effectivement un accident. Nous calculons donc les deux sorties pour chacun, puis leur produit. Appliquer un seuil de classification avant cette multiplication supprimerait une partie du risque attendu. Nous retenons la RMSE finale car elle pénalise les gros écarts et vise une moyenne conditionnelle. La décomposition porte sur la période observée. Faute de dictionnaire d’exposition et de dates claires, nous ne présentons pas ce montant comme un tarif annuel commercial.",
    refs=("td", "main"), seconds=50)

add("Les fichiers et l’unité d’observation", "Analyse des données", "table",
    lead="Une ligne correspond à un contrat associé à un véhicule, pas toujours à un client unique",
    rows=[["Fichier", "Contenu", "Utilisation"],
          ["train.csv", "50 000 contrats, 33 colonnes", "Apprendre et valider"],
          ["test.csv", "50 000 prospects, 28 colonnes", "Prédire, sans connaître les coûts"],
          ["Exemple de soumission", "index, pred", "Contrôler le format et l’ordre"]],
    left=[("45 785 clients dans le train", "Un client peut avoir plusieurs contrats")],
    takeaway="L’identifiant client définit les groupes de validation et ne sert jamais de prédicteur",
    notes="La première distinction porte sur le rôle des fichiers. Le train contient les résultats observés et permet de construire une validation locale. Le test de compétition ne contient aucune cible connue et ne permet pas de calculer une RMSE locale. Il manque également les trois identifiants client, véhicule et contrat. Le champ index reste disponible et permet d’aligner les prédictions. Il va de 50 000 à 99 999 dans le test. Les 50 000 contrats d’apprentissage appartiennent à 45 785 clients, ce qui rend un découpage naïf par ligne dangereux. La présence d’une variable dans le test est un indice de disponibilité, mais ne prouve pas à elle seule qu’elle était connue avant le sinistre. Ce point devra être confirmé par un dictionnaire temporel.",
    refs=("td", "main", "merge"), seconds=50)

add("Des sinistres rares et des montants asymétriques", "Analyse des données", "target",
    lead="2 917 sinistres positifs, soit 5,834 % des contrats",
    left=[("Fréquence", "94,166 % de contrats sans sinistre"),
          ("Accuracy trompeuse", "Tout prédire à zéro donne déjà environ 94 %")],
    right=[("Montant médian", "1 236,00 €"), ("Montant moyen", "1 771,84 €"),
           ("Maximum", "21 826,96 €")],
    takeaway="La moyenne dépasse la médiane et quelques sinistres pèsent très lourd dans l’erreur quadratique",
    notes="Les deux sous-problèmes ont des difficultés différentes. Pour la fréquence, la classe positive est rare. Une accuracy élevée peut donc cacher un modèle qui ne détecte rien et ne sait pas estimer les probabilités. Nous suivons la log-loss, le Brier et l’AUC. Pour la gravité, nous n’avons que 2 917 exemples, malgré un train de 50 000 contrats. La distribution est étirée vers les grands montants : sa moyenne est supérieure à sa médiane. Les gros sinistres sont plausibles et ils comptent fortement dans la métrique officielle. Nous les conservons à l’apprentissage. Le logarithme sert parfois à visualiser la distribution, mais cela ne signifie pas que nous avons transformé la cible du Gamma en logarithme avant le fit.",
    refs=("first", "friend", "main"), seconds=55)

add("Les anomalies deviennent des règles de préparation", "Analyse des données", "table",
    lead="Nous distinguons information absente, absence structurelle et valeur extrême plausible",
    rows=[["Observation", "Interprétation", "Décision"],
          ["1 883 poids nuls et 3 cylindrées nulles", "Mesures manquantes probables", "NaN puis imputation apprise"],
          ["33 406 contrats sans second conducteur", "Âges et permis nuls structurels", "Conserver l’indicateur d’absence"],
          ["Ancienneté de permis supérieure à l’âge", "Incohérence possible", "Signaler, traiter selon la composante"],
          ["Gros sinistres et véhicules chers", "Extrêmes potentiellement plausibles", "Conserver les observations"]],
    takeaway="Les règles restent fixes, les médianes et les catégories s’apprennent seulement dans le train autorisé",
    notes="Un zéro n’a pas toujours le même sens. Un poids de véhicule nul est probablement une mesure absente. En revanche, un âge nul quand aucun second conducteur n’est déclaré représente une absence structurelle. La première préparation de fréquence conserve ces zéros avec l’indicateur conducteur2. Les courbes de la correction Ridge utilisent ensuite des valeurs manquantes pour ces âges et permis, tout en conservant explicitement l’absence. Les incohérences de permis sont signalées, sans prétendre connaître leur vraie valeur. Il existe une valeur manquante d’ancienneté du véhicule dans le train. Nous séparons ces règles métier des statistiques apprises : le test reçoit les transformations et les médianes apprises sur le train. Nous ne supprimons pas les grands coûts pour améliorer artificiellement la RMSE.",
    refs=("friend", "ridge", "main"), seconds=55)

add("Corrélations et différences entre profils", "Analyse des données", "correlations",
    lead="Les associations guident des hypothèses, puis la validation décide de leur utilité",
    rows=[["Paire de variables numériques", "Spearman"],
          ["Âge véhicule / début de vente", "0,961"],
          ["Âge véhicule / fin de vente", "0,942"],
          ["Âge conducteur / ancienneté du permis", "0,925"],
          ["Puissance / prix du véhicule", "0,875"]],
    left=[("Contrat Maxi", "Taux de sinistre observé : 7,48 %"),
          ("Contrat Mini", "Taux de sinistre observé : 1,74 %")],
    takeaway="Une corrélation faible avec la cible n’exclut ni un effet non linéaire, ni une interaction utile",
    notes="Nous regardons les corrélations entre variables numériques, mais aussi les taux et montants par groupe pour les catégories. Les corrélations élevées entre âge du véhicule et anciennetés de commercialisation motivent un panneau compact pour la fréquence. Puissance et prix peuvent porter une partie de la même information. Cela ne démontre pas que le prix est inutile dans tout modèle, ce qui explique son retour ultérieur dans Ridge. Les différences de taux entre contrats sont descriptives. Elles peuvent refléter la couverture, le véhicule et la sélection des assurés, sans permettre une interprétation causale. Les chiffres de corrélation ici viennent de la matrice de Spearman réexécutée. Ils peuvent différer légèrement de chiffres historiques calculés avec une autre préparation ou une autre définition.",
    refs=("friend", "main"), seconds=55)

add("Une validation qui respecte les clients", "Traitement des données", "validation",
    lead="Les contrats d’un même client restent ensemble dans chaque découpage",
    left=[("Au départ", "Environ 80 % des clients pour apprendre, 20 % pour diagnostiquer"),
          ("Puis", "Validation croisée et sélection interne séparées par client")],
    right=[("Dans chaque apprentissage", "Réapprendre imputations, encodages et modèles"),
           ("Diagnostic final", "3 plis externes sur les 50 000 contrats")],
    takeaway="La validation historique déjà consultée perd son statut de test neuf au fil des itérations",
    notes="Le protocole évolue avec le projet. Au départ, nous réservions environ 20 % des clients et sélectionnions les modèles dans la partie apprentissage. Une fois ce diagnostic consulté à plusieurs reprises, nous ne pouvions plus le présenter comme une évaluation totalement indépendante. Les recherches suivantes utilisent des plis internes pour choisir les paramètres et des plis externes pour mesurer la procédure. Le regroupement par client empêche de retrouver les contrats d’une même personne des deux côtés d’un pli. Le diagnostic le plus récent réapprend la recette finale déjà figée dans trois plis. Il reproduit le résultat et permet d’étudier les erreurs. Il ne refait pas une nouvelle sélection imbriquée indépendante et n’efface pas les décisions humaines informées par les explorations précédentes.",
    refs=("gamma", "broad", "follow", "diag", "skcv"), seconds=60)

add("Des transformations adaptées à chaque modèle", "Traitement des données", "table",
    lead="La préparation fait partie du modèle et change avec sa famille",
    rows=[["Composante", "Transformation", "Pourquoi"],
          ["Logistique", "log1p, standardisation, one-hot", "Échelles cohérentes et effets plus souples"],
          ["Marque, modèle, département en fréquence", "Taux de sinistre lissé, calculé hors client", "Limiter une très forte cardinalité"],
          ["Gamma", "Âge standardisé et catégories rares regroupées", "Régulariser une régression positive"],
          ["XGBoost", "Catégories natives, arbres peu profonds", "Corriger des écarts non linéaires"],
          ["Ridge", "Splines, catégories et indicateurs d’anomalies", "Ajouter des effets souples sous pénalité L2"]],
    takeaway="Même transformation à la prédiction, avec tous les paramètres d’apprentissage figés",
    notes="Nous ne cherchons pas une préparation unique pour toutes les familles. La logistique utilise des logarithmes de certaines entrées, des échelles comparables et un encodage numérique des catégories. Les modèles de véhicule et départements sont trop nombreux pour un one-hot peu régularisé. Nous utilisons donc en fréquence un encodage par taux lissé, calculé hors client pour les données d’apprentissage. Le Gamma standardise seulement son âge numérique et encode le contrat et l’usage. Les arbres n’ont pas besoin de standardisation pour découper leurs entrées. La Ridge finale ajoute des bases de splines afin d’apprendre des courbes avec des coefficients régularisés. Le mot linéaire décrit alors les coefficients, tandis que la relation à la variable d’origine peut être courbe. Aucun traitement ne s’ajuste sur les cibles du test.",
    refs=("friend", "gamma", "main"), seconds=55)

add("Le département et le modèle automobile", "Traitement des données", "columns",
    lead="Une variable peut être peu utile dans un premier modèle et utile dans une correction ultérieure",
    left=[("Géographie détaillée", "12 790 codes distincts dans le train"),
          ("Réduction au département", "96 catégories dans la préparation historique"),
          ("Premier Gamma retenu", "Le département n’appartient pas au meilleur panneau testé")],
    right=[("Fréquence", "Département, marque et modèle : encodage cible lissé"),
           ("Gravité finale", "Département et modèle réintroduits dans Ridge"),
           ("Signal prudent", "Lissage et regroupement des catégories rares")],
    takeaway="Nous n’avons jamais démontré que la géographie ou la voiture étaient intrinsèquement inutiles",
    notes="Le passage du code détaillé au département réduit la fragmentation. La question initiale portait bien sur le département, et pas seulement sur le code à cinq caractères. Dans la sélection Gamma, certains panneaux incluaient le département. Le panneau minimal sans géographie gagnait légèrement, donc nous le retenions pour ce modèle précis. Plus tard, une réanalyse des résidus a motivé un retour aux variables étendues dans Ridge. Le modèle final utilise donc bien le département. Pour le modèle automobile, remplacer directement chaque catégorie par son coût moyen sur toutes les données aurait introduit une fuite et beaucoup de bruit. La fréquence emploie des taux lissés hors client. La Ridge utilise des indicatrices regroupées. La fréquence historique extrait deux caractères, ce qui demanderait une harmonisation pour étendre proprement le système aux DOM.",
    refs=("gamma", "friend", "ridge", "vars", "skenc"), seconds=55)

first_cv = pd.read_csv(PROOFS / "premiere_gravite_cv.csv")
add("Les premières références de gravité", "Modélisation", "chart",
    lead="Première comparaison : même apprentissage et mêmes trois plis clients",
    chart={"type": "bar", "categories": first_cv.modele.tolist(),
           "series": [{"name": "RMSE gravité CV (€)", "values": first_cv.RMSE_CV.tolist()}],
           "axis": "RMSE de gravité (€)", "decimals": 1},
    right=[("Gamma initial", "α = 1 et variables étendues"),
           ("Boosting", "RMSE train 1 325 € contre CV 1 517 €"),
           ("Décision", "Conserver Gamma plutôt que forcer la complexité")],
    takeaway="Gamma : 1 492,80 € en CV, moyenne : 1 501,71 €, boosting : 1 517,35 €",
    notes="Nous avons commencé par deux constantes, moyenne et médiane, puis Gamma et un boosting de petits arbres. La moyenne est la référence naturelle sous erreur quadratique, tandis que la médiane répond plutôt à l’erreur absolue. Sur cette première comparaison à trois plis clients, Gamma obtient le plus faible RMSE moyen : environ 1 492,80 euros. Le boosting apprend beaucoup mieux son train, mais son résultat de validation croisée se dégrade. Cet écart suggère une suradaptation. Nous n’avons pas de preuve chiffrée qu’une régression linéaire ordinaire perd ici, car elle ne faisait pas partie de cette première comparaison. Le choix Gamma combine son résultat mesuré et sa cohérence avec un montant strictement positif. Ces valeurs concernent la gravité et ne se comparent pas directement à la RMSE de prime.",
    refs=("first", "skgamma"), seconds=50)

add("La sélection de variables du Gamma", "Modélisation", "table",
    lead="50 configurations : 10 ensembles de variables × 5 pénalisations",
    rows=[["Ensemble", "Variables", "α", "RMSE CV gravité"],
          ["Minimal contrat-véhicule", "3", "0,1", "1 491,18 €"],
          ["Métier compact sans géographie", "9", "0,1", "1 493,62 €"],
          ["Métier compact avec département", "10", "0,1", "1 493,83 €"],
          ["Toutes avec département", "25", "1", "1 494,04 €"]],
    takeaway="Choix : contrat, ancienneté du véhicule et utilisation, avec α = 0,1",
    notes="Nous avons ensuite gardé la famille Gamma tout en comparant des groupes de variables. Nous avons testé dix panneaux et cinq valeurs d’alpha, soit cinquante configurations. L’ensemble minimal contient le type de contrat, l’ancienneté du véhicule et l’utilisation. Il obtient 1 491,18 euros en moyenne sur les cinq plis de sélection. Une validation imbriquée de cette procédure choisit ce panneau dans trois des quatre plis externes. Le quatrième choisit un panneau à neuf variables. La sélection n’est donc pas parfaitement stable. Son RMSE global hors pli vaut 1 498,97 euros contre 1 510,43 pour une moyenne constante. Il faut séparer ce résultat de procédure du meilleur score de sélection. Cette recherche porte sur des groupes prédéfinis, et pas sur tous les sous-ensembles possibles.",
    refs=("gamma",), seconds=60)

add("Le fonctionnement du Gamma à lien log", "Modélisation", "formula",
    lead="Les caractéristiques entrent linéairement dans le logarithme du montant moyen",
    formula="g(X) = exp(b₀ + Σ βⱼ zⱼ)",
    left=[("Pendant l’apprentissage", "Imputer l’âge, standardiser, encoder les catégories"),
          ("Optimisation", "Déviance Gamma et pénalisation L2, α = 0,1")],
    right=[("Pendant la prédiction", "Appliquer la préparation figée puis exponentier le score"),
           ("Interprétation", "Des effets multiplicatifs sur une moyenne positive")],
    takeaway="Le lien log du Gamma ne correspond pas à une régression ordinaire sur log(montant)",
    notes="Le Gamma est un modèle linéaire généralisé. Il construit une combinaison linéaire des entrées préparées, puis une exponentielle transforme ce score en montant positif. Le type de contrat et l’utilisation deviennent des indicatrices, et l’âge du véhicule est standardisé. Les coefficients s’interprètent sur le logarithme de la moyenne, donc leurs effets sont multiplicatifs sur le montant. Pendant le fit, la cible reste le montant en euros. Nous optimisons une déviance Gamma régularisée, puis utilisons la RMSE pour comparer les configurations. Ce choix évite de confondre la moyenne du montant avec l’exponentielle de la moyenne de son logarithme. Le modèle reste contraint et ne prévoit pas exactement chaque accident. Sa sortie est une espérance conditionnelle au fait qu’un sinistre survienne.",
    refs=("gamma", "main", "skgamma"), seconds=55)

add("Le premier modèle de fréquence du partenaire", "Modélisation", "table",
    lead="La logistique C = 0,1 est retenue après comparaison au taux moyen et au boosting",
    rows=[["Candidat", "Log-loss CV", "Log-loss validation historique"],
          ["Taux moyen", "Référence constante", "0,2225"],
          ["Logistique C = 0,1", "0,21420", "0,2137"],
          ["Boosting prudent", "Environ 0,2148", "0,2135"]],
    left=[("Ce qui fonctionne", "Amélioration face au taux moyen et probabilités interprétables")],
    takeaway="Le boosting gagne légèrement sur ce diagnostic, mais la logistique gagne la comparaison CV et reste la référence",
    notes="Le travail du partenaire commence lui aussi par une référence constante. La logistique réduit la log-loss de validation d’environ 0,2225 à 0,2137. Une grille compare C égal à 0,001, 0,01, 0,1, 1 et 10. Le meilleur C en validation croisée vaut 0,1. Le partenaire compare ensuite neuf réglages de boosting, dont un volontairement complexe. Le meilleur boosting obtient environ 0,2148 en CV contre 0,2142 pour la logistique, mais il est légèrement meilleur sur la validation historique. Nous expliquons cette différence au lieu de choisir discrètement selon le chiffre le plus favorable. Les faibles écarts et la simplicité soutiennent le maintien de la logistique. Des scores train et validation proches limitent le surapprentissage visible, sans démontrer son absence absolue.",
    refs=("friend", "audit"), seconds=60)

add("Une reprise méthodologique : l’encodage cible", "Traitement des données", "comparison",
    lead="Le découpage de validation était groupé, mais l’encodage interne devait aussi respecter les clients",
    left=[("Audit du premier encodage", "2 540 clients partagés entre plis d’encodage"),
          ("Lignes concernées", "5 322 contrats"),
          ("Risque", "Une ligne peut recevoir l’information d’un autre contrat du même client")],
    right=[("Correction", "5 plis d’encodage séparés par client"),
           ("Prior lissé", "Taux moyen calculé sans les clients encodés"),
           ("Même validation historique", "Log-loss : 0,213749 puis 0,213834")],
    takeaway="Nous conservons la correction méthodologique même si le score apparent devient légèrement moins bon",
    notes="Cet audit constitue une vraie reprise de décision. Un modèle peut utiliser une validation par client tout en gardant, à l’intérieur, un encodage cible calculé sur des plis par ligne. Le rapport retrouve 2 540 clients présents dans plusieurs plis d’encodage, pour 5 322 contrats concernés. Le risque est qu’un autre contrat du même client informe l’encodage d’une ligne. Nous avons remplacé ces plis par un découpage par client. Le prior de lissage est également calculé sur les autres plis. Sur les mêmes clients de validation historique, la log-loss passe de 0,2137486 à 0,2138339. L’écart est petit et incertain. Nous gardons la correction pour la solidité du protocole, car améliorer l’honnêteté de l’évaluation compte même lorsqu’un score brut se dégrade.",
    refs=("audit", "merge", "skenc"), seconds=55)

add("La première combinaison des deux modèles", "Validation du modèle", "formula",
    lead="Une partition commune permet de multiplier des prédictions réellement hors apprentissage",
    formula="prime = p_logistique × g_Gamma",
    left=[("Fréquence", "Logistique C = 0,1, encodage par client"),
          ("Gravité", "Gamma α = 0,1 sur les trois variables métier")],
    right=[("Diagnostic historique commun", "9 987 contrats, RMSE de prime 507,647 €"),
           ("Public Kaggle de référence", "1 365,943 dans le registre historique")],
    takeaway="La RMSE de gravité et la RMSE de prime répondent à des cibles et populations différentes",
    notes="Pour assembler les modèles, nous avons d’abord harmonisé la validation. Le partenaire et la gravité n’avaient pas exactement la même partition initiale. Nous utilisons une partition commune pour que chaque fréquence et chaque gravité soient calculées sans apprentissage sur les clients évalués. Les sorties sont jointes par identifiant, avec des contrôles contre les doublons et le mauvais ordre. Le diagnostic historique de cette combinaison porte sur 9 987 contrats et donne une RMSE de prime d’environ 507,647 euros. Ce nombre ne signifie pas que notre modèle était meilleur que le futur résultat à 541,916 : la population et le protocole changent. Le fichier de référence obtient ensuite le score public 1 365,943. Le test de compétition ne révèle pas ses coûts localement.",
    refs=("merge", "registry"), seconds=50)

add("Les premiers mélanges de fréquence et de Gamma", "Modélisation", "comparison",
    lead="391 paires comparent modèles et mélanges sur la MSE de prime",
    left=[("Fréquence retenue", "75 % logistique et 25 % boosting souple"),
          ("Gravité retenue", "Gamma à trois variables, α = 0,01"),
          ("Principe", "Choisir la paire sur les mêmes contrats hors pli")],
    right=[("Procédure imbriquée", "550,380829 € contre 550,386238 € pour la référence"),
           ("Gain local", "0,005409 €, intervalle descriptif contenant zéro"),
           ("Score public v4", "1 366,006, moins bon que la référence")],
    takeaway="Un léger gain local n’a pas suffi à améliorer le score public",
    notes="La première amélioration commune compare 391 paires. La fréquence retenue conserve soixante-quinze pour cent de la logistique et ajoute vingt-cinq pour cent de boosting souple. Le Gamma garde les trois mêmes variables et réduit alpha à 0,01. La sélection se fait sur la prime et pas seulement sur les métriques séparées. Dans la validation imbriquée du développement, le gain de procédure face à la référence est d’environ cinq millièmes d’euro de RMSE. L’intervalle descriptif contient zéro. Le diagnostic historique gagne lui aussi très légèrement, mais le score public v4 vaut 1 366,006 et se dégrade. Cette tentative nous apprend que les écarts locaux très faibles peuvent être instables. Elle reste archivée et ne doit pas être confondue avec le futur mélange CatBoost, qui constitue une autre recherche.",
    refs=("v4", "registry"), seconds=55)

add("Les recherches plus larges de modèles", "Modélisation", "table",
    lead="La complexité augmente, mais le protocole doit rester comparable",
    rows=[["Recherche", "Comparaison documentée", "Décision"],
          ["CatBoost et LightGBM", "1 215 paires, puis sélection imbriquée", "Gain local faible, pas de score public confirmé"],
          ["EBM et XGBoost", "525 paires, puis contrôle imbriqué", "550,299 contre 550,285 : pas de gain"],
          ["TabPFN-v2 en gravité", "3 ou 8 variables, mélanges avec Gamma", "550,070 contre 550,066 : pas de gain"]],
    takeaway="Les familles modernes restent des challengers lorsque la comparaison ne justifie pas leur promotion",
    notes="Nous avons élargi les familles au-delà des premiers modèles. Une recherche compare douze fréquences, quatorze gravités et leurs mélanges, soit 1 215 paires. Elle propose un mélange CatBoost-logistique et un Gamma, mais son gain imbriqué reste faible et son nouveau score public n’est pas confirmé. Les recherches suivantes ajoutent EBM, XGBoost, puis un benchmark TabPFN-v2 pour la gravité. Les meilleurs scores de sélection sont séduisants, mais une comparaison imbriquée de la procédure EBM-XGBoost ne bat pas la recette fixe de référence. TabPFN n’améliore aucun mélange testé. Les budgets, catalogues et populations diffèrent entre recherches. Nous n’additionnons donc pas ces résultats dans un classement unique. Nous conservons les tentatives pour expliquer pourquoi elles ne constituent pas le modèle principal.",
    refs=("broad", "modern", "registry"), seconds=55)

add("Le retour à la référence après deux essais XGBoost", "Validation du modèle", "table",
    lead="Les corrections limitées et la reprise complète n’apportent pas le gain attendu",
    rows=[["Essai", "Résultat local documenté", "Score public", "Suite"],
          ["Correction de logit XGB", "Gain sur 2 partitions, incertain", "1 366,010", "Écarté"],
          ["Reprise XGB fréquence et gravité", "542,128 contre 542,060 €", "1 366,025", "Écartée"],
          ["Logistique × Gamma", "Référence témoin", "1 365,943", "Conservée"]],
    takeaway="Le retour de validation remet ici en question l’architecture, pas seulement ses paramètres",
    notes="Deux nouvelles tentatives illustrent la boucle de retour de la roadmap. La première conserve le Gamma et ajoute une correction XGBoost aux marges de la logistique. Elle améliore deux partitions locales de quelques centièmes d’euro, avec une incertitude qui inclut zéro, mais le score public se dégrade à 1 366,010. La reprise XGBoost complète explore soixante configurations par composante, trois panneaux de variables et deux politiques de nettoyage. Sa procédure externe obtient 542,128 contre 542,060 pour la référence. Son score public de 1 366,025 est également moins bon. Nous ne présentons donc pas XGBoost comme automatiquement supérieur. Nous revenons à une référence stable, puis changeons la façon d’utiliser les arbres plutôt que d’augmenter simplement leur complexité.",
    refs=("anchor", "restart", "registry"), seconds=55)

add("Un nouveau raisonnement pour corriger la gravité", "Modélisation", "formula",
    lead="Conserver Gamma comme ancre et apprendre ses écarts avec des valeurs auxiliaires hors client",
    formula="poids ∝ p_aux × g_aux²",
    left=[("Cible XGBoost", "Ratio coût / Gamma auxiliaire"),
          ("Correction limitée", "Petits arbres et mélange avec Gamma")],
    right=[("Apprentissage", "Sinistres positifs seulement"),
           ("Sélection", "RMSE finale de prime sur tous les contrats")],
    takeaway="La pondération rapproche l’apprentissage du ratio de l’objectif de prime sans garantir un gain",
    notes="Le changement de raisonnement consiste à conserver le montant Gamma et à apprendre une correction plutôt qu’un remplacement complet. Pour chaque ligne sinistrée, une Gamma et une probabilité auxiliaires sont calculées hors client. La cible des arbres devient le coût divisé par cette Gamma. Les poids sont proportionnels à la probabilité auxiliaire multipliée par la Gamma au carré. L’identité p fois g au carré fois y sur g moins ratio au carré égale p fois y moins g fois ratio au carré permet d’interpréter la perte. La densité des sinistrés comporte déjà un facteur de probabilité. Cela rapproche la correction de la partie de l’erreur de prime qui dépend de la gravité. Les probabilités restent estimées et les modèles contraints, donc cette justification doit être confirmée par les comparaisons.",
    refs=("refine", "main"), seconds=55)

add("La première amélioration de gravité conservée", "Modélisation", "formula",
    lead="Fréquence originale inchangée, gravité corrigée avec prudence",
    formula="s₁ = 0,5 × Gamma + 0,5 × Gamma corrigé XGBoost",
    left=[("Variables", "Contrat, ancienneté du véhicule et utilisation"),
          ("XGBoost", "240 arbres, profondeur 2, taux 0,025")],
    right=[("Régularisation", "L2 = 100, L1 = 1, poids minimal d’enfant = 100"),
           ("Public communiqué", "1 365,913 contre 1 365,943 pour la référence")],
    takeaway="Le progrès vient d’une correction régularisée et d’un mélange, pas d’un XGBoost complet",
    notes="La recherche compare trente-quatre configurations de gravité avec plusieurs poids de mélange, en conservant la fréquence initiale. Le candidat finalement retenu mélange à parts égales Gamma et son montant corrigé par un seul XGBoost. Il utilise exactement les trois variables métier du Gamma. Les arbres sont peu profonds, le taux d’apprentissage faible et la régularisation forte. Nous conservons les grands sinistres et calculons les auxiliaires sans le client évalué. Le gain de la procédure locale est très petit et son intervalle contient zéro. Le résultat public transmis, 1 365,913, devient néanmoins le meilleur public observé à ce stade. Cette promotion se rapporte à nos essais et au score public. Elle ne transforme pas le petit gain local en preuve de supériorité sur toutes les populations.",
    refs=("refine", "registry", "xgb"), seconds=55)

add("Le successeur plus complexe est écarté", "Validation du modèle", "comparison",
    lead="Trois XGBoost de 480 arbres ne remplacent pas automatiquement un correcteur de 240 arbres",
    left=[("Essai", "Moyenne de trois graines, sous-échantillonnage 0,8"),
          ("Architecture", "Même ancre Gamma et même trio de variables"),
          ("Gain local", "Environ 0,0081 € de RMSE, intervalle contenant zéro")],
    right=[("Meilleur précédent", "1 365,913"),
           ("Successeur", "1 365,914"),
           ("Décision", "Conserver le correcteur simple précédent")],
    takeaway="Un millième de score ne prouve pas une différence stable, mais ne justifie pas de promouvoir le successeur",
    notes="Pour réduire la variabilité des arbres, nous avons testé une moyenne de trois XGBoost avec 480 arbres chacun et un sous-échantillonnage de 0,8. Le modèle utilise toujours les trois mêmes variables. La comparaison locale donne un gain d’environ 0,0081 euro de RMSE, mais l’intervalle descriptif inclut zéro. Le score public communiqué atteint 1 365,914 contre 1 365,913 pour le modèle précédent. Nous ne prétendons pas qu’un millième établit une hiérarchie robuste. En revanche, nous n’avons pas d’argument suffisant pour remplacer le précédent meilleur public par une recette plus complexe. Le modèle principal final conservera donc le correcteur unique à 240 arbres, et non les trois modèles de ce successeur.",
    refs=("bag", "registry"), seconds=45)

add("Le retour à l’analyse des variables", "Analyse des données", "table",
    lead="Les résidus du meilleur modèle révèlent des pistes, mais certains effets dépendent des grands sinistres",
    rows=[["Groupe descriptif", "Résidu relatif pondéré", "Sans les 25 plus grands coûts"],
          ["Mercedes Benz", "+22,3 %", "+5,6 %"],
          ["Permis obtenu après 26 ans", "+11,2 %", "+1,5 %"],
          ["Modèle automobile 206", "+17,5 %", "+12,6 %"]],
    takeaway="Le retrait des gros coûts sert uniquement au diagnostic d’influence, jamais à l’entraînement",
    notes="Nous sommes revenus à la case analyse des données en examinant les résidus hors client du meilleur modèle. L’idée est de repérer des groupes systématiquement sous-estimés, pas d’inventer une règle de prix à partir d’une moyenne. Pour Mercedes, le résidu relatif pondéré passe d’environ plus vingt-deux pour cent à plus six pour cent lorsque les vingt-cinq coûts les plus élevés sont retirés du seul calcul descriptif. L’effet de l’âge d’obtention du permis diminue également beaucoup. Pour le modèle 206, une partie du résidu reste visible. Ces résultats motivent de nouvelles hypothèses sur le véhicule et le conducteur. Ils montrent aussi pourquoi une moyenne de coût par marque ou modèle, calculée sans lissage ni validation, serait fragile. Les observations coûteuses restent intégralement dans les apprentissages.",
    refs=("ridge",), seconds=50)

add("La correction Ridge réintroduit les variables étendues", "Modélisation", "formula",
    lead="Une correction pénalisée ajoute du signal sans abandonner l’ancre de gravité",
    formula="s₂ = 0,75 × s₁ + 0,25 × Gamma corrigé Ridge",
    left=[("Variables revenues", "Prix, puissance, poids, marque, modèle, département, conducteurs"),
          ("Variables dérivées", "Puissance/poids, âge d’obtention du permis, contrat × usage")],
    right=[("Recette choisie", "Ridge α = 10, splines quadratiques, panneau complet"),
           ("Public communiqué", "1 365,745, soit 0,168 de mieux")],
    takeaway="Le modèle complet utilise davantage que trois variables, même si le Gamma conserve son trio initial",
    notes="La réanalyse conduit à quatre familles de variables : véhicule, conducteur, contrat et ensemble complet. Nous comparons des corrections Ridge avec quatre pénalisations et plusieurs corrections XGBoost, en les mélangeant au meilleur modèle précédent. Le choix final est une Ridge alpha dix, panneau complet, avec un poids de vingt-cinq pour cent. Sa cible est coût sur Gamma auxiliaire moins un. Elle utilise des splines quadratiques, des indicatrices et des catégories regroupées si elles ont trop peu de sinistres. Nous conservons soixante-quinze pour cent du mélange Gamma-XGBoost. Le score public transmis descend à 1 365,745. Cette étape répond à la question sur les trois variables : elles convenaient au Gamma simple, mais elles ne représentent plus tout le modèle final. Le département et le modèle automobile sont revenus dans une composante régularisée.",
    refs=("ridge", "vars", "model"), seconds=55)

add("Les nouvelles courbes de gravité ne gagnent pas", "Validation du modèle", "table",
    lead="Après Ridge, davantage de souplesse dégrade la comparaison externe",
    rows=[["Recherche", "RMSE de prime", "Référence fixe sur les mêmes plis", "Décision"],
          ["Courbes et pénalisations de gravité", "542,000491 €", "541,971031 €", "Pas d’export"],
          ["Interactions hiérarchiques", "541,986791 €", "541,971031 €", "Pas d’export"]],
    left=[("Hypothèses testées", "Changer les nœuds, la pénalité et les interactions couverture-véhicule")],
    takeaway="La validation conduit à limiter la gravité et à déplacer l’effort vers la fréquence",
    notes="La roadmap ne demande pas seulement d’optimiser, elle demande de revenir en arrière lorsque le résultat le justifie. Après la Ridge, nous avons testé davantage de nœuds, d’autres pénalités, des seuils de regroupement et des interactions hiérarchiques entre couverture et véhicule ou entre usage et conducteur. La recherche de courbes de gravité obtient une RMSE externe d’environ 542,0005 euros, et les interactions 541,9868, contre 541,9710 pour la recette précédente fixée et réapprise sur les mêmes plis. Ces recherches ne déclenchent aucun export. Nous ne choisissons pas un modèle simplement parce qu’il semble plus riche ou qu’il gagne une étape interne. À ce stade, la gravité reste fixe et nous cherchons une amélioration ciblée de la fréquence.",
    refs=("follow",), seconds=45)

add("La correction finale de fréquence", "Modélisation", "formula",
    lead="Ajuster les courbes d’âge et leur interaction avec le contrat",
    formula="p = sigmoid(logit(p₀) + δ(X))",
    left=[("Base conservée", "Logistique C = 0,1 et préparation hors client"),
          ("Correction choisie", "Âge véhicule, contrat, usage et âge × contrat")],
    right=[("Souplesse contrôlée", "4 nœuds, degré 2, L2 α = 100"),
           ("Sélection", "28 choix avec poids 0,25, 0,5 ou 1 et option inchangée")],
    takeaway="Gravité inchangée, score public communiqué : 1 365,657",
    notes="La dernière amélioration vise la forme des probabilités, pas le rééquilibrage artificiel des classes. Nous partons de la probabilité p zéro de la logistique et ajoutons une correction à son logit. La correction utilise des splines de l’âge du véhicule, le contrat, l’usage et des interactions âge-contrat. Elle minimise une log-loss binomiale régularisée avec alpha cent. La recherche compare neuf configurations et trois poids, plus le modèle inchangé. Les trois plis externes retiennent la même configuration, avec un poids de correction égal à un. Les probabilités utilisées pour apprendre cette correction sont calculées dans trois sous-plis séparés par client. La gravité est exactement celle de la recette à 1 365,745. Le nouveau score public transmis vaut 1 365,657, qui devient le meilleur observé dans le registre.",
    refs=("follow", "model", "skspline"), seconds=55)

add("L’architecture du modèle retenu", "Modélisation", "architecture",
    lead="Les deux composantes produisent des moyennes conditionnelles complémentaires",
    formula="prime(X) = p(X) × s(X)",
    left=[("Fréquence", "Logistique et correction de logit par splines")],
    right=[("Gravité", "37,5 % Gamma + 37,5 % Gamma corrigé XGB + 25 % Gamma corrigé Ridge")],
    takeaway="Gamma et XGB utilisent trois variables. La fréquence et Ridge exploitent des variables étendues",
    notes="Voici la recette exacte que nous avons promue comme modèle principal. La fréquence utilise la logistique initiale corrigée par des courbes régularisées. La gravité combine trois montants : trente-sept et demi pour cent du Gamma, trente-sept et demi pour cent du Gamma corrigé par XGBoost, et vingt-cinq pour cent du Gamma corrigé par Ridge. Ces poids viennent de deux mélanges successifs. Nous avions d’abord cinquante pour cent de Gamma et cinquante pour cent de correction XGBoost, puis nous avons conservé soixante-quinze pour cent de ce mélange et ajouté vingt-cinq pour cent de Ridge. Le Gamma et le correcteur XGBoost utilisent le trio métier. La fréquence et la Ridge ajoutent des variables. Le produit final représente le coût attendu par prospect. Aucun seuil de détection d’accident ne vient mettre arbitrairement les primes à zéro.",
    refs=("model", "main"), seconds=60)

add("Les scores publics et les décisions", "Validation du modèle", "public_history",
    lead="Le parcours comporte des progrès et des tentatives moins bonnes",
    chart={"type": "line", "categories": ["Référence", "v4", "Ancré", "Reprise XGB", "Ratio XGB", "3 graines", "Ridge", "Courbes p"],
           "series": [{"name": "Score public", "values": [1365.943, 1366.006, 1366.010, 1366.025, 1365.913, 1365.914, 1365.745, 1365.657]}],
           "axis": "Score public Kaggle, plus faible = meilleur", "min":1365.60, "max":1366.08, "decimals":3},
    takeaway="Amélioration de la référence au meilleur public observé : 0,286. Classement privé inconnu",
    notes="Cette courbe représente les scores publics documentés, avec un axe volontairement resserré et explicitement indiqué. Elle ne représente pas une validation locale. Les valeurs de référence et de v4 étaient consignées comme vérifiées dans le registre historique. Les essais suivants proviennent des résultats Complete transmis par l’utilisateur et n’ont pas été revérifiés indépendamment sur Kaggle. La courbe montre les retours à la référence après les essais ancré et reprise XGBoost, puis les progrès par correction de gravité, Ridge et correction de fréquence. Nous ne donnons aucun score aux challengers dont le résultat exact n’est pas connu. Le meilleur public est passé de 1 365,943 à 1 365,657, un gain de 0,286. Le seuil exploratoire de 1 365,500 n’a pas été atteint et le score privé reste inconnu.",
    refs=("registry", "follow"), seconds=50)

add("La validation locale du modèle final", "Validation du modèle", "table",
    lead="Recettes fixes réapprises sur les mêmes trois plis par client",
    rows=[["Recette", "RMSE prime", "MAE prime", "Total prédit / observé"],
          ["Logistique × Gamma", "542,0601 €", "190,8614 €", "0,99681"],
          ["Gravité enrichie, fréquence initiale", "541,9710 €", "190,7323 €", "0,99646"],
          ["Modèle final", "541,9156 €", "191,0908 €", "1,00120"]],
    takeaway="La RMSE s’améliore de 0,05546 €, mais la MAE se dégrade d’environ 0,3584 €",
    notes="La comparaison finale utilise les mêmes observations et les mêmes plis pour trois recettes fixes. Le modèle final obtient environ 541,9156 euros de RMSE de prime contre 541,9710 pour la gravité enrichie sans correction finale de fréquence. Le gain vaut environ 0,05546 euro, soit un centième de pour cent. Le bootstrap descriptif apparié par client fournit un intervalle d’environ 0,00465 à 0,11079 euro pour ce gain. Cet intervalle est conditionnel aux prédictions ajustées et ne couvre pas toute l’incertitude d’apprentissage ou les recherches précédentes. La MAE augmente légèrement, donc toutes les métriques ne s’améliorent pas simultanément. Le ratio des sommes est proche de un, ce qui vérifie une cohérence globale, sans démontrer la rentabilité commerciale. Ces données ont déjà été explorées : nous ne présentons pas ce diagnostic comme un test final neuf.",
    refs=("diag", "main"), seconds=55)

analysis = json.loads((ROOT / "reports/main/analyse.json").read_text())
importance = sorted(analysis["permutation_importance"], key=lambda a:a["mean"], reverse=True)
add("Ce qui explique les prédictions et les erreurs", "Validation du modèle", "importance",
    lead="Interprétation sur clients exclus, sans transformer les associations en causes",
    chart={"type":"bar", "categories":[a["group"] for a in importance[:6]],
           "series":[{"name":"Hausse de MSE après permutation (€²)","values":[a["mean"] for a in importance[:6]]}],
           "axis":"Hausse de MSE de prime (€²)","decimals":0},
    right=[("Variables dominantes", "Âge du véhicule, contrat et usage"),
           ("Département", "Signal présent dans le modèle complet"),
           ("Queue des coûts", "30 sinistres expliquent environ 23,7 % de l’erreur quadratique")],
    takeaway="Les catégories rares, les variables corrélées et les gros sinistres limitent la stabilité des conclusions",
    notes="L’importance par permutation mesure la hausse d’erreur quand on brouille un groupe de variables dans les clients exclus. L’âge du véhicule et le contrat avec l’usage dominent, devant les caractéristiques techniques et le conducteur principal. Le département produit une hausse moyenne d’environ deux cent huit euros carrés de MSE lorsqu’on le permute. Cela confirme qu’il est bien utilisé dans le modèle complet et ne dit rien sur une cause géographique. Les variables corrélées peuvent se partager leur importance, et une permutation peut créer des profils irréalistes. Le diagnostic des erreurs montre aussi que les trente plus grands sinistres, environ un pour cent des sinistrés, concentrent près de vingt-quatre pour cent de l’erreur quadratique. C’est une limite majeure de précision et de stabilité, conservée dans l’évaluation.",
    refs=("analysis", "main"), seconds=55)

add("La prédiction et le fichier de soumission", "Validation du modèle", "submission",
    lead="Réapprendre la recette retenue, puis prédire les 50 000 prospects",
    formula="8 % × 2 300 € = 184 € de prime attendue",
    left=[("Exemple fictif", "Gamma 2 000 €, Gamma-XGB 2 400 €, Gamma-Ridge 2 600 €"),
          ("Mélange", "0,375 × 2 000 + 0,375 × 2 400 + 0,25 × 2 600 = 2 300 €")],
    right=[("Colonnes", "index, pred"), ("Index", "50 000 à 99 999"),
           ("Contrôles", "Ordre, doublons, valeurs finies et identité p × s")],
    takeaway="pred est une prime en euros. La RMSE est une mesure globale calculée lorsque les coûts sont connus",
    notes="Après la validation, nous ajustons la fréquence sur les cinquante mille contrats et les gravités sur les deux mille neuf cent dix-sept sinistres positifs. Le modèle et toutes les transformations restent ensuite figés. L’exemple affiché est fictif et vise seulement à expliquer les unités. Les trois montants se combinent en une gravité de deux mille trois cents euros. Avec une probabilité de huit pour cent, la prime attendue vaut cent quatre-vingt-quatre euros. Cela ne veut pas dire que l’accident coûtera cent quatre-vingt-quatre euros. Le fichier contient index et pred, avec les index exacts du test. Nous contrôlons le nombre de lignes, l’ordre, les doublons et les domaines des prédictions. Le modèle rechargé reproduit les cinquante mille primes de référence avec un écart numérique maximal inférieur à 2,3 fois dix puissance moins neuf euro.",
    refs=("model", "analysis"), seconds=50)

add("Le bilan des décisions", "Conclusion", "columns",
    lead="Le meilleur modèle observé conserve une structure simple et lui ajoute des corrections contrôlées",
    left=[("Ce que nous conservons", "Décomposition fréquence-gravité et ancre Gamma"),
          ("Ce que nous avons corrigé", "Plis d’encodage, variables étendues et courbes de fréquence"),
          ("Ce que nous avons écarté", "Remplacements complets et complexité sans gain convaincant")],
    right=[("Limites", "Peu de sinistres, queue influente, données déjà explorées"),
           ("Prochaine validation", "Réserver une cohorte réellement fraîche"),
           ("Avant un tarif réel", "Clarifier exposition, disponibilité des variables et suivi de dérive")],
    takeaway="Meilleur public observé : 1 365,657. Le projet est reproductible pour l’exercice et reste à consolider pour un usage réel",
    notes="Nous pouvons résumer notre histoire par la manière dont les décisions ont évolué. La première sélection privilégiait un Gamma simple. L’assemblage a déplacé le critère vers la prime. Les essais de modèles plus complexes n’ont pas systématiquement gagné, ce qui nous a amenés à conserver une ancre et à apprendre des corrections limitées. L’audit d’encodage a renforcé la validité méthodologique, même sans amélioration de score apparent. La réanalyse des résidus a justifié le retour de variables que le premier Gamma ne retenait pas. Enfin, une correction de fréquence a fourni le meilleur public observé. Les gains restent faibles et les grands sinistres dominent une part de l’erreur. La prochaine amélioration de confiance viendra surtout d’une nouvelle cohorte et d’un meilleur dictionnaire de données, plutôt que d’une promesse de modèle optimal.",
    refs=("main", "registry"), seconds=50)

MAIN_COUNT = len(slides)

variables = pd.read_csv(ROOT / "reports/main/variables_par_composante.csv")
variable_notes = {
    "bonus":"log1p en fréquence, effet du conducteur dans Ridge",
    "type_contrat":"one-hot, présent dans toutes les composantes",
    "duree_contrat":"conservée sans conversion en exposition",
    "anciennete_info":"information d’ancienneté, unité à confirmer",
    "freq_paiement":"one-hot et regroupement des rares",
    "paiement":"catégorie, pas un montant supposé",
    "utilisation":"one-hot, présente dans toutes les composantes",
    "code_postal":"représentation par département",
    "conducteur2":"indicateur d’absence ou présence conservé",
    "age_conducteur1":"standardisation ou courbes selon composante",
    "age_conducteur2":"retiré de la logistique, courbes Ridge avec absence",
    "anciennete_permis1":"retirée de la logistique, revient dans Ridge",
    "anciennete_permis2":"zéros structurels et anomalies traités selon composante",
    "anciennete_vehicule":"variable centrale, linéaire ou splines",
    "cylindre_vehicule":"zéro traité comme manquant, log1p selon composante",
    "din_vehicule":"puissance, log1p et ratio puissance/poids",
    "essence_vehicule":"catégorie énergie, modalités rares",
    "marque_vehicule":"encodage cible en fréquence, one-hot Ridge",
    "modele_vehicule":"encodage cible en fréquence, one-hot Ridge",
    "debut_vente_vehicule":"retiré de la logistique, revient dans Ridge",
    "fin_vente_vehicule":"inutilisé dans le modèle final",
    "vitesse_vehicule":"retirée de la logistique, revient dans Ridge",
    "type_vehicule":"catégorie conservée en fréquence et Ridge",
    "prix_vehicule":"retiré de la logistique, log-prix dans Ridge",
    "poids_vehicule":"zéro manquant, médianes, log1p et ratio",
}
for part, frame in enumerate((variables.iloc[:13], variables.iloc[13:]),1):
    rows=[["Champ", "Base p", "Courbe p", "Gamma/XGB", "Ridge"]]
    for _,r in frame.iterrows():
        rows.append([r.champ]+["Oui" if bool(r[c]) else "-" for c in ["logistique_base","correction_frequence","Gamma","correction_Ridge"]])
    add(f"Variables conservées, partie {part}", "Annexes", "table",
        lead="La présence dans une composante ne signifie pas une contribution causale ni une importance élevée",
        rows=rows, compact=True,
        takeaway="Identifiants, index, sexes et cibles exclus. Le département est dérivé du code détaillé",
        notes="Détail des transformations :\n"+"\n".join(f"{c} : {variable_notes[c]}." for c in frame.champ)+
        "\nLes colonnes Gamma et correction XGBoost ont exactement le même trio de variables. Une colonne brute peut être nécessaire à un contrôle ou à une variable dérivée sans être directement utilisée par une composante.",
        refs=("vars", "main"), appendix=True)

add("L’encodage cible lissé, expliqué par un exemple", "Annexes", "formula",
    lead="Le taux d’une catégorie est ramené vers la moyenne quand son effectif est faible",
    formula="encodage(c) = [sinistres(c) + 100 × taux moyen] / [effectif(c) + 100]",
    left=[("Illustration fictive", "5 sinistres sur 50 contrats, prior à 6 %"),
          ("Valeur lissée", "(5 + 100 × 0,06) / (50 + 100) = 7,33 %")],
    right=[("Apprentissage", "5 plis séparés par client, prior des autres plis"),
           ("Catégorie inconnue", "Retour au taux moyen appris sur le train")],
    takeaway="Une moyenne de coût ou de fréquence calculée sur toutes les lignes avant la validation ferait fuiter la cible",
    notes="Le lissage évite de croire qu’une petite catégorie décrit un risque connu avec précision. L’exemple est volontairement fictif et utilise un prior rond de six pour cent, différent du taux exact du portefeuille. Le modèle final emploie ce principe pour la marque, le modèle automobile et le département en fréquence. Les cinq plis d’encodage sont groupés par client. Les lignes d’apprentissage reçoivent un taux calculé sans leur client, tandis que les lignes de validation ou de test reçoivent les tables ajustées sur tout l’apprentissage autorisé. Les catégories inconnues utilisent le prior. La Ridge de gravité n’utilise pas ce même encodage cible de coût : elle utilise des indicatrices et un regroupement des modalités trop rares.",
    refs=("friend", "merge", "main", "skenc"), appendix=True)

add("Les catalogues d’essais documentés", "Annexes", "table",
    lead="Les budgets de recherche changent. Ils ne forment pas une seule expérience comparable",
    rows=[["Étape", "Budget documenté", "Choix et protocole"],
          ["Gamma, variables", "10 panneaux × 5 α = 50", "5 plis de sélection, 4 externes"],
          ["Premières paires", "391 combinaisons", "Développement groupé et contrôle imbriqué"],
          ["CatBoost/LightGBM", "1 215 combinaisons", "3 internes et 3 externes sur développement"],
          ["Recherche moderne", "525 couples", "Contrôles fixes puis sélection imbriquée"],
          ["Reprise XGBoost", "60 configurations par composante", "5 internes et 3 externes, train complet"],
          ["Gravité ratio / Ridge / courbes p", "34 configs / 61 choix / 28 choix", "Auxiliaires hors client et plis groupés"]],
    takeaway="Grilles et tirages déterministes documentés. Aucune étude Optuna n’est présentée comme réalisée",
    notes="La roadmap mentionne des techniques possibles d’optimisation, dont Grid Search et Optuna. Nous ne devons pas prétendre avoir utilisé toutes les options du schéma. Le projet contient des grilles de paramètres, des panneaux prédéfinis, des tirages déterministes et des mélanges convexes. La première gravité comparait aussi sept réglages ajustables à des constantes. Les recherches suivantes de gravité incluent Gamma et Poisson avec décalages, pondérations et corrections. Le décalage de six cents euros était une hypothèse de modélisation motivée par le support observé, pas une franchise contractuelle connue. Il n’appartient pas à la recette finale. Les mêmes données ayant été explorées, les évaluations restent diagnostiques. Les résultats de sélection internes sont optimistes et ne remplacent pas un résultat externe.",
    refs=("gamma", "v4", "broad", "modern", "restart", "refine", "ridge", "follow"), appendix=True, compact=True)

oof = pd.read_csv(ROOT / "reports/main/diagnostic_oof.csv")
y = oof.claim.to_numpy(dtype=int)
def cm(t):
    pred = oof.proba_pred.to_numpy() >= t
    return [[int(((y==0)&(~pred)).sum()),int(((y==0)&pred).sum())],
            [int(((y==1)&(~pred)).sum()),int(((y==1)&pred).sum())]]
mat05 = cm(.5)
matrate = cm(float(y.mean()))
add("Métriques de fréquence et matrice de confusion", "Annexes", "confusion",
    lead="La probabilité continue sert au prix. Un seuil sert uniquement au diagnostic de classification",
    matrices=[{"title":"Seuil 0,5","values":mat05},{"title":"Seuil du taux global : 5,834 %","values":matrate}],
    left=[("Log-loss / Brier", "Qualité des probabilités, plus faible = meilleur"),
          ("AUC", "Qualité de classement, finale : 0,6533")],
    takeaway="Les seuils illustrés sont pédagogiques et ne sont jamais appliqués à la prime",
    notes="Les matrices utilisent les prédictions du diagnostic actuel, chaque client étant exclu de son apprentissage. Les lignes représentent la réalité, les colonnes la décision après application d’un seuil. Le seuil zéro virgule cinq convient mal à un événement rare et peut aboutir à classer tous les contrats comme non sinistrés. Le second seuil est choisi pour l’illustration, égal au taux global, et n’est pas un seuil optimal validé. Nous ne recherchons pas ici un compromis de coût de détection, car notre produit utilise directement les probabilités. La log-loss et le Brier évaluent leur qualité numérique, et l’AUC évalue le classement. La calibration complète ces métriques par des comparaisons observé-prédit en groupes. Aucune matrice binaire ne résume à elle seule une tarification probabiliste.",
    refs=("diag", "main"), appendix=True)

add("La lecture des erreurs de gravité", "Annexes", "table",
    lead="Une prédiction moyenne ne reproduit pas chaque montant exceptionnel",
    rows=[["Diagnostic", "Ce qu’il mesure", "Limite"],
          ["RMSE gravité finale : 1 452,48 €", "Erreur parmi les sinistres positifs exclus", "Différent de la RMSE de prime"],
          ["MAE gravité finale : 930,88 €", "Erreur absolue typique sur ces sinistres", "Moins sensible aux très gros coûts"],
          ["Matrice par tranches de montant", "Observed/predicted dans des classes artificielles", "Diagnostic de régression, pas détection d’accident"],
          ["Calibration par groupe", "Montants moyens prédits et observés", "Ne fournit pas un intervalle individuel"]],
    takeaway="Les extrêmes restent dans la cible. Une prédiction de moyenne peut se concentrer dans une tranche centrale",
    notes="Une matrice de confusion est naturellement une mesure de classification. Pour répondre à la demande initiale sur la gravité, le premier notebook avait construit une matrice par tranches de montants, avec des tertiles appris dans le train. Ces tranches étaient artificielles et ne changeaient pas Gamma en classificateur. Leur intérêt est de montrer comment les montants attendus se concentrent alors que les accidents réellement observés sont beaucoup plus dispersés. Le diagnostic actuel rapporte une RMSE de gravité de 1 452,48 euros et une MAE de 930,88, uniquement sur les sinistres exclus de l’apprentissage. Ces chiffres ne doivent pas être comparés à 1 245,80 de l’ancienne validation de 565 sinistres, car les observations et le protocole diffèrent.",
    refs=("gamma", "diag", "main"), appendix=True)

add("Les paramètres exacts de la recette finale", "Annexes", "table",
    lead="Recette frequency_curves_1365_657, conservée comme modèle principal",
    rows=[["Composante", "Réglages retenus"],
          ["Logistique", "C 0,1, L2, 5 plis d’encodage client, lissage 100"],
          ["Correction fréquence", "α 100, 4 nœuds uniformes, degré 2, interactions ×0,5, poids 1"],
          ["Gamma", "α 0,1, lien log, âge + contrat + usage, catégories rares 20"],
          ["XGBoost ratio", "240 arbres, profondeur 2, lr 0,025, L2 100, L1 1, min_child_weight 100"],
          ["Ridge", "α 10, panneau complet, splines 4 nœuds degré 2, rareté 15 sinistres"],
          ["Mélange gravité", "0,375 Gamma + 0,375 Gamma-XGB + 0,25 Gamma-Ridge"]],
    takeaway="p_final diffère de p_aux utilisé pour pondérer les corrections de gravité",
    notes="Les détails évitent deux confusions importantes. La correction finale de fréquence utilise des nœuds uniformes, alors que certaines anciennes variantes Gamma utilisaient des nœuds par quantiles. Le correcteur XGBoost final est un modèle unique à deux niveaux et deux cent quarante arbres, sans le bagging écarté à trois graines. Les cibles auxiliaires de gravité sont calculées sur trois plis clients et utilisent une logistique compacte distincte de la probabilité finale corrigée. Les poids des correcteurs sont normalisés à moyenne un. Les montants sont protégés par des planchers numériques positifs à dix puissance moins huit. Ces planchers ne correspondent pas à une franchise. La recette n’implique aucun recalage manuel du niveau des primes à partir du leaderboard.",
    refs=("model", "main"), appendix=True, compact=True)

add("Pourquoi les RMSE historiques ne forment pas un classement", "Annexes", "table",
    lead="La cible, les observations et la règle de sélection doivent être identiques pour comparer deux scores",
    rows=[["Valeur", "Périmètre", "Lecture correcte"],
          ["1 491,18 €", "Sélection Gamma, moyenne de 5 plis de gravité", "Optimiste après sélection"],
          ["1 245,80 €", "Ancienne validation de 565 sinistres", "Diagnostic de gravité déjà consulté"],
          ["507,647 €", "9 987 contrats, première fusion", "Ancien diagnostic de prime"],
          ["550,38 € environ", "40 013 contrats de développement", "Ancienne procédure imbriquée"],
          ["541,916 €", "50 000 contrats, recette finale fixe hors client", "Diagnostic actuel de prime"],
          ["1 365,657", "Sous-ensemble public Kaggle", "Score public transmis, privé inconnu"]],
    takeaway="Les gains annoncés dans le récit comparent uniquement des modèles sur le même périmètre",
    notes="Cette annexe est essentielle pour répondre aux questions sur les chiffres. La racine d’une moyenne de carrés n’est pas identique à la moyenne des RMSE de plusieurs plis. Une métrique de gravité se calcule seulement sur les sinistrés, alors qu’une métrique de prime inclut les zéros. La sélection d’une procédure peut choisir un modèle différent dans chaque pli, tandis qu’une recette fixe impose la même configuration partout. Ainsi, 541,9429 de la procédure de sélection Ridge n’est pas 541,9710 de sa recette finale fixe. Nous présentons les écarts locaux uniquement lorsque le comparateur a été évalué sur les mêmes observations. Le public Kaggle et les données locales diffèrent, et les coûts cachés du test ne permettent pas d’attribuer leur écart à une cause précise.",
    refs=("gamma", "v4", "broad", "ridge", "follow", "diag"), appendix=True, compact=True)

add("Questions à préparer pour la soutenance", "Annexes", "columns",
    left=[("Pourquoi pas une régression linéaire ordinaire ?", "Gamma impose une moyenne positive et un lien multiplicatif. Aucun benchmark OLS chiffré n’est revendiqué"),
          ("Pourquoi ne pas apprendre sur les zéros ?", "Les zéros appartiennent à la fréquence et au coût global, pas à la gravité conditionnelle")],
    right=[("Pourquoi ne pas exponentier une moyenne de log-coût ?", "Cela ne restitue généralement pas la moyenne en euros"),
           ("Pourquoi garder des essais rejetés ?", "Pour justifier les retours, mesurer l’incertitude et éviter de réinventer un échec")],
    takeaway="Comprendre les cibles, les plis, les transformations et les unités permet de défendre le projet",
    notes="On peut expliquer la différence avec une régression linéaire ordinaire sans affirmer un avantage empirique qui n’a pas été mesuré. Le Gamma à lien log vise directement une moyenne positive, tandis qu’une régression ordinaire non contrainte peut prévoir des valeurs négatives. Une régression sur log-montant change encore le problème, car l’exponentielle d’une moyenne de logarithmes ne donne pas automatiquement la moyenne du montant. Pour les zéros, l’apprentissage séparé évite de multiplier deux fois une probabilité implicite. Nous prédisons pourtant une gravité pour tous les prospects, car leur statut futur est inconnu. Enfin, conserver les essais ne signifie pas les mélanger tous dans le modèle final : les archives permettent d’expliquer les choix et de retrouver exactement la recette qui a produit un fichier.",
    refs=("td", "main", "skgamma"), appendix=True)

add("Reproduction et organisation du livrable", "Annexes", "columns",
    lead="Les anciens essais restent archivés, le notebook principal décrit la recette retenue",
    left=[("Point de lecture", "modele_principal.ipynb"),
          ("Exécution", "run.py : train, diagnostic, predict, verify"),
          ("Modèle local", "models/main_model.joblib")],
    right=[("Environnement", "Python 3.12, scikit-learn 1.9.1, XGBoost 3.4.1"),
           ("Traçabilité", "Paramètres, versions, empreintes et reçus de soumission"),
           ("Contrôles actuels", "35 tests actifs et reproduction des 50 000 primes")],
    takeaway="La reproduction numérique du fichier ne remplace pas une nouvelle mesure de performance sur données fraîches",
    notes="Le projet a changé d’organisation pour rendre la recette finale visible. Les anciens notebooks, rapports et soumissions sont conservés dans archive. Le notebook principal raconte les données, le code exact et les diagnostics du modèle actuel. Les commandes du point d’entrée séparent apprentissage, diagnostic, prédiction et vérification. Les modèles sérialisés sont locaux et les rapports enregistrent leurs versions et empreintes. Les trente-cinq tests mentionnés appartiennent au périmètre actif actuel, et ne représentent pas tous les anciens tests de benchmark archivés. Le rechargement reproduit les cinquante mille primes dans une tolérance numérique très serrée. Cette vérification confirme que le code et le fichier sont cohérents. Elle ne donne pas un score Kaggle à un nouveau fichier et ne démontre pas sa généralisation future.",
    refs=("main", "model", "analysis"), appendix=True)

add("La roadmap de référence", "Annexes", "reference_image",
    lead="Le récit suit le schéma fourni, y compris les retours vers les données, le traitement et la modélisation",
    image="deliverables/presentation_histoire/assets/roadmap_reference.png",
    takeaway="Les notes de chaque diapositive identifient ses sources précises et les limites de son résultat",
    notes="Le schéma original est repris comme référence visuelle, sans modifier ses instructions. Le cadrage et l’analyse occupent le début de l’exposé. Le traitement regroupe les règles d’anomalies, les encodages, les transformations et les découpages. La partie modélisation retrace les références, les catalogues de recherche et les mélanges. La validation mène aux retours documentés : abandon de complexité, retour au témoin, réanalyse des résidus et réintroduction de variables. La soumission conclut le parcours. L’énoncé insiste sur la compréhension et la justification de l’ensemble des étapes, ce qui motive le guide oral et les annexes. Les sources techniques externes sont les documentations officielles, tandis que les résultats propres au projet proviennent des artefacts et de Git.",
    refs=("td", "roadmap"), appendix=True)


def main():
    payload={"title":"L’histoire de notre modèle de tarification automobile",
             "date":"2026-10-20", "main_slide_count":MAIN_COUNT,
             "appendix_slide_count":len(slides)-MAIN_COUNT,
             "estimated_seconds":sum(s["seconds"] for s in slides),
             "score_status":"Meilleur score public observé dans le registre ; scores du 12 octobre transmis par l’utilisateur, non revérifiés sur Kaggle. Score privé inconnu.",
             "sources":sources,"historical_proofs":proof_manifest,"slides":slides}
    for s in slides:
        assert s["notes"] and s["refs"]
        assert all(r in sources for r in s["refs"])
    (OUT/"contenu_presentation.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    guide=["# Guide oral : histoire du modèle de tarification automobile", "",
           f"{MAIN_COUNT} diapositives principales, environ {payload['estimated_seconds']/60:.1f} minutes, puis {len(slides)-MAIN_COUNT} annexes.",
           "Les temps indiqués sont des repères, pas un texte à lire mécaniquement. Les annexes servent aux questions.",
           "", "## Répartition conseillée", "",
           "La personne chargée de la gravité présente les premières comparaisons Gamma, la sélection de variables et les corrections de gravité. La personne chargée de la fréquence présente sa préparation, l’audit d’encodage et la correction de probabilités. Présenter ensemble le cadrage, la validation commune et le résultat final.", ""]
    for s in slides:
        guide += [f"## {s['number']:02d}. {s['title']}","",
                  f"**Roadmap : {s['stage']}**"+(" — annexe" if s['appendix'] else f" — {s['seconds']} secondes"),"",
                  s["notes"],"", "**Message à retenir :** "+s["takeaway"], "", "**Sources :**"]
        for r in s["refs"]:
            ref=sources[r]
            guide += [f"- {ref['label']} : {ref.get('url',ref.get('path'))}"]
        guide += [""]
    (OUT/"guide_oral.md").write_text("\n".join(guide),encoding="utf-8")
    print(json.dumps({"main_slides":MAIN_COUNT,"appendices":len(slides)-MAIN_COUNT,
                      "minutes":payload["estimated_seconds"]/60,
                      "main_notes_words":sum(len(s["notes"].split()) for s in slides if not s["appendix"])},ensure_ascii=False))


if __name__=="__main__":main()
