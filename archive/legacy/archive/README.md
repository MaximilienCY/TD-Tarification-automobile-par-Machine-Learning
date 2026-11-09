# Versions antérieures et preuves historiques

Le modèle actif se lit dans `../modele_principal.ipynb` et se pilote avec `../run.py`.
Il correspond au CSV associé au score public rapporté de **1365,657**.

Ce dossier conserve les anciens notebooks, rapports, soumissions et présentation.
Leur contenu a été préservé lors du déplacement ; `organisation.json` décrit les
chemins déplacés et la vérification de leurs octets. Les références relatives à
l'intérieur des notebooks sont celles de leurs versions historiques. Ces
instantanés ne constituent plus les points d'entrée actifs.

- `notebooks/` : sept notebooks de recherche et le notebook historique protégé.
- `reports/` : résultats et reçus originaux, y compris les expériences écartées.
- `deliverables/` : anciens CSV candidats et ancienne présentation.
- `submission_avant_promotion.csv` : CSV racine qui provenait de l'ancien modèle.
- `output/` : ancien cache local, conservé mais ignoré par Git.
- `run_legacy.py` : entrée des anciennes recherches ; ses sorties restent ici.
- `tests/` : contrôles propres aux anciens benchmarks ; leurs dépendances
  optionnelles, comme CatBoost, LightGBM ou EBM, ne sont pas requises pour le modèle actif.
- `requirements/` : versions des anciens environnements et dépendances de recherche.

Les bibliothèques partagées dans `../src/` et les modules dans `../scripts/`
restent à leur place parce que la recette actuelle réutilise leurs composants.
Leurs commandes de recherche écrivent désormais dans `archive/` et ne remplacent
plus la soumission principale. Elles peuvent demander des dépendances historiques
supplémentaires décrites dans `archive/requirements/`.

Le notebook `Calcul de la prime d'assurance.ipynb` conserve exactement les octets
qu'il avait avant ce classement. Le contrôle accepte les deux représentations LF
et CRLF connues du même contenu. Les reçus Windows d'origine ne sont pas réécrits :
une empreinte de fichier contenant des CRLF diffère de celle du même CSV sous Linux.
