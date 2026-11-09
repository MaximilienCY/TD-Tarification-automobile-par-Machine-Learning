# Schémas du modèle principal

Quatre vues au format 16:9 expliquent la recette retenue
`frequency_curves_1365_657`. Elles décrivent le modèle actuel, sans réentraîner
les modèles ni modifier la soumission.

- **[Présentation PDF, quatre pages](../../output/pdf/schema_modele_principal.pdf)** :
  version vectorielle pour présenter et partager.
- **[Fichier Excalidraw modifiable](schema_modele_principal.excalidraw)** : quatre
  cadres, avec des textes, formes et flèches éditables. Dans
  [Excalidraw](https://excalidraw.com/), utiliser le menu puis « Open » pour ouvrir
  ce fichier local. Sauvegarder un éventuel tableau existant avant de le remplacer.
- **[Vue d’ensemble](schema-1.png)** : données, deux branches, prime et soumission.
- **[Fréquence](schema-2.png)** : variables, logistique, correction et calibration.
- **[Gravité](schema-3.png)** : Gamma, corrections XGBoost/Ridge et exemple chiffré.
- **[Validation et limites](schema-4.png)** : plis par client, RMSE, forces,
  faiblesses et prochaines étapes.

Les quatre fichiers `.svg` du dossier offrent les mêmes vues en vectoriel.
Les exports PDF/PNG préservent la typographie de présentation ; Excalidraw utilise
sa police standard pour garder les textes facilement modifiables.

## Fil conducteur pour la présentation orale

**1. Partir de la question métier.** Pour un prospect, on ne sait pas encore s’il
aura un sinistre. On estime donc sa probabilité et le montant moyen qu’il coûterait
si un sinistre survenait. Leur produit est le coût attendu, appelé ici prime pure.

**2. Expliquer la fréquence.** La logistique calcule une probabilité initiale `p₀`.
Une correction utilise des courbes d’âge du véhicule, le contrat et l’usage pour
l’ajuster. Les splines sont des bases qui représentent une courbe souple ; elles
n’imposent pas un effet d’âge constant. La fonction `logit(p) = ln(p / (1 - p))`
permet d’ajouter la correction, puis `sigmoid(z) = 1 / (1 + exp(-z))` ramène la
sortie entre 0 et 1. Les catégories et imputations sont apprises sans utiliser
les clients sur lesquels on mesure l’erreur.

**3. Expliquer la gravité.** L’apprentissage se limite aux 2 917 sinistres
positifs. Gamma fournit un montant positif à partir du contrat, de l’âge du
véhicule et de l’usage. XGBoost corrige ce montant grâce à de petits arbres ;
Ridge le corrige avec davantage de variables et une pénalisation des coefficients.
On moyenne les trois montants avec les poids 37,5 %, 37,5 % et 25 %.
Le département est donc bien utilisé dans la fréquence et dans Ridge, tandis que
Gamma et sa correction XGBoost utilisent les trois variables métier.

**4. Dire comment on apprend et vérifie.** Dans les trois plis de diagnostic,
chaque client reste entièrement dans un seul pli. Les modèles apprennent sur les
autres clients et prédisent ceux qui sont exclus. « Hors client » signifie que
la prédiction auxiliaire d’une ligne vient d’un modèle n’ayant appris sur aucune
ligne de ce client. Le diagnostic actuel reproduit une recette déjà choisie ;
il ne crée pas une nouvelle évaluation indépendante de toute la recherche.

**5. Terminer par le résultat concret.** Après ces vérifications, la fréquence
apprend sur tous les contrats et la gravité sur tous les sinistres positifs du
train. Pour chacun des 50 000 prospects du test, on prédit une probabilité, une
gravité conditionnelle et leur produit. On exporte `index,pred`, avec les index
50 000 à 99 999. `pred` représente une prime, tandis que la RMSE mesure l’erreur
globale d’un ensemble de prédictions dont les coûts réels sont connus.

L’exemple de la troisième vue est **fictif** : avec une gravité de 2 300 € et une
probabilité de 8 %, la prime attendue est de 184 €. Cela ne veut pas dire que
le prospect aura un sinistre de 184 €.

## Sources et reproduction

Les paramètres et résultats proviennent de
[l’analyse détaillée](../modele_principal.md), du
[notebook principal](../../modele_principal.ipynb), des
[métadonnées du modèle](../../reports/final_model.json) et du
[diagnostic réexécuté](../../reports/main/diagnostic.json).
Le score public de 1 365,657 est celui communiqué par l’utilisateur dans le reçu
historique ; il n’a pas été revérifié sur Kaggle. La RMSE locale de 541,916 €
provient des clients exclus du diagnostic.

Le format éditable suit le
[schéma de fichier officiel Excalidraw](https://docs.excalidraw.com/docs/codebase/json-schema).
Les détails internes des éléments sont importables par l’éditeur ; aucune
publication ni création de lien de partage n’est nécessaire.

Pour régénérer le PDF, les SVG et le fichier éditable avec Python, installer
`reportlab` dans l’environnement de travail et lancer :

```sh
python scripts/build_model_schema.py
pdftoppm -png -scale-to-x 1920 -scale-to-y 1080 output/pdf/schema_modele_principal.pdf docs/schemas/schema
```

La génération du PDF nécessite les polices DejaVu Sans du système. Le fichier
`provenance.json` rattache les vues aux métadonnées exactes utilisées.
