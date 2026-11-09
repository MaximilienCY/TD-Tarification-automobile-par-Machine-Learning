# Deuxième recherche : objectif Kaggle inférieur à 1365.500

Recherche du 11 octobre 2026. Cet objectif est une cible, pas un résultat obtenu.
Le notebook protégé et le pipeline de la première livraison restent inchangés.

**Recommandation actualisée après retour utilisateur : conserver la référence initiale logistique × Gamma (score public vérifié précédemment : 1365.943).** L'utilisateur indique que les challengers sont moins bons sur Kaggle ; les nouveaux scores exacts n'ont pas été fournis. Le fichier original a été récupéré à l'identique dans `deliverables/candidates/submission_reference.csv`. Son SHA-256 `78515c47fdc519634175ff36b1a82e41a325713d0e3aea4090d356ff53d6df8c` correspond au fichier de la meilleure soumission précédemment téléchargée. L'ordre d'essais ci-dessous documente la proposition antérieure et ne constitue plus une recommandation de remplacement.

## Sources et décisions

- [Comparaison actuarielle des algorithmes de boosting, Chevalier et Côté](https://arxiv.org/abs/2412.14916) : comparaison de modèles de fréquence et de gravité sur cinq jeux de données. Motive l'ajout de XGBoost et d'un modèle additif EBM, en complément de CatBoost et LightGBM.
- [InterpretML : Explainable Boosting Machine](https://interpret.ml/docs/ebm.html) : modèle additif de boosting avec interactions possibles. Ici, comparaison de versions additives et avec interactions, sans sélection anticipée à partir des données du pli externe.
- [Objectifs XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html) : objectif binaire pour la fréquence, `reg:gamma` pour la gravité positive.
- [Objectifs CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-regression) : comparaison de pertes Poisson et Tweedie pour la gravité, avec cibles normalisées à l'apprentissage et prédiction exponentielle.
- [Exemple actuariel scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html) : situe les modèles séparés fréquence/gravité et les modèles directs de prime. Les candidats de cette livraison conservent deux composantes distinctes conformément à l'exercice.
- [TabPFN, dépôt officiel](https://github.com/PriorLabs/TabPFN) : les modèles de fondation tabulaires constituent une autre famille intéressante, mais ne garantissent pas le meilleur score sur ces données. Un benchmark local GPU de la gravité avec les poids publics v2 est prévu dans un environnement isolé. Cette version n'est pas présentée comme la version la plus récente.

## Protocole

`scripts/advanced_search.py` compare huit nouveaux candidats de fréquence et quatorze de gravité. Il utilise les cinq plis par client de la première recherche, les mêmes lignes et des mélanges prédéfinis avec le modèle courant (25 %, 50 %, 75 %, 100 %). Au total : 1 881 couples.

`scripts/modern_search.py` compare six candidats EBM/XGBoost de fréquence et cinq de gravité, soit 525 couples avec les mêmes règles de mélange.

Les deux meilleurs couples distincts sont ensuite évalués sur une autre partition en cinq plis par client. **Ce contrôle de sensibilité réutilise les mêmes contrats : ce n'est ni une validation indépendante, ni une validation imbriquée de la nouvelle recherche.** Le diagnostic historique est calculé seulement après sélection des couples et n'est pas utilisé pour les choisir. Les résultats de validation imbriquée de la première livraison ne sont pas attribués aux nouveaux candidats.

Toutes les gravités sont apprises sur les sinistres positifs. Les identifiants, cibles et sexes restent exclus. Les coûts extrêmes ne sont pas supprimés. Aucune transformation des prédictions n'est ajustée au leaderboard.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe scripts/advanced_search.py search
.\.venv\Scripts\python.exe scripts/advanced_search.py confirm
.\.venv\Scripts\python.exe scripts/advanced_search.py export
.\.venv\Scripts\python.exe scripts/modern_search.py search
.\.venv\Scripts\python.exe scripts/modern_search.py confirm
.\.venv\Scripts\python.exe scripts/nested_modern.py
.\.venv\Scripts\python.exe scripts/export_modern.py
.\.venv\Scripts\python.exe scripts/export_modern.py predict
```

Les résultats détaillés, configurations, empreintes et diagnostics sont dans `reports/advanced/` et `reports/modern/`. Les fichiers candidats sont séparés dans `deliverables/candidates/`, pour préserver la soumission précédente.

## Résultats des candidats fixes

| Candidat | Recherche sur cinq plis | Autre partition en cinq plis | Diagnostic historique |
| --- | ---: | ---: | ---: |
| Modèle courant | 550,065781 | 550,185015 | 507,318911 |
| CatBoost peu profond / LightGBM, candidat 1 | 549,988963 | 550,215978 | 507,419525 |
| CatBoost peu profond / LightGBM, candidat 2 | 549,994485 | 550,280285 | 507,516322 |
| EBM / XGBoost, candidat 1 | 549,916937 | 550,123037 | 507,419311 |
| EBM / XGBoost, candidat 2 | 549,945486 | 550,106625 | 507,355223 |

Les deux candidats EBM/XGBoost gagnent sur les deux partitions du développement, mais pas sur le diagnostic historique. Les intervalles descriptifs à 95 % du gain sur la seconde partition contiennent zéro : [-0,083154 ; 0,206406] et [-0,053823 ; 0,208017]. Ces résultats motivent un essai Kaggle, sans garantir un score inférieur à 1365.500.

Les fichiers `submission_modern_1.csv` et `submission_modern_2.csv` contiennent chacun 50 000 lignes valides. Les modèles sérialisés ont été rechargés ; une nouvelle exécution reproduit les fichiers à l'octet près. Les huit tests initiaux et les six tests de recherche passent. Les vérifications sont consignées dans `reports/research_verification.json`.

La procédure `nested_modern.py` refait la sélection des 525 couples dans trois plis internes, puis évalue sur trois plis externes. Son benchmark est la recette courante fixée. Elle corrige le biais de cette nouvelle sélection, mais ne supprime pas les effets des explorations antérieures du développement.

**Résultat imbriqué : 550,299265 pour la nouvelle recherche, contre 550,284880 pour la recette courante fixe.** Le gain est donc négatif (-0,014384), avec un intervalle descriptif à 95 % [-0,138602 ; 0,114446]. La recherche plus vaste n'établit pas de supériorité robuste. Le modèle courant est conservé ; les nouveaux CSV restent des challengers. Le score 550,355148 de la première livraison évaluait une autre procédure de sélection et ne doit pas être confondu avec le benchmark fixe ci-dessus.

## Benchmark TabPFN local

Le script `scripts/foundation_search.py` teste les poids publics **TabPFN-v2**, avec quatre estimateurs sur CUDA, sur la gravité uniquement. Il compare trois variables métier à huit variables, puis des mélanges prédéfinis avec la gravité Gamma courante. Les probabilités de fréquence hors pli restent celles du modèle courant. Les données ne sont pas envoyées à un service d'inférence ; la télémétrie est désactivée. Attribution : **Built with TabPFN — Prior Labs** ; [licence des poids](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt).

Cette expérience utilise un environnement séparé, car TabPFN 2.2.1 dépend de versions de pandas et scikit-learn différentes du pipeline principal. Sur une machine Windows avec GPU NVIDIA compatible :

```powershell
python -m venv .venv-foundation
.\.venv-foundation\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-foundation\Scripts\python.exe -m pip install -r requirements-foundation-lock.txt
New-Item -ItemType Directory -Force .cache/tabpfn-models
Invoke-WebRequest -Uri 'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt' -OutFile '.cache/tabpfn-models/tabpfn-v2-regressor.ckpt'
.\.venv-foundation\Scripts\python.exe scripts/foundation_search.py
```

Les résultats, versions et empreintes se trouvent dans `reports/foundation/`.

Le benchmark GPU a été exécuté intégralement : deux jeux de variables, cinq plis par client, quatre poids de mélange. Le meilleur résultat est **550,069976**, contre **550,065781** pour la gravité Gamma courante. TabPFN seul obtient 550,152909 avec trois variables et 550,177627 avec huit. Aucun de ces huit mélanges n'améliore le benchmark ; aucune soumission TabPFN n'est donc retenue.

## Ordre proposé des essais Kaggle

1. `deliverables/candidates/submission_modern_1.csv` : meilleur candidat de la recherche EBM/XGBoost ; amélioration sur les deux partitions de développement.
2. `deliverables/candidates/submission_modern_2.csv` : alternative plus simple, meilleure sur la seconde partition et moins dégradée sur le diagnostic historique.

Les autres fichiers restent disponibles pour traçabilité. La validation imbriquée ne démontrant pas de gain robuste, le modèle courant et `submission.csv` ne sont pas remplacés. Les candidats sont des essais à évaluer, pas des meilleurs modèles Kaggle confirmés.

## Accès Kaggle

Le contrôle du navigateur a refusé l'accès à Kaggle en raison d'une préférence utilisateur enregistrée. Aucun contournement n'est utilisé. Aucun nouveau score ne peut être confirmé tant que cette permission n'est pas rétablie ou que l'utilisateur ne soumet pas lui-même les fichiers et ne transmet pas les résultats.
