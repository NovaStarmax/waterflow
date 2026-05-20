## Conclusion — Démarche Data Science

### Contexte et objectif

L'objectif de ce projet est de prédire la potabilité de l'eau à partir de 9 mesures physico-chimiques (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity) sur un dataset de 3276 échantillons. L'enjeu est sanitaire : une mauvaise prédiction dans le sens "potable" alors que l'eau ne l'est pas expose directement des personnes à un risque de santé.

### Choix méthodologiques

L'analyse exploratoire a révélé trois contraintes structurelles du dataset : un déséquilibre de classes (61% non potable / 39% potable), des valeurs manquantes sur ph (~15%), Sulfate (~24%) et Trihalomethanes (~5%), et surtout de très faibles corrélations entre les features et la target (|r| < 0.1). Ce dernier point signifie qu'aucune feature ne prédit seule la potabilité — le problème est intrinsèquement difficile.

Face à l'enjeu sanitaire, nous avons défini la **Precision** comme métrique prioritaire. Dire qu'une eau est potable alors qu'elle ne l'est pas constitue le cas le plus dangereux (Faux Positif). À l'inverse, rejeter une eau potable est un gaspillage sans conséquence sanitaire. Cette asymétrie justifie de privilégier la Precision au détriment du Recall.

### Modélisation

Trois modèles ont été entraînés et comparés via MLflow :

| Modèle | Precision | F1 | Recall |
|---|---|---|---|
| DummyClassifier | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.65 | 0.40 | 0.29 |
| XGBoost (tuné) | **0.73** | **0.41** | 0.29 |

Le choix s'est porté sur des modèles à base d'arbres de décision (Random Forest, XGBoost), naturellement robustes aux outliers et aux faibles corrélations, sans nécessiter de normalisation préalable. Le deep learning (MLP) a été écarté car inadapté à des données tabulaires avec si peu de signal discriminant.

### Tuning et seuil de décision

L'optimisation des hyperparamètres XGBoost via Optuna (100 trials) a permis d'améliorer la Precision de 0.57 à **0.73** avec une contrainte de F1 ≥ 0.40. Le seuil de décision retenu est 0.50 — au-delà, la Precision progresse mais le Recall s'effondre à des niveaux rendant le modèle inutilisable.

### Limites et honnêteté des résultats

Ce dataset est connu pour sa difficulté. Les meilleures solutions publiées sur Kaggle plafonnent à F1 ≈ 0.65–0.68 avec un feature engineering poussé. Nos résultats (Precision 0.73, F1 0.41) sont honnêtes et non surestimés — aucun data leakage n'a été introduit grâce à l'utilisation systématique de Pipelines scikit-learn intégrant l'imputation.

Le modèle final a été versionné dans le MLflow Model Registry (`water_quality_classifier_xgboost v1`) et déployé via une API FastAPI pour des prédictions en temps réel.