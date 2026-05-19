import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# ── Chemins ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "water_potability.csv"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
TARGET = "Potability"

# ── Configuration MLflow ──────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("experiment_water_quality")

# ── Chargement et split ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Paramètres du modèle ──────────────────────────────────────────────────────
rf_params = {
    "n_estimators": 100,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": 42,
}

# ── Pipeline : imputation médiane + Random Forest ─────────────────────────────
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(**rf_params)),
])

# ── Run MLflow ────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="RandomForest_baseline"):

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_proba)

    # Log des paramètres
    for param, value in rf_params.items():
        mlflow.log_param(param, value)
    mlflow.log_param("imputer_strategy", "median")

    # Log des métriques
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Log du pipeline complet
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

# ── Récapitulatif terminal ────────────────────────────────────────────────────
print(f"\n{'Modèle':<10} | {'Accuracy':<8} | {'F1':<8} | {'AUC-ROC':<8}")
print("-" * 46)
print(f"{'RF':<10} | {accuracy:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")
