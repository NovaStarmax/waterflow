import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
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

# ── Run 2 : Dummy Classifier (baseline naïf) ──────────────────────────────────
dummy_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DummyClassifier(strategy="most_frequent")),
])

with mlflow.start_run(run_name="Dummy_baseline"):

    dummy_pipeline.fit(X_train, y_train)
    y_pred_dummy = dummy_pipeline.predict(X_test)
    y_proba_dummy = dummy_pipeline.predict_proba(X_test)[:, 1]

    # Métriques
    accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
    f1_dummy = f1_score(y_test, y_pred_dummy, average="binary")
    auc_dummy = roc_auc_score(y_test, y_proba_dummy)

    # Log des paramètres
    mlflow.log_param("strategy", "most_frequent")

    # Log des métriques
    mlflow.log_metric("accuracy", accuracy_dummy)
    mlflow.log_metric("f1_score", f1_dummy)
    mlflow.log_metric("roc_auc", auc_dummy)

    # Log du pipeline complet
    mlflow.sklearn.log_model(dummy_pipeline, artifact_path="model")

# ── Run 3 : XGBoost ───────────────────────────────────────────────────────────
# Rééquilibrage des classes via scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

xgb_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", XGBClassifier(**xgb_params)),
])

with mlflow.start_run(run_name="XGBoost_baseline"):

    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

    # Métriques
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average="binary")
    auc_xgb = roc_auc_score(y_test, y_proba_xgb)

    # Log des paramètres
    mlflow.log_param("n_estimators", xgb_params["n_estimators"])
    mlflow.log_param("max_depth", xgb_params["max_depth"])
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))

    # Log des métriques
    mlflow.log_metric("accuracy", accuracy_xgb)
    mlflow.log_metric("f1_score", f1_xgb)
    mlflow.log_metric("roc_auc", auc_xgb)

    # Log du pipeline complet
    mlflow.sklearn.log_model(xgb_pipeline, artifact_path="model")

# ── Récapitulatif terminal ────────────────────────────────────────────────────
print(f"\n{'Modèle':<10} | {'Accuracy':<8} | {'F1':<8} | {'AUC-ROC':<8}")
print("-" * 46)
print(f"{'Dummy':<10} | {accuracy_dummy:<8.4f} | {f1_dummy:<8.4f} | {auc_dummy:<8.4f}")
print(f"{'RF':<10} | {accuracy:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")
print(f"{'XGBoost':<10} | {accuracy_xgb:<8.4f} | {f1_xgb:<8.4f} | {auc_xgb:<8.4f}")
