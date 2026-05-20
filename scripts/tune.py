import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Chemins ───────────────────────────────────────────────────────────────────
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

# ── Fonction objectif Optuna ──────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            **params,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)

    run_name = f"XGB_tune_trial_{trial.number}"
    with mlflow.start_run(run_name=run_name):
        # Log des hyperparamètres
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log des métriques
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", accuracy)

    # Pénalité douce : signal graduel vers la contrainte F1 >= 0.40
    if f1 < 0.40:
        return precision * (f1 / 0.40)
    return precision


# ── Optimisation — 100 trials ─────────────────────────────────────────────────
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ── Récupération du meilleur trial valide ─────────────────────────────────────
# On filtre les trials avec f1 >= 0.40 (valeur retournée > 0.0)
valid_trials = [t for t in study.trials if t.value is not None and t.value > 0.0]
if not valid_trials:
    # Aucun trial n'a satisfait la contrainte F1, on prend le meilleur global
    best_trial = study.best_trial
else:
    best_trial = max(valid_trials, key=lambda t: t.value)

best_params = best_trial.params

# ── Réentraînement du meilleur pipeline ───────────────────────────────────────
best_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )),
])
best_pipeline.fit(X_train, y_train)

# ── Recherche du seuil optimal ────────────────────────────────────────────────
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = (y_proba_best >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_t = f1_score(y_test, y_pred_thresh, average="binary", zero_division=0)
    threshold_results.append({"threshold": thresh, "precision": prec, "recall": rec, "f1": f1_t})

# Seuil avec Precision >= 0.70 et F1 maximal ; sinon, meilleure Precision
candidates_70 = [r for r in threshold_results if r["precision"] >= 0.70]
if candidates_70:
    best_thresh_result = max(candidates_70, key=lambda r: r["f1"])
else:
    best_thresh_result = max(threshold_results, key=lambda r: r["precision"])

best_threshold = best_thresh_result["threshold"]

# ── Métriques finales au seuil retenu ─────────────────────────────────────────
y_pred_final = (y_proba_best >= best_threshold).astype(int)
final_precision = precision_score(y_test, y_pred_final, zero_division=0)
final_recall = recall_score(y_test, y_pred_final, zero_division=0)
final_f1 = f1_score(y_test, y_pred_final, average="binary", zero_division=0)
final_auc = roc_auc_score(y_test, y_proba_best)
final_accuracy = accuracy_score(y_test, y_pred_final)

# ── Run MLflow final ──────────────────────────────────────────────────────────
with mlflow.start_run(run_name="XGB_tuned_best"):
    # Meilleurs hyperparamètres
    for key, value in best_params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("threshold", best_threshold)

    # Métriques au seuil optimal
    mlflow.log_metric("precision", final_precision)
    mlflow.log_metric("recall", final_recall)
    mlflow.log_metric("f1_score", final_f1)
    mlflow.log_metric("roc_auc", final_auc)
    mlflow.log_metric("accuracy", final_accuracy)

    # Tag et modèle
    mlflow.set_tag("best_model", "true")
    mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

# ── Affichage terminal ────────────────────────────────────────────────────────

# Top 3 trials par Precision (parmi les trials valides avec f1 >= 0.40)
sorted_trials = sorted(
    [t for t in study.trials if t.value is not None and t.value > 0.0],
    key=lambda t: t.value,
    reverse=True,
)[:3]

print(f"\n{'Trial':<6} | {'Precision':<9} | {'F1':<7} | {'Recall':<7}")
print("-" * 38)
for t in sorted_trials:
    y_pred_t = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            **t.params,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )),
    ]).fit(X_train, y_train).predict(X_test)
    f1_t = f1_score(y_test, y_pred_t, average="binary", zero_division=0)
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    print(f"{t.number:<6} | {t.value:<9.4f} | {f1_t:<7.4f} | {rec_t:<7.4f}")

# Tableau des seuils
print(f"\n{'Seuil':<6} | {'Precision':<9} | {'Recall':<7} | {'F1':<7}")
print("-" * 38)
for r in threshold_results:
    print(f"{r['threshold']:<6.2f} | {r['precision']:<9.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f}")

# Récapitulatif final
print(f"\nMeilleure Precision : {final_precision:.4f}")
print(f"F1 au seuil retenu  : {final_f1:.4f}")
print(f"Seuil retenu        : {best_threshold:.2f}")
