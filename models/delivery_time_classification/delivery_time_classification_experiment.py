"""Run the end-to-end experiment: train tuned logistic model, evaluate, save artifacts.
Usage: python run_experiment.py
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import joblib
import matplotlib.pyplot as plt
import shap

# reproducibility
RANDOM_STATE = 42

PATH = Path.cwd()
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

REPORTS_DIR = PATH / "reports"
MODELS_DIR = PATH / "models"
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

query = """
select
    order_ts,
    delivery_time,
    distance_km,
    order_hour,
    order_dow,
    avg_rider_rating_hist,
    user_zone,
    rider_zone
from public.fct_delivery_time
where avg_rider_rating_hist is not null
order by order_ts
"""

df = pd.read_sql(query, engine)
df.sort_values("order_ts", inplace=True)
df["is_delayed"] = (df["delivery_time"] > df["delivery_time"].mean()).astype(int)

num_cols = [
    "distance_km",
    "order_hour",
    "avg_rider_rating_hist",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "zone_load_rolling",
]
cat_cols = ["user_zone", "rider_zone", "order_dow", "same_zone", "is_weekend"]

# feature engineering (same as notebook)
df["same_zone"] = (df["user_zone"] == df["rider_zone"]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["order_dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["order_dow"] / 7)
df["is_weekend"] = df["order_dow"].apply(lambda x: 1 if x in [0, 6] else 0)
df["zone_load_rolling"] = df.groupby("user_zone")["is_delayed"].transform(lambda x: x.rolling(window=5, closed="left").mean())
df["zone_load_rolling"] = df["zone_load_rolling"].fillna(df["is_delayed"].mean())

user_zones = sorted(df["user_zone"].astype(str).unique().tolist())
rider_zones = sorted(df["rider_zone"].astype(str).unique().tolist())

categories_list = [
    user_zones,
    rider_zones,
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    sorted(df["same_zone"].unique().tolist()),
    sorted(df["is_weekend"].unique().tolist()),
]

preprocess = ColumnTransformer([
    ("num", PowerTransformer(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", categories=categories_list), cat_cols),
])

model = LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE)
pipe = Pipeline([("prep", preprocess), ("model", model)])

tss = TimeSeriesSplit(n_splits=5)
param_grid = {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]}  # simpler grid for quick runs

grid = GridSearchCV(pipe, param_grid, cv=tss, scoring="roc_auc", n_jobs=1)

# fit
grid.fit(df[num_cols + cat_cols], df["is_delayed"])
best = grid.best_estimator_

# predictions (on full set for reporting purposes — for production use keep a holdout)
X = df[num_cols + cat_cols]
y = df["is_delayed"]
probs = best.predict_proba(X)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y, probs)),
    "pr_auc": float(average_precision_score(y, probs)),
}

# save metrics
with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save model
joblib.dump(best, MODELS_DIR / "best_model.joblib")

# SHAP
prep = best.named_steps["prep"]
model_final = best.named_steps["model"]
X_trans = prep.transform(X)
explainer = shap.Explainer(model_final, X_trans)
shap_values = explainer(X_trans)
joblib.dump(shap_values, REPORTS_DIR / "shap_values.joblib")

# plots: ROC and PR
fpr, tpr, _ = roc_curve(y, probs)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.savefig(REPORTS_DIR / "roc.png")
plt.close()

precision, recall, _ = precision_recall_curve(y, probs)
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {metrics['pr_auc']:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall")
plt.legend()
plt.savefig(REPORTS_DIR / "pr_curve.png")
plt.close()

print("Artifacts saved to:")
print(REPORTS_DIR)
print(MODELS_DIR)
print(metrics)
