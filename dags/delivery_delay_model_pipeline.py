"""
Airflow DAG for daily delivery delay classification model training and evaluation.
Trains logistic regression model on temporal features to predict delivery delays.
Runs daily and generates confusion matrices, metrics, and model artifacts.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv

# ML imports
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Metrics & Visualization
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from sqlalchemy import create_engine

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Paths and configs
PROJ_PATH = Path("/home/kheaw/projects/food-delivery-data-platform")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_OUTPUT_DIR = PROJ_PATH / "models" / "delivery_time_classification"
REPORTS_DIR = MODEL_OUTPUT_DIR / "reports"
MODELS_DIR = MODEL_OUTPUT_DIR / "models"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_data_from_db(**context):
    """Fetch delivery data from PostgreSQL database."""
    logger.info("Fetching data from database...")

    engine = create_engine(DATABASE_URL)

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
    engine.dispose()

    logger.info(f"Fetched {len(df)} records from database")

    # Push to XCom for downstream tasks
    context["task_instance"].xcom_push(key="df_shape", value=df.shape)

    # Save to temp location for next task
    temp_file = f"/tmp/delivery_data_{context['execution_date']}.pkl"
    df.to_pickle(temp_file)
    context["task_instance"].xcom_push(key="data_file", value=temp_file)

    return temp_file


def preprocess_data(data_file: str, **context):
    """Preprocess data: define target, engineer features."""
    logger.info("Preprocessing data...")

    df = pd.read_pickle(data_file)
    df.sort_values("order_ts", inplace=True)

    # Define target: is_delayed based on distance-normalized threshold
    df["expected_time"] = df.groupby(pd.qcut(df["distance_km"], 10))[
        "delivery_time"
    ].transform("median")

    threshold_factor = 1.33
    df["is_delayed"] = (
        df["delivery_time"] > (df["expected_time"] * threshold_factor)
    ).astype(int)

    logger.info(f"Target distribution: {df['is_delayed'].value_counts().to_dict()}")

    # Feature engineering
    df["same_zone"] = (df["user_zone"] == df["rider_zone"]).astype(int)

    # Cyclical encoding for temporal features
    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["order_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["order_dow"] / 7)

    # Hour binning for granular temporal signal
    def bin_hour(hour):
        if hour in [22, 23, 0]:
            return "night"
        elif 1 <= hour <= 5:
            return "late_night"
        elif 6 <= hour <= 8:
            return "early morning"
        return str(hour)

    df["hour_granular"] = df["order_hour"].apply(bin_hour)

    # Rolling zone load (recent delays per zone)
    df["zone_load_rolling"] = df.groupby("user_zone")["is_delayed"].transform(
        lambda x: x.rolling(window=5, closed="left").mean()
    )
    df["zone_load_rolling"] = df["zone_load_rolling"].fillna(df["is_delayed"].mean())

    logger.info("Feature engineering complete")

    # Save preprocessed data
    processed_file = f"/tmp/delivery_data_processed_{context['execution_date']}.pkl"
    df.to_pickle(processed_file)
    context["task_instance"].xcom_push(key="processed_data_file", value=processed_file)

    return processed_file


def train_model(processed_data_file: str, **context):
    """Train tuned logistic regression model with GridSearchCV."""
    logger.info("Training model with hyperparameter tuning...")

    df = pd.read_pickle(processed_data_file)

    # Define feature groups
    num_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "zone_load_rolling"]
    cat_cols = ["hour_granular", "user_zone", "rider_zone", "order_dow", "same_zone"]
    num_mixture_cols = ["distance_km"]
    num_power_cols = ["avg_rider_rating_hist"]

    # Category mappings
    user_zones = sorted(df["user_zone"].astype(str).unique().tolist())
    rider_zones = sorted(df["rider_zone"].astype(str).unique().tolist())
    dow_categories = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    same_zone_vals = sorted(df["same_zone"].unique().tolist())
    hour_granular_vals = sorted(df["hour_granular"].unique().tolist())

    categories_list = [
        hour_granular_vals,
        user_zones,
        rider_zones,
        dow_categories,
        same_zone_vals,
    ]

    # Preprocessing pipeline
    preprocess = ColumnTransformer(
        [
            (
                "mix",
                QuantileTransformer(output_distribution="normal"),
                num_mixture_cols,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    categories=categories_list,
                ),
                cat_cols,
            ),
            ("pow", PowerTransformer(), num_power_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=10000, solver="saga", random_state=42, class_weight="balanced"
    )

    param_grid = [
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "model__tol": [1e-5, 1e-4, 1e-3],
        }
    ]

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    # Time series cross-validation
    tss = TimeSeriesSplit(n_splits=5, test_size=None, gap=0)

    grid = GridSearchCV(
        pipe, param_grid, cv=tss, scoring="roc_auc", return_train_score=True, n_jobs=-1
    )

    X = df[num_mixture_cols + num_power_cols + num_cols + cat_cols]
    y = df["is_delayed"]

    grid.fit(X, y)

    logger.info(f"Best ROC-AUC: {grid.best_score_:.4f}")
    logger.info(f"Best Params: {grid.best_params_}")

    # Get best model and feature names for SHAP
    best_pipe = grid.best_estimator_
    X_transformed = best_pipe.named_steps["prep"].transform(X)

    # Store objects in context
    context["task_instance"].xcom_push(
        key="grid_best_score", value=float(grid.best_score_)
    )
    context["task_instance"].xcom_push(
        key="feature_cols",
        value={
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "num_mixture_cols": num_mixture_cols,
            "num_power_cols": num_power_cols,
        },
    )

    # Save model and intermediate data
    model_file = str(MODELS_DIR / "best_pipe.joblib")
    joblib.dump(best_pipe, model_file)

    data_for_eval = {
        "X": X,
        "y": y,
        "best_pipe": best_pipe,
        "X_transformed": X_transformed,
        "feature_cols": {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "num_mixture_cols": num_mixture_cols,
            "num_power_cols": num_power_cols,
        },
    }

    eval_file = f"/tmp/eval_data_{context['execution_date']}.pkl"
    joblib.dump(data_for_eval, eval_file)
    context["task_instance"].xcom_push(key="eval_data_file", value=eval_file)

    logger.info(f"Model saved to {model_file}")

    return model_file


def evaluate_model(eval_data_file: str, **context):
    """Evaluate model: generate confusion matrix, compute metrics."""
    logger.info("Evaluating model...")

    eval_data = joblib.load(eval_data_file)
    best_pipe = eval_data["best_pipe"]
    X = eval_data["X"]
    y = eval_data["y"]

    # Get predictions and probabilities
    probs = best_pipe.predict_proba(X)[:, 1]

    # Compute PR-AUC
    pr_auc = average_precision_score(y, probs)

    # Find optimal threshold based on F1 score
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    )
    opt_idx = np.argmax(f1_scores)
    opt_threshold = thresholds[opt_idx]

    logger.info(f"Optimal threshold: {opt_threshold:.4f}")
    logger.info(f"F1 at optimal threshold: {f1_scores[opt_idx]:.4f}")

    # Generate predictions with optimal threshold
    preds = (probs >= opt_threshold).astype(int)

    # Compute confusion matrix and metrics
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "roc_auc": float(context["task_instance"].xcom_pull(key="grid_best_score")),
        "pr_auc": float(pr_auc),
        "optimal_threshold": float(opt_threshold),
        "f1_score": float(f1_scores[opt_idx]),
        "accuracy": float(accuracy_score(y, preds)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    logger.info("Metrics computed:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

    # Create enhanced confusion matrix visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Confusion Matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Delayed"])
    disp.plot(ax=ax1, cmap="Blues", values_format="d")
    ax1.set_title(
        f"Confusion Matrix\n(Threshold: {opt_threshold:.4f})",
        fontsize=12,
        fontweight="bold",
    )

    # Right: F1 Score vs Threshold
    f1_range = np.linspace(0, 1, 100)
    f1_scores_range = []

    for thresh in f1_range:
        preds_test = (probs >= thresh).astype(int)
        if len(np.unique(preds_test)) > 1:
            f1_scores_range.append(f1_score(y, preds_test))
        else:
            f1_scores_range.append(0)

    ax2.plot(f1_range, f1_scores_range, "b-", linewidth=2, label="F1 Score")
    ax2.axvline(
        opt_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {opt_threshold:.4f}",
    )
    ax2.scatter([opt_threshold], [f1_scores[opt_idx]], color="r", s=100, zorder=5)
    ax2.set_xlabel("Probability Threshold", fontsize=11)
    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_title("F1 Score vs Threshold", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save visualization
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    cm_file = REPORTS_DIR / f"confusion_matrix_{date_str}.png"
    plt.savefig(cm_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {cm_file}")

    # Save metrics to JSON
    metrics_file = REPORTS_DIR / f"metrics_summary_{date_str}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")

    # Store for SHAP analysis
    context["task_instance"].xcom_push(
        key="probs", value=probs.tolist()[:1000]
    )  # Sample
    context["task_instance"].xcom_push(key="metrics_file", value=str(metrics_file))
    context["task_instance"].xcom_push(key="cm_file", value=str(cm_file))

    return str(metrics_file)


def generate_shap_analysis(eval_data_file: str, **context):
    """Generate SHAP values for model interpretability."""
    logger.info("Generating SHAP analysis...")

    try:
        eval_data = joblib.load(eval_data_file)
        best_pipe = eval_data["best_pipe"]
        X_transformed = eval_data["X_transformed"]

        # Get best model for SHAP
        best_model = best_pipe.named_steps["model"]
        feature_names = best_pipe.named_steps["prep"].get_feature_names_out()

        # Generate SHAP values
        explainer = shap.Explainer(
            best_model, X_transformed, feature_names=feature_names
        )
        shap_values = explainer(X_transformed)

        # Save SHAP values
        date_str = context["execution_date"].strftime("%Y-%m-%d")
        shap_file = REPORTS_DIR / f"shap_values_{date_str}.joblib"
        joblib.dump(shap_values, shap_file)

        logger.info(f"SHAP values saved to {shap_file}")

        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_transformed, feature_names=feature_names, show=False
        )

        shap_plot_file = REPORTS_DIR / f"shap_summary_{date_str}.png"
        plt.savefig(shap_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP summary plot saved to {shap_plot_file}")

        return str(shap_file)

    except Exception as e:
        logger.warning(f"SHAP analysis failed (non-critical): {e}")
        return None


def log_summary(**context):
    """Log summary of the pipeline execution."""
    logger.info("=" * 60)
    logger.info("DELIVERY DELAY MODEL TRAINING PIPELINE COMPLETED")
    logger.info("=" * 60)

    metrics_file = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="metrics_file"
    )
    cm_file = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="cm_file"
    )

    if metrics_file:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        logger.info("Final Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    logger.info(f"Confusion matrix: {cm_file}")
    logger.info(f"Model artifacts: {MODELS_DIR}")
    logger.info(f"Reports: {REPORTS_DIR}")
    logger.info("=" * 60)


# Define DAG
default_args = {
    "owner": "data-science",
    "retries": 1,
    "retry_delay": {"minutes": 5},
}

dag = DAG(
    "delivery_delay_model_training",
    default_args=default_args,
    description="Daily training and evaluation of delivery delay prediction model",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 10),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "forecasting"],
)

# Task 1: Fetch data
task_fetch = PythonOperator(
    task_id="fetch_data",
    python_callable=fetch_data_from_db,
    dag=dag,
)

# Task 2: Preprocess data
task_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    op_kwargs={"data_file": "{{ ti.xcom_pull(task_ids='fetch_data') }}"},
    dag=dag,
)

# Task 3: Train model
task_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    op_kwargs={"processed_data_file": "{{ ti.xcom_pull(task_ids='preprocess_data') }}"},
    dag=dag,
)

# Task 4: Evaluate model
task_evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    op_kwargs={
        "eval_data_file": "{{ ti.xcom_pull(task_ids='train_model', key='eval_data_file') }}"
    },
    dag=dag,
)

# Task 5: SHAP analysis (optional, non-blocking)
task_shap = PythonOperator(
    task_id="shap_analysis",
    python_callable=generate_shap_analysis,
    op_kwargs={
        "eval_data_file": "{{ ti.xcom_pull(task_ids='train_model', key='eval_data_file') }}"
    },
    trigger_rule="none_failed",  # Doesn't fail pipeline if SHAP fails
    dag=dag,
)

# Task 6: Logging
task_log = PythonOperator(
    task_id="log_summary",
    python_callable=log_summary,
    dag=dag,
)

# Define task dependencies
task_fetch >> task_preprocess >> task_train >> task_evaluate >> task_log
task_evaluate >> task_shap >> task_log
