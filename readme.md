# Food Delivery Data Platform
Data-Driven ML System for Delivery Time & Performance Prediction
## Overview

This project demonstrates an **end-to-end data science + machine learning system** that tackles real business problems in food delivery platforms. It combines data engineering pipelines, statistical analysis, and ML models to predict delivery times and optimize operations.

The focus is on **data-driven decision making**: clean data → exploratory analysis → feature engineering → model training → production serving.

---

## Business Problems (ML Perspective)

**Delayed Deliveries** → ML Solution: Build classification models to predict delays before order confirmation
   - Impacts customer satisfaction & user retention
   - Enable dynamic pricing & realistic ETAs
   - Prevents high network traffic during peak times

This system is designed to:

* Collect, clean, and engineer features from raw data
* Train, validate, and evaluate ML models rigorously
* Deploy models as production services with monitoring
* Measure business impact through KPIs

---

## System Architecture

```
Data Generation (Simulated Events)
   ↓
Raw Data (JSON/CSV)
   ↓
Data Ingestion (Python + PostgreSQL)
   ↓
Feature Engineering (dbt transformations + custom SQL)
   ↓
Feature Store (Staging tables)
   ↓
ML Training (Scikit-learn)
   ↓
Model Evaluation & Validation (Train/Val/Test splits)
   ↓
Model Monitoring & Performance Tracking
```

**Key Assumption**: This is a batch learning system with scheduled model retraining (via Airflow).

---

## Data Sources (Simulated)

The system generates realistic delivery-platform data:

### Simulated Data Tables

The platform generates the following core tables based on the data generator:

#### 1. Orders

- `order_id`: Unique order identifier
- `user_id`: User placing the order
- `rider_id`: Assigned delivery rider
- `order_ts`: Order placement timestamp
- `food_ready_ts`: Time when food is ready for pickup
- `distance_km`: Distance between user and restaurant (km)
- `delivered_ts`: Delivery completion timestamp
- `price_baht`: Total order price (in Baht)
- `rider_rating`: Customer rating for the delivery

#### 2. Users

- `user_id`: Unique user identifier
- `signup_date`: Registration date
- `zone`: User's delivery zone

#### 3. Riders

- `rider_id`: Unique rider identifier
- `signup_date`: Registration date
- `zone`: Rider's operating zone

All data is generated using Python scripts in `/src/data_generator.py` to simulate realistic food delivery platform events.

---

## Data Engineering (Supporting DS)

### Tech Stack

* Python (data processing + ML training)
* PostgreSQL (feature store / data warehouse)
* Apache Airflow (orchestration of ingestion + model retraining)
* dbt (feature engineering transformations)
* Docker (reproducible environments)

### Data Pipeline

1. **Ingestion**: Raw JSON → PostgreSQL raw tables
2. **Staging**: Clean + standardize raw data
3. **Feature Engineering**: Create ML-ready tables (distances, time features, rider metrics, etc.)
4. **Data Mart**: Aggregate tables for analytics & model serving

### Data Quality Checks

* Schema validation on ingestion
* Null / outlier detection
* Duplicate handling
* Data drift monitoring during model training

---
## Analytics & Feature Analysis

### Key Features & Metrics for ML and Business Impact

**Order-Level Features (Model Inputs):**
- `distance_km`: Delivery distance, a core driver of delivery time.
- `order_hour`, `order_dow`: Time of order (hour/day); peak hours show higher delays.
- `price_baht`: Order value; premium orders may require priority handling.
- `user_zone`, `rider_zone`, `same_zone`: Delivery and rider zones; cross-zone deliveries may increase delays.
- `avg_rider_rating_hist`: Historical average rider rating; higher ratings often correlate with fewer delays.
- `zone_load_rolling`: Rolling average of recent delays in a zone, capturing congestion effects.
- `hour_granular`: Engineered time bins to capture peak/off-peak patterns.

**Rider-Level Features (Assignment & Optimization):**
- Completed deliveries and workload: High workload may increase error rates.
- Average delivery time: Predicts future performance.
- Recent availability: On-shift status impacts assignment.
- Distance from pickup: Affects ETA accuracy.

**Business Metrics (KPI Tracking):**
- Average delivery time (by hour, restaurant, rider): Core SLA metric.
- Delivery time variance: Lower variance improves customer trust.
- On-time delivery rate: Key for user satisfaction and retention.
- Rider utilization rate: Orders per hour, indicating efficiency.
- User return rate: Inverse of churn, linked to long-term revenue.
- Estimated cost per delivery: Tracks profitability by segment.

**How ML Insights Drive Business Actions:**

| ML Insight                        | Business Action                        | Expected Outcome                |
|------------------------------------|----------------------------------------|---------------------------------|
| High delay probability predicted   | Flag order, offer incentive/reassign   | ↑ On-time delivery, ↑ retention |
| High-risk restaurant identified    | Pre-stage riders, adjust pricing       | ↑ SLA compliance                |
| Rider performance decline detected | Pause assignments, offer training      | ↓ Quality issues, ↑ ratings     |

These features and metrics are engineered in dbt and Python pipelines, used for:
1. **Feature engineering**: Transforming raw data into predictive signals for ML models.
2. **Model evaluation**: Benchmarking model performance against business baselines.
3. **Production monitoring**: Tracking drift and operational changes over time.


---

## Machine Learning Models

### Delivery Delay Classification (Primary)

**Goal:** Predict if an order will be delayed (binary) before confirmation, enabling dynamic ETAs and proactive interventions.
- Inaccurate ETAs frustrate customers & drive churn
- **Model:** Logistic Regression (with feature engineering, cyclical time encoding, and hyperparameter tuning)gs
- **Features:** Distance, order hour/day (with cyclical encoding), user/rider zones, rider rating, rolling zone delay rate, engineered time bins- Overestimated times = lose customers to competitors
- **Target:** `is_delayed` (1 = delayed, 0 = on-time) Achieve ±5 min accuracy on 80% of orders → 10% improvement in on-time delivery
- **Validation:** TimeSeriesSplit (avoids leakage), ROC-AUC, PR-AUC, F1, SHAP for explainability
- **Artifacts:** `models/delivery_time_classification/best_pipe.joblib`, metrics & SHAP in `/reports/`
- **Experimentation:** See `models/delivery_time_classification/delivery_time_classification_experiment.ipynb`

### Model Development Workflow: Captures interaction effects (e.g., busy rider + rush hour), better accuracy
d interpretability with SHAP, slightly lower accuracy than XGBoost
1. **EDA & Feature Engineering:** Analyze distributions, correlations, create time/zone features, handle missing/outliers.nt Boosting**: Similar to XGBoost, often comparable performance
2. **Model Training:** Logistic regression with time-based validation, grid search for tuning.
3. **Evaluation:** ROC-AUC, F1, confusion matrix, SHAP analysis for interpretability.
4. **Deployment:** Model serialized as `best_pipe.joblib`, served via FastAPI (`/predict`), monitored for drift.* **RMSE / MAE**: Target < 5 minutes (achievable with good features)
5. **Retraining:** Scheduled via Airflow; triggers on drift or performance drop. 0.75 (explains 75% of variance)
10 min (worst case acceptable)
---n > 80% (threshold for customer satisfaction)
estimation for any rider or restaurant
### MLOps & Monitoring
(Critical for Time-Series):
- **Serving:** FastAPI + Docker, input: order features, output: delay probability or ETA.5
- **Monitoring:** Track prediction vs. actual, feature drift, rolling metrics, alert on degradation.edictions
- **Versioning:** Models and metrics tracked in `/models/` and `/reports/`, code in git.
* **Cross-validation**: K-fold on historical periods, NOT random shuffle
* **Holdout test period**: Final Jan 14-15 data completely unseen during training

**See**:  
- Notebook: `models/delivery_time_classification/delivery_time_classification_experiment.ipynb`  * Location: `models/delivery_time_classification/delivery_time_classification_experiment.ipynb`
- DAG: `dags/delivery_delay_model_pipeline.py`  
- Reports: `models/delivery_time_classification/reports_*`* Reports: Stored in `reports_2026-01-15_20/` 
* Store predictions + actuals for comparison
* Compute rolling metrics (daily/weekly RMSE)
* Log model version & deployment timestamp

### Model Versioning

* Keep serialized models in `models/` directory
* Track metrics in experiment reports
* Version control via git + tags for deployments

---

## Repository Structure

```
/data                          # Raw simulated data (JSON)
/data_platform
  /dbt                         # Feature engineering transformations
    /models
      /staging                 # Raw → Clean tables
      /mart                    # Features for ML + Analytics
  /ingestion                   # Data loading scripts
/models
  /delivery_time_classification
    /delivery_time_classification_experiment.ipynb  # Main EDA + training
    /models                    # Serialized models (best_pipe.joblib)
    /reports                   # Metrics & evaluation results
/src                           # Utility functions & data generators
/dags                          # Airflow DAGs (orchestration)
/bash                          # Setup & start scripts
docker-compose.yml             # Docker setup
requirements.txt               # Python dependencies
README.md                       # This file
```

### Key DS Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| EDA + Training | Model experimentation | Jupyter, Pandas, Scikit-learn |
| Feature Engineering | Transform raw → features | dbt, SQL, Python |
| Model Storage | Serialized models | Joblib |
| Orchestration | Schedule retraining | Airflow |
| Deployment | Serve predictions | FastAPI, Docker |

---

## Engineering & DS Principles

* **Data-Driven**: All decisions backed by metrics and validation
* **Reproducibility**: Versioned code, serialized models, documented experiments
* **Time-Series Awareness**: Respect temporal order to avoid data leakage
* **Simplicity First**: Clear code & explainable models over black boxes
* **Business-Focused**: ML metrics tied to business KPIs
* **Production-Ready**: From day 1 — monitoring, versioning, error handling

## Getting Started

### Prerequisites
```
Python 3.9+
PostgreSQL
Docker & Docker Compose
```

### Quick Start
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start data platform
docker-compose up -d

# 3. Run data ingestion
python data_platform/ingestion/ingest_*.py

# 4. Run dbt transformations
cd data_platform/dbt && dbt run

# 5. Open training notebook
jupyter notebook models/delivery_time_classification/delivery_time_classification_experiment.ipynb
```

## Future Improvements

* Real-time predictions (streaming features)
* Ensemble methods combining multiple models
* Causal inference for rider assignment optimization
* A/B testing framework for model comparisons
* Automated hyperparameter tuning (Optuna/Hyperopt)
* Production dashboard for model monitoring
* Cloud deployment (GCP / AWS)

---