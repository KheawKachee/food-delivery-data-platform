# Food Delivery Data Platform

## Overview

This project is an **end-to-end data engineering + data science + machine learning system** inspired by real food delivery platforms (e.g. LINE MAN). The goal is to simulate how a real company collects data, processes it, analyzes it, and uses machine learning models to support business decisions.

The project is intentionally **simple but realistic**, focusing on correctness, structure, and production mindset rather than complex algorithms.

---

## Business Problems

1. **Unstable delivery times** → affects customer satisfaction
2. **User churn** → reduces long-term revenue
3. **Lack of reliable data pipelines** → slows down decision-making

This system is designed to:

* Build reliable data pipelines
* Produce clean, trustworthy analytics
* Deploy machine learning models that can be used in production

---

## System Architecture (High Level)

```
Data Generator
   ↓
Raw Events (JSON)
   ↓
ETL Pipelines (Airflow)
   ↓
Data Warehouse (PostgreSQL)
   ↓
Analytics Tables (SQL)
   ↓
ML Training & Inference
   ↓
FastAPI Service + Monitoring
```

---

## Data Sources (Simulated)

The system generates realistic delivery-platform data:

### 1. Orders

* order_id
* user_id
* restaurant_id
* rider_id
* order_time
* delivery_time
* distance
* price

### 2. Users

* user_id
* signup_date
* total_orders
* last_order_date

### 3. Riders

* rider_id
* location
* availability_status
* completed_orders

All data is generated using Python to simulate real production events.

---

## Data Engineering

### Tech Stack

* Python
* SQL
* PostgreSQL
* Apache Airflow
* Docker

### Pipelines

* **Ingestion**: Load raw JSON events into raw tables
* **Cleaning**: Handle nulls, duplicates, invalid values
* **Aggregation**: Create analytics-ready tables

### Data Quality

* Null checks
* Duplicate checks
* Basic schema validation

---

## Analytics & Metrics

Key business metrics:

* Average delivery time
* Daily orders & GMV
* Rider utilization
* User churn rate

Metrics are calculated using SQL and stored as aggregated tables for dashboards or reporting.

---

## Machine Learning

### Model 1: Delivery Time Prediction

* Goal: Predict delivery time before order confirmation
* Features:

  * Distance
  * Time of day
  * Rider workload
  * Restaurant load

### Model 2 (Optional): User Churn Prediction

* Goal: Identify users likely to stop ordering
* Features:

  * Order frequency
  * Time since last order
  * Average spend

Models are trained using clean warehouse data with time-based validation.

---

## Deployment & MLOps

### Model Serving

* FastAPI REST endpoint
* Dockerized service

### Monitoring

* Data drift checks
* Model performance tracking
* Simple alerting logic

This setup reflects real-world ML production workflows.

---

## Repository Structure

```
/data_ingestion     # Data generators & loaders
/airflow            # DAGs for ETL pipelines
/warehouse          # SQL schemas & transformations
/analytics          # Business metrics & queries
/ml                 # Model training & evaluation
/api                # FastAPI service
/monitoring         # Data & model monitoring
README.md
```

---

## Engineering Principles

* Simple > complex
* Reproducible pipelines
* Clear data contracts
* Business-driven ML
* Production-first mindset

---

## Future Improvements

* Real-time streaming (Kafka)
* A/B testing for models
* Cloud deployment (GCP / AWS)
* Automated retraining

---

## Why This Project

This project is designed to demonstrate readiness for:

* Data Engineer Intern
* Data Scientist Intern
* Machine Learning Engineer Intern

It reflects real problems, real trade-offs, and real workflows used in modern data teams.
