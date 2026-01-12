#!/usr/bin/env bash
set -e

wget https://jdbc.postgresql.org/download/postgresql-42.7.3.jar
cp postgresql-42.7.3.jar $SPARK_HOME/jars/


# -------- CONFIG --------
export AIRFLOW_HOME="$HOME/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW_CONN_POSTGRES_FOOD='postgresql://postgres:postgres@localhost:5432/food'
PROJ_PATH="$HOME/projects/food-delivery-data-platform"

echo "Initializing Airflow at $AIRFLOW_HOME"

# Create folders
mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/projects"

# Symlink project source
ln -sfn "$PROJ_PATH/src" "$AIRFLOW_HOME/projects/food_delivery_src"

cp dags/pipeline.py $AIRFLOW_HOME/dags

echo "✅ One-time Airflow setup complete"
