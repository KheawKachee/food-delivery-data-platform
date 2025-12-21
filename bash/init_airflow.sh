#!/usr/bin/env bash
set -e

# -------- CONFIG --------
export AIRFLOW_HOME="$HOME/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
PROJ_PATH="$HOME/proj/food-delivery-data-platform"

echo "Initializing Airflow at $AIRFLOW_HOME"

# Create folders
mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/projects"

# Symlink project source
ln -sfn "$PROJ_PATH/src" "$AIRFLOW_HOME/projects/food_delivery_src"

cp dags/pipeline.py $AIRFLOW_HOME/dags

echo "✅ One-time Airflow setup complete"
