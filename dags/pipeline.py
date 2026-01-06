from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime

# Setup paths once
PROJ_PATH = "/home/kheaw/projects/food-delivery-data-platform"
PYTHON_EXEC = f"{PROJ_PATH}/venv/bin/python"

default_args = {
    "cwd": PROJ_PATH,  # use this as current dir
    "env": {"PATH": f"{PROJ_PATH}/venv/bin:$PATH"},  # use venv
}

with DAG(
    "order_pipeline",
    start_date=datetime(2024, 12, 26),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    max_active_tasks=1,
) as dag:

    generate = BashOperator(
        task_id="generate",
        bash_command="python src/data_ingestion/data_generator.py {{ ds }}",
    )

    load_raw_orders = BashOperator(
        task_id="raw_orders_json_loader",
        bash_command="python src/data_ingestion/data_loader/ingest_orders.py",
    )

    load_raw_riders = BashOperator(
        task_id="raw_riders_json_loader",
        bash_command="python src/data_ingestion/data_loader/ingest_riders.py",
    )

    load_raw_users = BashOperator(
        task_id="raw_users_json_loader",
        bash_command="python src/data_ingestion/data_loader/ingest_users.py",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd dbt && dbt run",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd dbt && dbt test",
    )

    generate >> [load_raw_orders,load_raw_riders,load_raw_users] >> dbt_run >> dbt_test
