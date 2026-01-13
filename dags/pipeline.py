from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime

from dotenv import load_dotenv


import os

load_dotenv()

PROJ_PATH = "/home/kheaw/projects/food-delivery-data-platform"
DBT_BIN = "$PROJ_PATH/venv/bin/dbt"
DBT_PROJECT_DIR = "$PROJ_PATH/data_platform/dbt"

DATABASE_URL = os.getenv("DATABASE_URL")

default_args = {
    "cwd": PROJ_PATH,
}

with DAG(
    "order_pipeline",
    start_date=datetime(2024, 12, 26),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
) as dag:

    create_raw_tables = SQLExecuteQueryOperator(
        task_id="create_raw_tables",
        conn_id="postgres_food",
        sql="""
        create schema if not exists raw;

        create table if not exists raw.raw_orders (
            order_id   bigint,
            payload    jsonb,
            ingest_ts  timestamp
        );

        create table if not exists raw.raw_users (
            user_id    bigint,
            payload    jsonb,
            ingest_ts  timestamp
        );

        create table if not exists raw.raw_riders (
            rider_id   bigint,
            payload    jsonb,
            ingest_ts  timestamp
        );
        """,
    )

    generate = BashOperator(
        task_id="generate_data",
        bash_command="python src/data_generator.py {{ ds }}",
    )

    ingest_orders = BashOperator(
        task_id="ingest_orders",
        bash_command="""
    export PYTHONPATH=/home/kheaw/projects/food-delivery-data-platform
    python data_platform/ingestion/ingest_orders.py
    """,
    )

    ingest_riders = BashOperator(
        task_id="ingest_riders",
        bash_command="""
        export PYTHONPATH=/home/kheaw/projects/food-delivery-data-platform
        python data_platform/ingestion/ingest_riders.py
        """,
    )

    ingest_users = BashOperator(
        task_id="ingest_users",
        bash_command="""
        export PYTHONPATH=/home/kheaw/projects/food-delivery-data-platform
        python data_platform/ingestion/ingest_users.py
        """,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=(
            f"{DBT_BIN} run "
            f"--project-dir {DBT_PROJECT_DIR} "
            f"--profiles-dir {DBT_PROJECT_DIR}"
        ),
        env={"PROJ_PATH": "/home/kheaw/projects/food-delivery-data-platform"},
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=(
            f"{DBT_BIN} test "
            f"--project-dir {DBT_PROJECT_DIR} "
            f"--profiles-dir {DBT_PROJECT_DIR}"
        ),
        env={"PROJ_PATH": "/home/kheaw/projects/food-delivery-data-platform"},
    )

    (
        create_raw_tables
        >> generate
        >> [ingest_orders, ingest_riders, ingest_users]
        >> dbt_run
        >> dbt_test
    )
