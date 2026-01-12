from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime

from dotenv import load_dotenv


import os

load_dotenv()

PROJ_PATH = "/home/kheaw/projects/food-delivery-data-platform"

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

    spark_orders = BashOperator(
        task_id="spark_ingest_orders",
        bash_command="""
        export PYTHONPATH=/home/kheaw/projects/food-delivery-data-platform
        spark-submit \
    --master local[*] \
    --packages org.postgresql:postgresql:42.7.3 \
    data_platform/ingestion/ingest_orders.py
    """,
    )

    spark_riders = BashOperator(
        task_id="spark_ingest_riders",
        bash_command="""
        export PYTHONPATH=/home/kheaw/projects/food-delivery-data-platform
    spark-submit \
       --master local[*] \
    --packages org.postgresql:postgresql:42.7.3 \
      data_platform/ingestion/ingest_riders.py
    """,
    )

    spark_users = BashOperator(
        task_id="spark_ingest_users",
        bash_command="""
    export PYTHONPATH=/home/kheaw/projects/food-delivery-data-platform
    spark-submit \
       --master local[*] \
    --packages org.postgresql:postgresql:42.7.3 \
      data_platform/ingestion/ingest_users.py
    """,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd dbt && dbt run",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd dbt && dbt test",
    )

    (
        create_raw_tables
        >> generate
        >> [spark_orders, spark_riders, spark_users]
        >> dbt_run
        >> dbt_test
    )
