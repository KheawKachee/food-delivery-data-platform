from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime

# Setup paths once
PROJ_PATH = "/home/kheaw/proj/food-delivery-data-platform"
PYTHON_EXEC = f"{PROJ_PATH}/venv/bin/python"

default_args = {
    "cwd": PROJ_PATH,  # use this as current dir
    "env": {"PATH": f"{PROJ_PATH}/venv/bin:$PATH"},  # use venv
}

with DAG(
    "order_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
) as dag:

    generate = BashOperator(
        task_id="generate",
        bash_command="python src/data_ingestion/data_generator.py {{ ds }}",
    )

    load_raw = BashOperator(
        task_id="raw_json_loader",
        bash_command=f"python src/data_ingestion/data_loader/ingestion.py",
    )

    etl = BashOperator(
        task_id="etl_to_staging",
        bash_command=f"python src/data_ingestion/data_loader/ETL_orders.py",
    )

    query_stg_order = SQLExecuteQueryOperator(
        task_id="query_stg_order",
        sql="SELECT * FROM stg_orders WHERE order_id <= 5;",
        conn_id="postgres",  # must match a defined connection
    )


generate >> load_raw >> etl >> query_stg_order
