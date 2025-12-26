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
    max_active_runs=1,  # parallelism control
    max_active_tasks=1,  # parallelism control
) as dag:

    generate = BashOperator(
        task_id="generate",
        bash_command="python src/data_ingestion/data_generator.py {{ ds }}",
    )

    load_raw = BashOperator(
        task_id="raw_json_loader",
        bash_command=f"python src/data_ingestion/data_loader/ingestion.py",
    )

    stg_order = BashOperator(
        task_id="to_stg_order",
        bash_command=f"python src/data_ingestion/data_loader/stg_orders.py",
    )

    mart_avg_rider_rating = BashOperator(
        task_id="mart_avg_rider_rating",
        bash_command=f"python src/data_ingestion/data_loader/mart_avg_rider_rating.py",
    )
    mart_delivery_time = BashOperator(
        task_id="mart_delivery_time",
        bash_command=f"python src/data_ingestion/data_loader/mart_delivery_time.py",
    )
    mart_hourly_total_spends = BashOperator(
        task_id="mart_hourly_total_spends",
        bash_command=f"python src/data_ingestion/data_loader/mart_hourly_total_spends.py",
    )


(
    generate
    >> load_raw
    >> stg_order
    >> [mart_avg_rider_rating, mart_delivery_time, mart_hourly_total_spends]
)
