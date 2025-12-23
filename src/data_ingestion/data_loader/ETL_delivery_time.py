import pandas as pd
import datetime as dt
import os
import traceback
import sys
import glob
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


def ETL_avg_rider_rating():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        stg_orders_df = pd.read_sql_table("stg_orders", engine)
        df = pd.DataFrame(
            columns=[
                "order_ts",
                "delivery_time",
                "distance_km",
                "user_zone",
                "rider_zone",
                "avg_rider_rating",
            ]
        )

        df["order_ts"] = stg_orders_df["order_ts"]

        time_interval = (  # delivery time interval
            stg_orders_df["delivered_ts"] - stg_orders_df["food_ready_ts"]
        ).dt.total_seconds() / 3600
        stg_orders_df["delivery_hours"] = time_interval.dt.total_seconds() / 3600
        stg_orders_df["delivery_time_str"] = time_interval.astype(str)
        pass

    except Exception as e:
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ETL_avg_rider_rating()
