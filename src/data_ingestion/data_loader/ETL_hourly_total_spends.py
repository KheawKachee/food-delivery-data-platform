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


def ETL_hourly_total_spends():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        stg_orders_df = pd.read_sql_table("stg_orders", engine)
        df = pd.DataFrame(columns=["time", "n_orders", "total_price_baht"])

        df["hourly"] = stg_orders_df["order_ts"].dt.floor("H")
        df["total_price_baht"] = stg_orders_df.groupby("hour")["total_price_baht"].agg
        df["n_jobs"] = stg_orders_df.goroupby("hour")["order_id"].agg("count")

        stmt = text(
            f"""
        INSERT INTO avg_rider_rating (hourly, total_price_baht, n_jobs)
        VALUES (:hourly, :total_price_baht, :n_jobs)
        ON CONFLICT (hourly) DO UPDATE SET
            total_price_baht = EXCLUDED.total_price_baht,
            n_jobs = EXCLUDED.n_jobs,
        """
        )

        with engine.begin() as conn:
            conn.execute(stmt, df.to_dict(orient="records"))
            log.info("query successfully")
    except Exception as e:
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ETL_hourly_total_spends()
