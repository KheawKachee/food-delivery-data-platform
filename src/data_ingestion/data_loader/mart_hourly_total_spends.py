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

        stg_orders_df["hourly"] = stg_orders_df["order_ts"].dt.floor("H")

        df = (
            stg_orders_df.groupby("hourly", as_index=False)
            .agg(
                total_price_baht=("price_baht", "sum"),
                n_orders=("order_id", "count"),
            )
            .sort_values("hourly")
        )

        stmt = text(
            f"""
        INSERT INTO hourly_total_spends (hourly, n_orders,total_price_baht)
        VALUES (:hourly, :n_orders, :total_price_baht)
        ON CONFLICT (hourly) DO UPDATE SET
            n_orders = EXCLUDED.n_orders,
            total_price_baht = EXCLUDED.total_price_baht
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
