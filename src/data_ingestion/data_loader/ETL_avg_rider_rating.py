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
        stg_riders_df = pd.read_sql_table("stg_riders", engine)
        df = pd.DataFrame(
            columns=["rider_id", "rider_zone", "avg_rider_rating", "n_jobs"]
        )

        df["rider_id"] = stg_riders_df["rider_id"]
        df["rider_zone"] = stg_riders_df["zone"]
        df["n_jobs"] = stg_orders_df.groupby("rider_id")["order_id"].transform("count")
        df["avg_rider_rating"] = stg_orders_df.groupby("rider_id")[
            "rider_rating"
        ].transform("mean")

        stmt = text(
            f"""
        INSERT INTO avg_rider_rating (rider_id, rider_zone, n_jobs, avg_rider_rating)
        VALUES (:rider_id, :rider_zone, :n_jobs, :avg_rider_rating )
        ON CONFLICT (rider_id) DO UPDATE SET
            rider_id = EXCLUDED.rider_id,
            rider_zone = EXCLUDED.rider_zone,
            n_jobs = EXCLUDED.n_jobs,
            avg_rider_rating = EXCLUDED.avg_rider_rating
        """
        )

        with engine.begin() as conn:
            conn.execute(stmt, df.to_dict(orient="records"))
            log.info("query successfully")

    except Exception as e:
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ETL_avg_rider_rating()
