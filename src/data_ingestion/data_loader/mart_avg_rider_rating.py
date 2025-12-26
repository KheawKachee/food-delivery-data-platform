import pandas as pd
import datetime as dt
import os
import traceback
import sys
import glob
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import logging

log = logging.getLogger(__name__)


def ETL_avg_rider_rating():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        stg_orders_df = pd.read_sql_table("stg_orders", engine)
        stg_riders_df = pd.read_sql_table("stg_riders", engine)

        # Aggregate per rider
        agg_df = (
            stg_orders_df.groupby("rider_id")
            .agg(
                n_jobs=("order_id", "count"), avg_rider_rating=("rider_rating", "mean")
            )
            .reset_index()
        )

        df = stg_riders_df.merge(agg_df, on="rider_id", how="left")
        df.rename(columns={"zone": "rider_zone"}, inplace=True)
        df["n_jobs"] = df["n_jobs"].fillna(0).astype(int)
        df["avg_rider_rating"] = df["avg_rider_rating"].fillna(0)

        stmt = text(
            """
        INSERT INTO avg_rider_rating (rider_id, rider_zone, n_jobs, avg_rider_rating)
        VALUES (:rider_id, :rider_zone, :n_jobs, :avg_rider_rating)
        ON CONFLICT (rider_id) DO UPDATE SET
            rider_zone = EXCLUDED.rider_zone,
            n_jobs = EXCLUDED.n_jobs,
            avg_rider_rating = EXCLUDED.avg_rider_rating
        """
        )

        with engine.begin() as conn:
            conn.execute(stmt, df.to_dict(orient="records"))
            print("ETL successfully completed")

    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ETL_avg_rider_rating()
