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
        stg_users_df = pd.read_sql_table("stg_users", engine)
        stg_riders_df = pd.read_sql_table("stg_riders", engine)
        df = pd.DataFrame(
            columns=[
                "order_id" "order_ts",
                "delivery_time",
                "distance_km",
                "user_zone",
                "rider_zone",
                "avg_rider_rating",
            ]
        )
        df["order_id"] = stg_orders_df["order_id"]

        df["order_ts"] = stg_orders_df["order_ts"]

        df["delivery_time"] = stg_orders_df["deliveried_ts"].apply(
            pd.to_datetime
        ) - stg_orders_df["food_ready_ts"].apply(pd.to_datetime)
        df["distance_km"] = stg_orders_df["distance_km"]
        df["user_zone"] = stg_users_df["zone"]
        df["rider_zone"] = stg_riders_df["zone"]
        df["avg_rider_rating"] = stg_orders_df.groupby("rider_id")[
            "rider_rating"
        ].transform("mean")

        stmt = text(
            f"""
        INSERT INTO delivery_time (order_id, order_ts, delivery_time, distance_km, user_zone, rider_zone, avg_rider_rating)
        VALUES (:order_id, :order_ts, :delivery_time, :distance_km, :user_zone, :rider_zone, :avg_rider_rating)
        ON CONFLICT (order_id) DO UPDATE SET
            order_ts = EXCLUDED.order_ts,
            delivery_time = EXCLUDED.delivery_time,
            distance_km = EXCLUDED.distance_km, 
            user_zone = EXCLUDED.user_zone,  
            rider_zone = EXCLUDED.rider_zone,
            avg_rider_rating = EXCLUDED.avg_rider_rating
        """
        )

        with engine.begin() as conn:
            conn.execute(stmt, df.to_dict(orient="records"))
            print("query successfully")

    except Exception as e:
        print(
            "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )
        sys.exit(1)


if __name__ == "__main__":
    ETL_avg_rider_rating()
