import pandas as pd
import datetime as dt
import os
import traceback
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def ETL_avg_rider_rating():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        # Read staging tables
        stg_orders_df = pd.read_sql_table("stg_orders", engine)
        stg_users_df = pd.read_sql_table("stg_users", engine)
        stg_riders_df = pd.read_sql_table("stg_riders", engine)

        # Merge user and rider zones into orders
        df = stg_orders_df.copy()

        df = df.merge(
            stg_users_df[["user_id", "zone"]], how="left", on="user_id"
        ).rename(columns={"zone": "user_zone"})

        df = df.merge(
            stg_riders_df[["rider_id", "zone"]], how="left", on="rider_id"
        ).rename(columns={"zone": "rider_zone"})

        # Compute delivery time
        df["delivery_time"] = pd.to_datetime(df["delivered_ts"]) - pd.to_datetime(
            df["food_ready_ts"]
        )

        # Compute average rating per rider
        df["avg_rider_rating"] = df.groupby("rider_id")["rider_rating"].transform(
            "mean"
        )

        # Select only columns to insert
        insert_df = df[
            [
                "order_id",
                "order_ts",
                "delivery_time",
                "distance_km",
                "user_zone",
                "rider_zone",
                "avg_rider_rating",
            ]
        ]

        # Upsert into delivery_time table
        stmt = text(
            """
        INSERT INTO delivery_time 
            (order_id, order_ts, delivery_time, distance_km, user_zone, rider_zone, avg_rider_rating)
        VALUES 
            (:order_id, :order_ts, :delivery_time, :distance_km, :user_zone, :rider_zone, :avg_rider_rating)
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
            conn.execute(stmt, insert_df.to_dict(orient="records"))

        print("ETL completed successfully.")

    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ETL_avg_rider_rating()
