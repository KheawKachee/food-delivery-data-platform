import pandas as pd
import datetime as dt
import os
import glob
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# CONFIG
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

raw_order_df = pd.read_sql_table("raw_orders", engine)

raw_payload_df = pd.json_normalize(
    raw_order_df["payload"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
)


stg_orders_df = raw_payload_df[
    [
        "order_id",
        "user_id",
        "rider_id",
        "order_ts",
        "food_ready_ts",
        "distance_km",
        "deliveried_ts",
        "price_baht",
        "rider_rating",
    ]
]

if stg_orders_df.duplicated(subset=["order_id"]).sum() == 0:
    print("no duplication")
else:
    stg_orders_df = stg_orders_df.drop_duplicates(subset=["order_id"], keep="first")

stmt = text(
    """
INSERT INTO stg_orders (order_id, user_id, rider_id, order_ts, food_ready_ts, distance_km, deliveried_ts, price_baht, rider_rating )
VALUES (:order_id, :user_id, :rider_id, :order_ts, :food_ready_ts, :distance_km, :deliveried_ts, :price_baht, :rider_rating )
ON CONFLICT (order_id) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    rider_id = EXCLUDED.rider_id,
    order_ts = EXCLUDED.order_ts,
    food_ready_ts = EXCLUDED.food_ready_ts,
    distance_km = EXCLUDED.distance_km,
    deliveried_ts = EXCLUDED.deliveried_ts,
    price_baht = EXCLUDED.price_baht,
    rider_rating = EXCLUDED.rider_rating;
"""
)

print(stg_orders_df[stg_orders_df["rider_rating"].isna()])
with engine.begin() as conn:
    conn.execute(stmt, stg_orders_df.to_dict(orient="records"))
