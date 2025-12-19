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

raw_payload_df = pd.json_normalize(raw_order_df["payload"])

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

print(stg_orders_df.duplicated())
