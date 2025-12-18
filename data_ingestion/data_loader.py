import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# CONFIG
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
DATA_PATH = "data/"

engine = create_engine(DATABASE_URL)

# LOAD TABLE
users_df = pd.read_sql_table("users", engine)
riders_df = pd.read_sql_table("riders", engine)
orders_df = pd.read_sql_table("orders", engine)

# LOAD PAYLOAD
payload_df = pd.read_json("")

# STAGE DATA
users_df.to_sql("staging_users", engine, if_exists="append", index=False)
riders_df.to_sql("staging_riders", engine, if_exists="append", index=False)
orders_df.to_sql("staging_orders", engine, if_exists="append", index=False)


# MERGE (UPSERT)
with engine.begin() as conn:

    conn.execute(
        text(
            """
        INSERT INTO users (user_id, signup_date, zone)
        SELECT
            user_id,
            signup_date::timestamp,
            zone
        FROM staging_users
        ON CONFLICT (user_id) DO NOTHING;
    """
        )
    )

    conn.execute(
        text(
            """
        INSERT INTO riders (rider_id, signup_date, zone)
        SELECT DISTINCT rider_id, signup_date::timestamp, zone
        FROM staging_riders
        ON CONFLICT (rider_id) DO NOTHING;
    """
        )
    )

    conn.execute(
        text(
            """
        INSERT INTO orders (
            order_id, user_id, rider_id, order_time,
            prep_time_minutes, distance_km,
            delivery_time_minutes, price_baht, rider_rating
        )
        SELECT DISTINCT *
        FROM staging_orders
        ON CONFLICT (order_id) DO NOTHING;
    """
        )
    )

    conn.execute(text("TRUNCATE staging_users;"))
    conn.execute(text("TRUNCATE staging_riders;"))
    conn.execute(text("TRUNCATE staging_orders;"))

print("Load complete")
