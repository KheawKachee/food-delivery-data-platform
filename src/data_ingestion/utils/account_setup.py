import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import os


# CONFIG
def account_setup():
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    DATA_PATH = "data/"

    engine = create_engine(DATABASE_URL)

    rider_df = pd.read_json(os.path.join(DATA_PATH, "riders.json"))
    rider_records = rider_df.to_dict(orient="records")

    user_df = pd.read_json(os.path.join(DATA_PATH, "users.json"))
    user_records = user_df.to_dict(orient="records")

    # sql statement
    user_stmt = text(
        """ 
    INSERT INTO stg_users (user_id, signup_date, zone)
    VALUES (:user_id, :signup_date, :zone)
    ON CONFLICT (user_id)
    DO UPDATE SET
        signup_date = EXCLUDED.signup_date,
        zone = EXCLUDED.zone;
    """
    )

    rider_stmt = text(
        """ 
    INSERT INTO stg_riders (rider_id, signup_date, zone)
    VALUES (:rider_id, :signup_date, :zone)
    ON CONFLICT (rider_id)
    DO UPDATE SET
        signup_date = EXCLUDED.signup_date,
        zone = EXCLUDED.zone;
    """
    )

    # STAGE DATA
    with engine.begin() as conn:
        conn.execute(user_stmt, user_records)
        conn.execute(rider_stmt, rider_records)


if __name__ == "__main__":
    account_setup()
