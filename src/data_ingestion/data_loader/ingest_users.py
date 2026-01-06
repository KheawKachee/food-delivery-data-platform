import pandas as pd
import datetime as dt
import sys
import traceback
import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def ingest_users():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        with open("data/users.json", "r") as f:
            file_content = json.load(f)
            records = file_content if isinstance(file_content, list) else [file_content]

        rows = [
            {
                "user_id": u["user_id"],
                "payload": json.dumps(u),
                "ingest_ts": dt.datetime.now(),
            }
            for u in records
        ]

        df = pd.DataFrame(rows)

        stmt = text(
            """
            INSERT INTO raw_users (user_id, payload, ingest_ts)
            VALUES (:user_id, :payload, :ingest_ts)
            ON CONFLICT (user_id) DO UPDATE SET
                payload = EXCLUDED.payload,
                ingest_ts = EXCLUDED.ingest_ts;
            """
        )

        with engine.begin() as conn:
            conn.execute(stmt, df.to_dict(orient="records"))

        print(">>> users ingested successfully")

    except Exception as e:
        print(">>> ".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ingest_users()
