import pandas as pd
import datetime as dt
import sys
import traceback
import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def ingest_riders():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        with open("data/riders.json", "r") as f:
            file_content = json.load(f)
            records = file_content if isinstance(file_content, list) else [file_content]

        rows = [
            {
                "rider_id": r["rider_id"],
                "payload": json.dumps(r),
                "ingest_ts": dt.datetime.now(),
            }
            for r in records
        ]

        df = pd.DataFrame(rows)

        stmt = text(
            """
            INSERT INTO raw_riders (rider_id, payload, ingest_ts)
            VALUES (:rider_id, :payload, :ingest_ts)
            ON CONFLICT (rider_id) DO UPDATE SET
                payload = EXCLUDED.payload,
                ingest_ts = EXCLUDED.ingest_ts;
            """
        )

        with engine.begin() as conn:
            conn.execute(stmt, df.to_dict(orient="records"))

        print(">>> riders ingested successfully")

    except Exception as e:
        print(">>> ".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ingest_riders()
