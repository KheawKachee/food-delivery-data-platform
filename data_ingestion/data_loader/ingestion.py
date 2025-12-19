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

rows = []

for filepath in glob.glob("data/orders_*.json"):
    filename = os.path.basename(filepath)

    with open(filepath, "r") as f:

        file_content = json.load(f)
        records = file_content if isinstance(file_content, list) else [file_content]

        for record in records:
            rows.append(
                {
                    "order_id": record["order_id"],
                    "payload": json.dumps(record),  # The specific sub-payload
                    "ingest_ts": dt.datetime.now(),
                }
            )

payloads_df = pd.DataFrame(rows)


stmt = text(
    """
INSERT INTO raw_orders (order_id, payload, ingest_ts)
VALUES (:order_id, :payload, :ingest_ts)
ON CONFLICT (order_id) DO NOTHING;
"""
)

with engine.begin() as conn:
    conn.execute(stmt, payloads_df.to_dict(orient="records"))
