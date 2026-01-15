import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


DATABASE_URL = os.getenv("DATABASE_URL")
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data"

files = list(DATA_PATH.glob("orders_*.json"))
if not files:
    raise FileNotFoundError("No order files found")

df = pd.concat(
    [pd.read_json(p) for p in files],
    ignore_index=True,
)

df["order_ts"] = pd.to_datetime(df["order_ts"])
df = df.sort_values(by=["order_ts"], ascending=False).drop_duplicates(
    subset="order_id", keep="first"
)

REQUIRED_COLS = {"order_id", "order_ts"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.replace({np.nan: None})
df["payload"] = df.apply(lambda r: json.dumps(r.to_dict(), cls=JSONEncoder), axis=1)


ingest_ts = datetime.now(timezone.utc)
df["ingest_ts"] = ingest_ts

raw_orders = df[["order_id", "payload", "ingest_ts"]]


engine = create_engine(DATABASE_URL)

raw_orders.to_sql(
    "raw_orders",
    engine,
    schema="raw",
    if_exists="append",
    index=False,
)

print(f">>> Ingested {len(raw_orders)} rows at {ingest_ts}")
