import json
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np  # Add this for NaN handling

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB

from dotenv import load_dotenv

load_dotenv()

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

REQUIRED_COLS = {"order_id", "order_ts"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")


df = df.replace({np.nan: None})
df["payload"] = df.apply(lambda r: json.dumps(r.to_dict()), axis=1)


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
    chunksize=1000,
    dtype={"payload": JSONB},  # Specify JSONB type for payload
)

print(f">>> Ingested {len(raw_orders)} rows at {ingest_ts}")
