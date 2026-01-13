import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "users.json"

# ---------- read ----------
df = pd.read_json(DATA_PATH)

# ---------- validate ----------
if "user_id" not in df.columns:
    raise ValueError("Missing user_id")

# ---------- transform ----------
ingest_ts = datetime.utcnow()
df["payload"] = df.apply(lambda r: json.dumps(r.to_dict()), axis=1)
df["ingest_ts"] = ingest_ts

raw_users = df[["user_id", "payload", "ingest_ts"]]

# ---------- write ----------
engine = create_engine(DATABASE_URL)

raw_users.to_sql(
    "raw_users",
    engine,
    schema="raw",
    if_exists="replace",
    index=False,
)

print(f">>> Ingested {len(raw_users)} users at {ingest_ts}")
