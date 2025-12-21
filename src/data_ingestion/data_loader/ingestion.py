import pandas as pd
import datetime as dt
import sys
import traceback
import os
import glob
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


# CONFIG
def ingest():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DATABASE_URL"))

        rows = []

        for filepath in glob.glob("data/orders_*.json"):
            filename = os.path.basename(filepath)

            with open(filepath, "r") as f:

                file_content = json.load(f)
                records = (
                    file_content if isinstance(file_content, list) else [file_content]
                )

                for record in records:
                    rows.append(
                        {
                            "order_id": record["order_id"],
                            "payload": json.dumps(record),  # The specific sub-payload
                            "ingest_ts": dt.datetime.now(),
                        }
                    )

        payloads_df = pd.DataFrame(rows)

        log.info(f"payloads_df head:\n{payloads_df.head().to_string()}")

        stmt = text(
            """
        INSERT INTO raw_orders (order_id, payload, ingest_ts)
        VALUES (:order_id, :payload, :ingest_ts)
        ON CONFLICT (order_id) DO NOTHING;
        """
        )

        with engine.begin() as conn:
            conn.execute(stmt, payloads_df.to_dict(orient="records"))
            log.info("query successfully")

    except Exception as e:
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ingest()
