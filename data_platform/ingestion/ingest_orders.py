from pyspark.sql import functions as F
from spark_session import *

from pathlib import Path

spark = get_spark("raw_ingestion")

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = str(BASE_DIR / "data" / "orders_*.json")

df = spark.read.json(DATA_PATH)

raw_orders = (
    df.withColumn("order_id", F.col("order_id"))
    .withColumn("event_date", F.to_date("order_ts"))
    .withColumn("payload", F.to_json(F.struct("*")))
    .withColumn("ingest_ts", F.current_timestamp())
)

(
    raw_orders.select("order_id", "payload", "ingest_ts")
    .write.format("jdbc")
    .option("url", POSTGRES_URL)
    .option("dbtable", "raw.raw_orders")
    .option("user", POSTGRES_USER)
    .option("password", POSTGRES_PASSWORD)
    .option("driver", "org.postgresql.Driver")
    .mode("append")
    .save()
)
