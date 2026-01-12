from pyspark.sql import functions as F
from data_platform.ingestion.spark_session import get_spark

from pathlib import Path

spark = get_spark("raw_ingestion")

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = str(BASE_DIR / "data" / "orders_*.json")

df = spark.read.json("data/riders.json")

raw_riders = (
    df.withColumn("rider_id", F.col("rider_id"))
    .withColumn("payload", F.to_json(F.struct("*")))
    .withColumn("ingest_ts", F.current_timestamp())
)

raw_riders.write.mode("overwrite").format("delta").save("data_lake/raw/riders")
