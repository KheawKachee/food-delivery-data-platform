from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("raw_ingestion").getOrCreate()

df = spark.read.json("data/riders.json")

raw_riders = (
    df.withColumn("rider_id", F.col("rider_id"))
    .withColumn("payload", F.to_json(F.struct("*")))
    .withColumn("ingest_ts", F.current_timestamp())
)

raw_riders.write.mode("overwrite").format("delta").save("data_lake/raw/riders")
