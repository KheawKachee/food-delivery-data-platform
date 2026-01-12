from pyspark.sql import SparkSession

import os

from dotenv import load_dotenv

load_dotenv()
POSTGRES_URL = "jdbc:postgresql://localhost:5432/food"
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")


def get_spark(app_name: str = "raw_ingestion"):
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
