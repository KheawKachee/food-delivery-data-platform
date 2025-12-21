import pandas as pd
import datetime as dt
import os
import traceback
import sys
import glob
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from airflow.utils.log.logging_mixin import LoggingMixin

log = LoggingMixin().log


def ETL_delivery_time():
    try:
        pass
    except Exception as e:
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)


if __name__ == "__main__":
    ETL_delivery_time()
