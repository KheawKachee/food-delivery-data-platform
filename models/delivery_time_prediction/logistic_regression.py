import pandas as pd
import os
from sqlalchemy import create_engine

# Transformers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

query = """
select
    order_ts,
    delivery_time,
    is_delayed,
    distance_km,
    order_hour,
    order_dow,
    avg_rider_rating_hist,
    user_zone,
    rider_zone
from public.fct_delivery_time
where avg_rider_rating_hist is not null
"""

df = pd.read_sql(query, engine)

df.sort_values("order_ts", inplace=True)


tss = TimeSeriesSplit()

model = LogisticRegression()
scores = []


num_cols = ["distance_km", "order_hour", "order_dow", "avg_rider_rating_hist"]
cat_cols = ["user_zone", "rider_zone"]
target_col = "is_delayed"

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
    ]
)

model = LogisticRegression(max_iter=1000)

pipe = Pipeline([("prep", preprocess), ("model", model)])

cv_scores = cross_val_score(pipe, df[num_cols + cat_cols], df["is_delayed"], cv=tss)

print(f"Mean Accuracy: {cv_scores.mean():.4f}")
