import pandas as pd
from sqlalchemy import create_engine

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

engine = create_engine("postgresql://user:pw@host:5432/db")

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
from analytics.fct_delivery_time
where avg_rider_rating_hist is not null
"""

df = pd.read_sql(query, engine)

df.sort_values("order_ts", inplace=True)

kf = KFold(n_splits=4, shuffle=True, random_state=42)

model = LogisticRegression()
scores = []


num_cols = ["distance_km", "order_hour", "order_dow", "avg_rider_rating_hist"]
cat_cols = ["user_zone", "rider_zone"]


preprocess = ColumnTransformer(
    [
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

model = LogisticRegression(max_iter=1000)

pipe = Pipeline([("prep", preprocess), ("clf", model)])

pipe.fit(train[num_cols + cat_cols], train["is_delayed"])
