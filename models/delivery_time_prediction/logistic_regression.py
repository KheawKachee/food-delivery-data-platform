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
from sklearn.model_selection import GridSearchCV

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
order by order_ts
"""

df = pd.read_sql(query, engine)

df.sort_values("order_ts", inplace=True)


print(f"na contain in cols :\n {df.isna().sum()}")

tss = TimeSeriesSplit()


num_cols = ["distance_km", "order_hour", "order_dow", "avg_rider_rating_hist"]
cat_cols = ["user_zone", "rider_zone"]
target_col = "is_delayed"

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
    ]
)

model = LogisticRegression(
    max_iter=1000, solver="saga"
)  # saga for hypertuning regulator
scores = []

param_grid = [
    {
        "model__C": [0.1, 1.0, 10.0, 50.0, 100.0],
        "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
    }
]

pipe = Pipeline([("prep", preprocess), ("model", model)])

grid = GridSearchCV(pipe, param_grid, cv=tss, scoring="accuracy")
grid.fit(df[num_cols + cat_cols], df["is_delayed"])

print(f"Best Params: {grid.best_params_}")
cv_result = pd.DataFrame(grid.cv_results_)
cv_result.to_csv("models/delivery_time_prediction/cv_result.csv")

print(f"Best Accuracy: {grid.best_score_:.4f}")
