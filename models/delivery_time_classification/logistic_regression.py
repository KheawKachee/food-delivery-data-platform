import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

query = """
select
    order_ts,
    delivery_time,
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

# delayed based on mean delivery time
df["is_delayed"] = (df["delivery_time"] > df["delivery_time"].mean()).astype(int)

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
        "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
    }
]

pipe = Pipeline([("prep", preprocess), ("model", model)])

grid = GridSearchCV(pipe, param_grid, cv=tss, scoring="roc_auc")
grid.fit(df[num_cols + cat_cols], df["is_delayed"])

print(f"Best Params: {grid.best_params_}")
cv_result = pd.DataFrame(grid.cv_results_)
cv_result.to_csv("models/delivery_time_prediction/cv_result.csv")

print(f"Best Accuracy: {grid.best_score_:.4f}")

weights = pipe.named_steps["model"].coef_
print(weights)
