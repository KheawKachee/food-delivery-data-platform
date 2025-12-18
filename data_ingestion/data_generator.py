import numpy as np
import pandas as pd

import glob
import os
from datetime import datetime, timedelta
from utils import *

# one fixed sample path -> one realization
seed = None
rng = np.random.default_rng(seed)

N_USERS = 5000
N_RIDERS = 1000
N_ORDERS = 5000

DATA_PATH = "data/"
USERS_DATA_PATH = os.path.join(DATA_PATH, "users.json")
RIDERS_DATA_PATH = os.path.join(DATA_PATH, "riders.json")
ORDERS_DATA_PATH = glob.glob(os.path.join(DATA_PATH, "orders_*.json"))

start_date = datetime(2024, 1, 1)
start_order_id = 0

if os.path.exists(USERS_DATA_PATH) and os.path.exists(RIDERS_DATA_PATH):
    print("Files found. Loading existing data...")
    # Load the existing JSON files into DataFrames
    users_df = pd.read_json(USERS_DATA_PATH)
    riders_df = pd.read_json(RIDERS_DATA_PATH)
    users_zone = users_df["zone"].unique()
    riders_zone = riders_df["zone"].unique()

    zones = np.union1d(users_zone, riders_zone)
    if ORDERS_DATA_PATH:
        print("orders.json found. Loading existing orders...")
        previous_orders_df = pd.read_json(max(ORDERS_DATA_PATH))
        N_ORDERS = 5000  # just add more n orders
        start_date = pd.to_datetime(previous_orders_df["order_ts"].max())
        start_order_id = previous_orders_df["order_id"].max() + 1

else:
    print("Files not found. Generating inital data...")
    zones = np.array(["A", "B", "C"])

    # USERS
    users_signup_days = rng.integers(0, 180, N_USERS)

    users_df = pd.DataFrame(
        {
            "user_id": np.arange(N_USERS),
            "signup_date": [
                start_date + timedelta(days=int(d)) for d in users_signup_days
            ],
            "zone": rng.choice(zones, N_USERS),
        }
    )

    users_df.to_json(USERS_DATA_PATH, orient="records", date_format="iso", index=False)

    # RIDERS
    riders_signup_days = rng.integers(0, 180, N_RIDERS)
    riders_df = pd.DataFrame(
        {
            "rider_id": np.arange(N_RIDERS),
            "signup_date": [
                start_date + timedelta(days=int(d)) for d in riders_signup_days
            ],
            "zone": rng.choice(["A", "B", "C"], N_RIDERS),
        }
    )

    riders_df.to_json(
        RIDERS_DATA_PATH, orient="records", date_format="iso", index=False
    )

# ORDERS : if rider in the same zone as users, more likely to pick that rider


user_zones = users_df["zone"].values
rider_zones = riders_df["zone"].values

W_SAME = 3.0
W_DIFF = 1.0

rider_prob_matrix = np.zeros((len(zones), N_RIDERS))

for i, z in enumerate(zones):  # create prob matrix for rider weighted by zone
    weights = np.where(rider_zones == z, W_SAME, W_DIFF)
    rider_prob_matrix[i] = weights / weights.sum()


zone_to_idx = {z: i for i, z in enumerate(zones)}
user_zone_idx = np.array([zone_to_idx[z] for z in users_df["zone"]])

order_rider_ids = np.empty(N_ORDERS, dtype=int)
order_user_ids = rng.integers(0, N_USERS, size=N_ORDERS, dtype=int)
for i in range(N_ORDERS):
    z_idx = user_zone_idx[order_user_ids[i]]
    order_rider_ids[i] = rng.choice(N_RIDERS, p=rider_prob_matrix[z_idx])


distance = np.round(rng.uniform(0.5, 50, N_ORDERS), 2)

order_ts = generate_order_times(start_date, N_ORDERS, rng=rng)

prep_mins = 5 + 2.5 * rng.exponential(scale=3, size=N_ORDERS)

prep_ts = [ts + timedelta(minutes=float(d)) for ts, d in zip(order_ts, prep_mins)]

speed = get_speed(order_ts)
transport_mins = compute_transport_time(distance, speed, rng=rng)
delivery_ts = [
    ts + timedelta(minutes=float(d)) for ts, d in zip(prep_ts, transport_mins)
]
rating = compute_rating(transport_mins + 0.25 * prep_mins, rng=rng)


orders_df = pd.DataFrame(
    {
        "order_id": np.arange(start_order_id, start_order_id + N_ORDERS),
        "user_id": order_user_ids,
        "rider_id": order_rider_ids,
        "order_ts": order_ts,
        "food_ready_ts": prep_ts,
        "distance_km": distance,
        "deliveried_ts": delivery_ts,
        "price_baht": np.round(rng.uniform(50, 500, N_ORDERS), 2),
        "rider_rating": rating,
    }
)


orders_df.to_json(
    os.path.join(DATA_PATH, f"orders_{datetime.now()}.json"),
    orient="records",
    date_format="iso",
    index=False,
)
print(f"generate {N_ORDERS} orders finished")
print(orders_df.describe().T)
