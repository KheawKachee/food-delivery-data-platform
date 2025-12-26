import numpy as np
import pandas as pd

import traceback
import sys
import glob
import os
import sys
import datetime as dt
from utils import *

import logging

log = logging.getLogger(__name__)


def data_generator(execution_date: str):
    try:
        # one fixed sample path -> one realization
        execution_date = pd.to_datetime(execution_date)
        start_date = execution_date
        start_order_id = 0
        seed = None
        rng = np.random.default_rng(seed)

        N_USERS = 5000
        N_RIDERS = 1000
        N_ORDERS = 1000

        debug_vars(
            execution_date=execution_date,
            seed=seed,
            n_users=N_USERS,
            n_riders=N_RIDERS,
            n_orders=N_ORDERS,
        )

        DATA_PATH = "data/"
        USERS_DATA_PATH = os.path.join(DATA_PATH, "users.json")
        RIDERS_DATA_PATH = os.path.join(DATA_PATH, "riders.json")

        debug_vars(data_path=DATA_PATH)

        if os.path.exists(USERS_DATA_PATH) and os.path.exists(RIDERS_DATA_PATH):
            print("Files found. Loading existing data...")
            # Load the existing JSON files into DataFrames
            users_df = pd.read_json(USERS_DATA_PATH)
            riders_df = pd.read_json(RIDERS_DATA_PATH)
            users_zone = users_df["zone"].unique()
            riders_zone = riders_df["zone"].unique()

            zones = np.union1d(users_zone, riders_zone)

            order_file = glob.glob(os.path.join(DATA_PATH, "orders_*.json"))
            if order_file:
                latest_file = max(order_file, key=os.path.getmtime)

                latest_filename = (
                    os.path.basename(latest_file)
                    .replace("orders_", "")
                    .replace(".json", "")
                )

                file_dt = pd.to_datetime(latest_filename)

                if file_dt.date() != execution_date.date():
                    previous_orders_df = pd.read_json(latest_file)

                    start_order_id = previous_orders_df["order_id"].max() + 1
                    start_date = pd.to_datetime(previous_orders_df["order_ts"].max())

                    debug_vars(start_date=start_date, start_order_id=start_order_id)
                else:
                    print(f"already generated for this date ({execution_date.date()})")
                    return None

        else:
            print("Files not found. Generating inital data...")
            zones = np.array(["A", "B", "C"])

            # USERS
            users_signup_days = rng.integers(0, 180, N_USERS)

            users_df = pd.DataFrame(
                {
                    "user_id": np.arange(N_USERS),
                    "signup_date": [
                        start_date + dt.timedelta(days=int(d))
                        for d in users_signup_days
                    ],
                    "zone": rng.choice(zones, N_USERS),
                }
            )

            users_df.to_json(
                USERS_DATA_PATH, orient="records", date_format="iso", index=False
            )

            # RIDERS
            riders_signup_days = rng.integers(0, 180, N_RIDERS)
            riders_df = pd.DataFrame(
                {
                    "rider_id": np.arange(N_RIDERS),
                    "signup_date": [
                        start_date + dt.timedelta(days=int(d))
                        for d in riders_signup_days
                    ],
                    "zone": rng.choice(["A", "B", "C"], N_RIDERS),
                }
            )

            riders_df.to_json(
                RIDERS_DATA_PATH, orient="records", date_format="iso", index=False
            )

        # ORDERS : if rider in the same zone as users, more likely to pick that rider , proceeds in matrix operations

        order_user_ids, order_rider_ids, distance = generate_orders(
            users_df, riders_df, zones, N_ORDERS, rng
        )

        order_ts = generate_order_times(start_date, N_ORDERS, rng=rng)

        prep_mins = 5 + 2.5 * rng.exponential(scale=3, size=N_ORDERS)
        prep_timedelta = [dt.timedelta(minutes=m) for m in prep_mins]

        prep_ts = [
            ts + dt.timedelta(minutes=float(d)) for ts, d in zip(order_ts, prep_mins)
        ]

        speed = get_speed(order_ts)
        transport_timedelta = compute_transport_time(distance, speed, rng=rng)
        delivery_ts = [ts + d for ts, d in zip(prep_ts, transport_timedelta)]
        rating = compute_rating(
            [t + 0.25 * p for t, p in zip(transport_timedelta, prep_timedelta)], rng=rng
        )

        price_baht = (
            np.round(50 + distance * rng.uniform(10, 20, N_ORDERS), 2) + distance * 1.25
        )

        orders_df = pd.DataFrame(
            {
                "order_id": np.arange(start_order_id, start_order_id + N_ORDERS),
                "user_id": order_user_ids,
                "rider_id": order_rider_ids,
                "order_ts": order_ts,
                "food_ready_ts": prep_ts,
                "distance_km": distance,
                "delivered_ts": delivery_ts,
                "price_baht": price_baht,
                "rider_rating": rating,
            }
        )

        orders_df["order_ts"] = pd.to_datetime(orders_df["order_ts"])

        orders_df = orders_df[
            orders_df["order_ts"].dt.date
            <= (execution_date + pd.Timedelta(days=1)).date()
        ]

        print((execution_date + pd.Timedelta(days=1)).date())

        orders_df.to_json(
            os.path.join(DATA_PATH, f"orders_{execution_date}.json"),
            orient="records",
            date_format="iso",
            index=False,
        )
        print(f"generate {N_ORDERS} orders finished")
        print(orders_df.describe().T)
    except Exception as e:
        print(
            "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        execution_date = pd.to_datetime(sys.argv[1])
    else:
        execution_date = pd.Timestamp.now()

    data_generator(execution_date)
