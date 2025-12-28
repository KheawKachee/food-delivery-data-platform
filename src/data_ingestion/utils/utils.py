import numpy as np
import pandas as pd

import datetime as dt

import logging

log = logging.getLogger(__name__)


def debug_vars(**variables):
    for name, value in variables.items():
        print(f"{name} : {value}")


def generate_orders(users_df, riders_df, zones, N_ORDERS, rng, mean_extra_distance=25):
    """
    Generate orders with rider assignment weighted by zone and compute distances.

    Args:
        users_df: DataFrame with at least a 'zone' column.
        riders_df: DataFrame with at least a 'zone' column.
        zones: List of all zones.
        N_ORDERS: Number of orders to simulate.
        rng: np.random.Generator instance.
        extra_distance: Additional distance if rider is in a different zone.

    Returns:
        dict with keys: 'order_user_ids', 'order_rider_ids', 'distance'
    """
    rider_zones = riders_df["zone"].values
    W_SAME, W_DIFF = 3.0, 1.0

    # Build rider probability matrix: zones x riders
    rider_prob_matrix = np.array(
        [
            np.where(rider_zones == z, W_SAME, W_DIFF)
            / np.sum(np.where(rider_zones == z, W_SAME, W_DIFF))
            for z in zones
        ]
    )

    # Map user zones to indices
    zone_to_idx = {z: i for i, z in enumerate(zones)}
    user_zone_idx = np.array([zone_to_idx[z] for z in users_df["zone"]])

    # Generate orders
    order_user_ids = rng.integers(0, len(users_df), size=N_ORDERS)
    uz_indices = user_zone_idx[order_user_ids]

    # Assign riders based on zone probability
    order_rider_ids = np.array(
        [rng.choice(len(riders_df), p=rider_prob_matrix[uz]) for uz in uz_indices]
    )

    # Compute distances: compare actual rider zone with user zone
    user_zone = users_df["zone"].iloc[order_user_ids].values
    rider_zone = riders_df["zone"].iloc[order_rider_ids].values
    base_distances = np.round(rng.uniform(0.5, 25, size=N_ORDERS), 2)
    extra = rng.normal(mean_extra_distance, 5, size=N_ORDERS)
    distance = base_distances + np.where(user_zone != rider_zone, extra, 0)
    return order_user_ids, order_rider_ids, distance


def generate_order_times(
    start_date, N_ORDERS, days_offset=1, rng=None, certainness=0.6
):
    rng = rng or np.random.default_rng()

    means = [12 * 3600, 18 * 3600]
    stds = [1.5 * 3600, 2 * 3600]

    n_certain = int(N_ORDERS * certainness)
    choices = rng.choice([0, 1], size=n_certain, p=[0.55, 0.45])
    seconds_peaks = rng.normal(
        loc=np.array(means)[choices], scale=np.array(stds)[choices]
    )

    # 2. Generate 10% pure random noise (Any time of day)
    n_noise = N_ORDERS - n_certain
    seconds_noise = rng.random((n_noise,)) * 24 * 3600

    # 3. Combine them
    seconds = np.concatenate([seconds_peaks, seconds_noise])
    seconds = np.clip(seconds, 0, 24 * 3600 - 1).astype(int)

    return np.array([start_date + dt.timedelta(seconds=int(s)) for s in seconds])


def get_speed(order_times, base_speed=40, rush_speed=24):
    hours = np.array([ot.hour for ot in order_times])
    return np.where(
        ((7 <= hours) & (hours <= 9)) | ((17 <= hours) & (hours <= 19)),
        rush_speed,
        base_speed,
    ).astype(float)


def compute_transport_time(distance, speed, noise_std=0.2, rng=None):
    rng = rng or np.random.default_rng()
    distance = np.array(distance, dtype=float)
    speed = np.array(speed, dtype=float)
    noise = rng.normal(1.0, noise_std, size=len(distance))

    hours = distance / (speed * noise)
    transport_timedelta = [dt.timedelta(hours=h) for h in hours]
    return transport_timedelta


def compute_rating(transport_time, rating_noise=0.3, NAN_RATE=0.1, rng=None):
    rng = rng or np.random.default_rng()

    # Convert timedelta to hours
    transport_seconds = np.array([t.total_seconds() for t in transport_time])

    expected_time = transport_seconds.mean()
    rating = 5 - np.clip((transport_seconds - expected_time) / expected_time * 2, 0, 4)

    rating = np.round(rating + rng.normal(0, rating_noise, len(rating)), 1)
    rating = np.clip(rating, 1, 5)

    no_rating_mask = rng.random(len(rating)) < NAN_RATE
    rating[no_rating_mask] = np.nan

    return np.round(rating, 1)


if __name__ == "__main__":
    # Simple test
    rng = np.random.default_rng(42)
    users_df = pd.DataFrame({"user_id": [0, 1, 2, 3], "zone": ["A", "B", "A", "C"]})
    riders_df = pd.DataFrame(
        {"rider_id": [0, 1, 2, 3, 4], "zone": ["A", "B", "C", "A", "B"]}
    )
    zones = ["A", "B", "C"]
    N_ORDERS = 10

    order_user_ids, order_rider_ids, distance = generate_orders(
        users_df, riders_df, zones, N_ORDERS, rng
    )

    print("Order User IDs:", order_user_ids)
    print("Order Rider IDs:", order_rider_ids)
    print("Distances:", distance)
