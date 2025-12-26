import numpy as np
from datetime import timedelta

import logging

log = logging.getLogger(__name__)


def debug_vars(**variables):
    for name, value in variables.items():
        print(f"{name} : {value}")


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

    return np.array(
        [start_date + timedelta(days=int(days_offset), seconds=int(s)) for s in seconds]
    )


def get_speed(order_times, base_speed=40, rush_speed=24):
    hours = np.array([ot.hour for ot in order_times])
    return np.where(
        ((7 <= hours) & (hours < 9)) | ((17 <= hours) & (hours < 19)),
        rush_speed,
        base_speed,
    )


def compute_transport_time(distance, speed, noise_std=0.2, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(1.0, noise_std, size=len(distance))
    return distance / speed * noise * 60  # minutes


def compute_rating(transport_time, rating_noise=0.3, NAN_RATE=0.1, rng=None):
    rng = rng or np.random.default_rng()

    expected_time = transport_time.mean()
    rating = 5 - np.clip((transport_time - expected_time) / expected_time * 2, 0, 4)

    rating = np.round(rating + rng.normal(0, rating_noise, len(rating)), 1)
    rating = np.clip(rating, 1, 5)

    no_rating_mask = rng.random(len(rating)) < NAN_RATE
    rating[no_rating_mask] = np.nan

    return np.round(rating, 1)
