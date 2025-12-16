import numpy as np
from datetime import timedelta

def generate_order_times(start_date, N_ORDERS, max_days=180, rng=None):
    rng = rng or np.random.default_rng()
    day_offsets = rng.integers(0, max_days, N_ORDERS)

    weights = [0.6, 0.4]  # probability of picking lunch vs dinner (mixture of gaussian)
    means = [12*3600, 18*3600] # lunch(12) vs dinner(18)
    stds = [2*3600, 3*3600]

    choices = rng.choice([0,1], size=N_ORDERS, p=weights)
    seconds = rng.normal(loc=np.array(means)[choices], scale=np.array(stds)[choices])
    offpeak_noise = rng.standard_normal(N_ORDERS) * 300  # small uniform noise with mean at 5 mins
    seconds += offpeak_noise
    seconds = np.clip(seconds, 0, 24*3600-1).astype(int)

    return np.array([start_date + timedelta(days=int(d), seconds=int(s))
                     for d, s in zip(day_offsets, seconds)])


def get_speed(order_times, base_speed=40, rush_speed=24):
    hours = np.array([ot.hour for ot in order_times])
    return np.where(
        ((7 <= hours) & (hours < 9)) | ((17 <= hours) & (hours < 19)),
        rush_speed,
        base_speed
    )

def compute_transport_time(distance, speed, noise_std=0.2, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(1.0, noise_std, size=len(distance))
    return distance / speed * noise * 60  # minutes

def compute_rating(transport_time, rating_noise=0.3, rng=None):
    rng = rng or np.random.default_rng()
    expected_time = transport_time.mean()
    rating = 5 - np.clip((transport_time - expected_time) / expected_time * 2, 0, 4)
    rating = np.round(rating + rng.normal(0, rating_noise, len(rating)), 1)
    return np.clip(rating, 1, 5)