"""Data utilities: synthetic sensor-style data and a loader for the UCI HAR dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

FEATURES = [
    "acc_mean_x", "acc_mean_y", "acc_mean_z",
    "acc_std_x", "acc_std_y", "acc_std_z",
    "acc_mag_mean", "acc_mag_std", "gyro_std",
    "dominant_freq", "energy", "jerk_mean",
]

# Typical feature values (one row per activity, same order as FEATURES).
# Gravity direction separates upright (x) from laying (z); the std/energy/frequency
# features separate resting from walking and the three walking types from each other.
_PROFILE = {
    "WALKING":            [0.95, -0.05, 0.10, 0.45, 0.30, 0.35, 1.10, 0.40, 0.55, 1.90, 0.50, 0.60],
    "WALKING_UPSTAIRS":   [0.90, -0.08, 0.18, 0.55, 0.38, 0.45, 1.15, 0.50, 0.70, 1.60, 0.70, 0.75],
    "WALKING_DOWNSTAIRS": [0.92, -0.04, 0.12, 0.62, 0.42, 0.50, 1.20, 0.58, 0.80, 2.15, 0.85, 0.95],
    "SITTING":            [0.82, -0.10, 0.30, 0.04, 0.04, 0.04, 1.00, 0.04, 0.06, 0.30, 0.03, 0.05],
    "STANDING":           [0.95, -0.06, 0.15, 0.05, 0.05, 0.05, 1.00, 0.05, 0.07, 0.35, 0.04, 0.06],
    "LAYING":             [0.12,  0.05, 0.95, 0.03, 0.03, 0.03, 1.00, 0.03, 0.04, 0.20, 0.02, 0.03],
}
_NOISE = np.array([0.07, 0.07, 0.08, 0.10, 0.08, 0.09, 0.10, 0.10, 0.12, 0.25, 0.14, 0.15])
_REST_NOISE = np.array([0.07, 0.07, 0.08, 0.025, 0.025, 0.025, 0.03, 0.025, 0.03, 0.12, 0.02, 0.03])


def sample_activity(activity, n, rng):
    """Draw n feature windows for one activity."""
    mean = np.array(_PROFILE[activity])
    noise = _REST_NOISE if activity in ("SITTING", "STANDING", "LAYING") else _NOISE
    x = rng.normal(mean, noise, size=(n, len(FEATURES)))
    x[:, 3:] = np.abs(x[:, 3:])  # std, frequency, energy and jerk cannot be negative
    return pd.DataFrame(x, columns=FEATURES)


def generate_dataset(n_per_class=600, seed=42):
    """Labelled synthetic training data (clearly synthetic: for testing and demos)."""
    rng = np.random.default_rng(seed)
    parts = []
    for act in ACTIVITIES:
        df = sample_activity(act, n_per_class, rng)
        df["activity"] = act
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# Share of each day spent in each activity (order of ACTIVITIES); the last day is a rest day.
_DAY_MIX = [
    [0.30, 0.05, 0.05, 0.30, 0.15, 0.15],
    [0.25, 0.08, 0.07, 0.30, 0.15, 0.15],
    [0.35, 0.06, 0.04, 0.25, 0.15, 0.15],
    [0.10, 0.02, 0.02, 0.40, 0.21, 0.25],
    [0.32, 0.07, 0.06, 0.25, 0.15, 0.15],
    [0.28, 0.10, 0.07, 0.25, 0.15, 0.15],
    [0.03, 0.00, 0.00, 0.40, 0.17, 0.40],
]


def generate_sample_session(user="User 01", start="2026-05-04", windows_per_day=1500, seed=7):
    """One week of unlabelled readings (user, date + feature columns) for the demo."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    parts = []
    for d, mix in enumerate(_DAY_MIX):
        counts = rng.multinomial(windows_per_day, mix)
        day = pd.concat([sample_activity(a, c, rng) for a, c in zip(ACTIVITIES, counts) if c > 0], ignore_index=True)
        day = day.sample(frac=1, random_state=d).reset_index(drop=True)
        day.insert(0, "date", (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d"))
        day.insert(0, "user", user)
        parts.append(day)
    return pd.concat(parts, ignore_index=True)


def load_uci_har(root):
    """Load the real UCI HAR dataset from the extracted 'UCI HAR Dataset' folder.

    Download: https://archive.ics.uci.edu/dataset/240 (Human Activity Recognition Using Smartphones).
    """
    root = Path(root)
    names = pd.read_csv(root / "features.txt", sep=r"\s+", header=None)[1].tolist()
    names = [f"{n}_{i}" for i, n in enumerate(names)]  # the original names contain duplicates
    labels = pd.read_csv(root / "activity_labels.txt", sep=r"\s+", header=None, index_col=0)[1].to_dict()
    frames = []
    for split in ("train", "test"):
        x = pd.read_csv(root / split / f"X_{split}.txt", sep=r"\s+", header=None, names=names)
        y = pd.read_csv(root / split / f"y_{split}.txt", header=None)[0].map(labels)
        x["activity"] = y.values
        frames.append(x)
    return pd.concat(frames, ignore_index=True)
