"""
Per-trial feature extraction for fall detection.

Each SisFall file = one trial (one activity performed once). We compute a
single feature vector per trial -- same granularity as the SisFall paper's
own C1-C14 features, so you can sanity-check your numbers against their
published Table S1/S2/S3/S4 results.

Features here are simpler than the paper's exact 14 (they didn't publish
exact formulas for all of them), but cover the same physical intuitions:
peak magnitude, energy, jerk, orientation change -- the standard fall
detection feature families used across the literature.
"""

import numpy as np
import pandas as pd


def _svm(x, y, z):
    """Signal Vector Magnitude: sqrt(x^2+y^2+z^2), per-sample."""
    return np.sqrt(x**2 + y**2 + z**2)


def _jerk(signal, dt):
    """Rate of change of a signal (derivative), per-sample."""
    return np.diff(signal, prepend=signal[0]) / dt


def extract_features(df: pd.DataFrame) -> dict:
    """
    Takes one trial's dataframe (from sisfall_loader.load_file) and returns
    a flat dict of scalar features for that whole trial.
    """
    ax, ay, az = df["accel_x_g"].to_numpy(), df["accel_y_g"].to_numpy(), df["accel_z_g"].to_numpy()
    gx, gy, gz = df["gyro_x_dps"].to_numpy(), df["gyro_y_dps"].to_numpy(), df["gyro_z_dps"].to_numpy()

    dt = df["t"].iloc[1] - df["t"].iloc[0] if len(df) > 1 else 1 / 200.0

    a_svm = _svm(ax, ay, az)
    g_svm = _svm(gx, gy, gz)
    a_jerk = _jerk(a_svm, dt)

    # tilt angle from vertical, assuming z is roughly "up" at rest -- adjust
    # axis if your mounting orientation differs. This is a coarse proxy for
    # orientation change, not a calibrated inclination.
    tilt = np.degrees(np.arctan2(np.sqrt(ax**2 + ay**2), np.abs(az) + 1e-9))

    feats = {
        # accel magnitude features
        "accel_svm_max": np.max(a_svm),
        "accel_svm_min": np.min(a_svm),
        "accel_svm_std": np.std(a_svm),
        "accel_svm_mean": np.mean(a_svm),
        "accel_svm_range": np.max(a_svm) - np.min(a_svm),

        # jerk (impact sharpness)
        "jerk_max": np.max(np.abs(a_jerk)),
        "jerk_std": np.std(a_jerk),

        # gyro magnitude features (rotation during fall)
        "gyro_svm_max": np.max(g_svm),
        "gyro_svm_std": np.std(g_svm),
        "gyro_svm_mean": np.mean(g_svm),

        # orientation change
        "tilt_max": np.max(tilt),
        "tilt_range": np.max(tilt) - np.min(tilt),
        "tilt_std": np.std(tilt),

        # per-axis extremes (falls are often direction-specific: forward/back/lateral)
        "az_min": np.min(az),   # sharp negative Z = impact after free-fall dip
        "az_max": np.max(az),
        "ax_range": np.max(ax) - np.min(ax),
        "ay_range": np.max(ay) - np.min(ay),

        # energy
        "accel_energy": np.sum(a_svm**2) / len(a_svm),
    }
    # NOTE: duration_s intentionally excluded. SisFall's protocol fixes fall
    # trials at 15s and varies ADL duration by activity type (12-100s) --
    # any model using trial length picks up on the *protocol*, not the fall
    # physics, and will do nothing useful on continuous real-time streams
    # where there's no such thing as "trial duration" ahead of time.
    return feats


def build_feature_row(meta: dict, df: pd.DataFrame) -> dict:
    """Combines metadata + extracted features into one row for the feature table."""
    row = dict(meta)
    row.update(extract_features(df))
    return row