"""
detection.py — The decision-making "brain" of the pipeline.

  - extract_features()   : 18 kinematic features fed to the frozen RandomForest
                            (fall_rf.joblib). NAMES AND COUNT MUST NOT CHANGE —
                            the model was trained against this exact contract.
  - check_stillness()    : post-impact motionlessness check
  - check_altitude_drop()  : compares a before/after altitude estimate
  - fuse_decision()       : combines ML probability + stillness + altitude
                            into one confirm/suppress decision

Reliability notes
------------------
`fuse_decision()` now clamps its inputs to [0, 1] before combining them.
Previously, a corrupted sensor read or an out-of-range model output could
push the fused score outside [0, 1] in an unbounded way (verified by fuzz
testing), which is exactly the kind of "silently wrong output" a safety
device cannot afford. Clamping makes the fusion score's worst case bounded
and predictable regardless of what upstream sensors report.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Fusion configuration
# ---------------------------------------------------------------------
# ML gets the majority vote (it's the trained classifier). Stillness is a
# well-established, low-noise corroborator. Altitude gets the smallest
# weight because barometric SNR is poor even after filtering.
FUSION_WEIGHTS = {
    "ml": 0.55,
    "stillness": 0.25,
    "altitude": 0.20,
}
assert abs(sum(FUSION_WEIGHTS.values()) - 1.0) < 1e-9, "FUSION_WEIGHTS must sum to 1.0"

FUSION_THRESHOLD = 0.65

# Require the ML model to stay above threshold for this many CONSECUTIVE
# 0.5s evaluation steps before starting fall verification. This is a cheap,
# high-value robustness improvement: it filters out single-window sensor
# glitches without adding any latency to a real fall (a real impact stays
# elevated for multiple consecutive windows).
ML_CONSECUTIVE_TRIGGERS_REQUIRED = 2


def _svm(x, y, z):
    """Signal Vector Magnitude: sqrt(x^2+y^2+z^2), per-sample."""
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def _jerk(signal, dt):
    return np.diff(signal, prepend=signal[0]) / dt


def validate_window(df: pd.DataFrame) -> bool:
    """Guards against corrupted sensor bursts (NaN/Inf/dropped I2C frames)
    reaching the model and producing a meaningless prediction. Returns False
    if the window should be skipped rather than evaluated."""
    cols = ["accel_x_g", "accel_y_g", "accel_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]
    values = df[cols].to_numpy()
    if not np.all(np.isfinite(values)):
        return False
    # A physically impossible reading (way beyond the sensor's +-16g / +-2000dps
    # range) indicates a corrupted I2C frame, not a real fall.
    if np.any(np.abs(df[["accel_x_g", "accel_y_g", "accel_z_g"]].to_numpy()) > 20.0):
        return False
    if np.any(np.abs(df[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()) > 2200.0):
        return False
    return True


def check_stillness(df: pd.DataFrame, accel_std_threshold: float = 0.15,
                     gyro_std_threshold: float = 15.0) -> dict:
    """Evaluates post-impact stillness over a window. A person lying
    motionless after a fall shows low SVM variation in accel and gyro."""
    ax, ay, az = df["accel_x_g"].to_numpy(), df["accel_y_g"].to_numpy(), df["accel_z_g"].to_numpy()
    gx, gy, gz = df["gyro_x_dps"].to_numpy(), df["gyro_y_dps"].to_numpy(), df["gyro_z_dps"].to_numpy()

    a_std = float(np.std(_svm(ax, ay, az)))
    g_std = float(np.std(_svm(gx, gy, gz)))
    is_still = (a_std <= accel_std_threshold) and (g_std <= gyro_std_threshold)

    return {"is_still": is_still, "accel_svm_std": a_std, "gyro_svm_std": g_std}


def extract_features(df: pd.DataFrame) -> dict:
    """Takes one window's dataframe and returns the 18 scalar features the
    frozen RandomForest (fall_rf.joblib) expects, in the exact names it was
    trained with. Do not add/remove/rename features here."""
    ax, ay, az = df["accel_x_g"].to_numpy(), df["accel_y_g"].to_numpy(), df["accel_z_g"].to_numpy()
    gx, gy, gz = df["gyro_x_dps"].to_numpy(), df["gyro_y_dps"].to_numpy(), df["gyro_z_dps"].to_numpy()
    dt = df["t"].iloc[1] - df["t"].iloc[0] if len(df) > 1 else 1 / 200.0

    a_svm = _svm(ax, ay, az)
    g_svm = _svm(gx, gy, gz)
    a_jerk = _jerk(a_svm, dt)

    # Tilt from vertical; Y is "up" at rest on this hardware mounting
    # (accel_y_g ~= -1.0 when worn upright at the waist).
    tilt = np.degrees(np.arctan2(np.sqrt(ax ** 2 + az ** 2), np.abs(ay) + 1e-9))

    return {
        "accel_svm_max": np.max(a_svm),
        "accel_svm_min": np.min(a_svm),
        "accel_svm_std": np.std(a_svm),
        "accel_svm_mean": np.mean(a_svm),
        "accel_svm_range": np.max(a_svm) - np.min(a_svm),
        "jerk_max": np.max(np.abs(a_jerk)),
        "jerk_std": np.std(a_jerk),
        "gyro_svm_max": np.max(g_svm),
        "gyro_svm_std": np.std(g_svm),
        "gyro_svm_mean": np.mean(g_svm),
        "tilt_max": np.max(tilt),
        "tilt_range": np.max(tilt) - np.min(tilt),
        "tilt_std": np.std(tilt),
        "az_min": np.min(az),
        "az_max": np.max(az),
        "ax_range": np.max(ax) - np.min(ax),
        "ay_range": np.max(ay) - np.min(ay),
        "accel_energy": np.sum(a_svm ** 2) / len(a_svm),
    }


def check_altitude_drop(altitude_before_m: float, altitude_after_m: float,
                         drop_threshold_m: float = 0.3) -> dict:
    """Compares a pre-impact baseline altitude to a post-impact settled
    altitude. Feed this KALMAN-FILTERED altitude estimates (see sensors.py)
    — raw pressure readings are too noisy for this comparison to be
    meaningful on their own."""
    drop_m = altitude_before_m - altitude_after_m
    is_drop = drop_m >= drop_threshold_m
    altitude_score = float(np.clip(drop_m / drop_threshold_m, 0.0, 1.0))
    return {
        "is_drop": is_drop,
        "drop_m": drop_m,
        "altitude_score": altitude_score,
        "altitude_before_m": altitude_before_m,
        "altitude_after_m": altitude_after_m,
    }


def fuse_decision(ml_prob: float, is_still: bool, altitude_score: float,
                   weights: dict = FUSION_WEIGHTS,
                   threshold: float = FUSION_THRESHOLD) -> dict:
    """Combine ML confidence + stillness + altitude into one bounded,
    predictable fused score.

    Both ml_prob and altitude_score are clamped to [0, 1] before use, so a
    corrupted or out-of-spec upstream reading can never push fused_score
    outside a known, bounded range.
    """
    ml_prob_c = float(np.clip(ml_prob, 0.0, 1.0))
    altitude_score_c = float(np.clip(altitude_score, 0.0, 1.0))
    stillness_component = 1.0 if is_still else 0.0

    fused_score = (
        weights["ml"] * ml_prob_c
        + weights["stillness"] * stillness_component
        + weights["altitude"] * altitude_score_c
    )

    return {
        "fused_score": fused_score,
        "confirm_fall": fused_score >= threshold,
        "ml_component": weights["ml"] * ml_prob_c,
        "stillness_component": stillness_component,
        "altitude_component": weights["altitude"] * altitude_score_c,
    }
