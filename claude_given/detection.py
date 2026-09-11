"""
detection.py (v2) — The decision-making "brain" of the pipeline.

  - extract_features()   : 22 GRAVITY-RELATIVE kinematic features fed to the
                            retrained RandomForest (fall_rf_v2.joblib).
                            NAMES AND COUNT MUST NOT CHANGE — the model was
                            trained against this exact contract.
  - check_stillness()    : post-impact motionlessness check
  - check_altitude_drop()  : compares a before/after altitude estimate
  - fuse_decision()       : combines ML probability + stillness + altitude
                            into one confirm/suppress decision

v2 CHANGE — why extract_features() now takes a gravity_vec argument
---------------------------------------------------------------------
The original feature set used raw signed ax/ay/az (e.g. ax_range, az_min/max)
under the assumption that the device always mounts with a fixed axis "up"
(the old comment claimed accel_y_g ~= -1.0 at rest). Checking real recorded
sessions showed this assumption doesn't hold: resting accel_y_g was measured
at +1.0 in one session and the estimated gravity direction differed subject
to subject and even session to session for the same subject (e.g. one
session's dominant gravity axis had the OPPOSITE sign vs. the other nine).
Axis-specific features silently break under that kind of mounting drift.

The fix: every window is now decomposed relative to a locally-estimated
gravity direction (gravity_vec, a unit vector) into VERTICAL (along gravity)
and HORIZONTAL (perpendicular to gravity) components, and tilt is computed
as the angle between the instantaneous accel vector and gravity_vec. This
makes the feature set robust to how the device actually sits on the body,
instead of assuming one fixed orientation.

gravity_vec should be estimated once at startup (see
sensors.estimate_gravity_vector / main.py's calibrate_orientation()) from a
short still period, the same way the barometer's reference pressure is
calibrated at startup.

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


FEATURE_COLS_V2 = [
    "accel_svm_max", "accel_svm_min", "accel_svm_std", "accel_svm_mean", "accel_svm_range",
    "jerk_max", "jerk_std", "gyro_svm_max", "gyro_svm_std", "gyro_svm_mean",
    "tilt_max", "tilt_range", "tilt_std",
    "vertical_min", "vertical_max", "vertical_range", "horizontal_max", "horizontal_std",
    "accel_energy", "free_fall_dip", "time_dip_to_peak", "post_tail_std",
]


def extract_features(df: pd.DataFrame, gravity_vec) -> dict:
    """Takes one window's dataframe plus the locally-calibrated gravity unit
    vector and returns the 22 scalar features the retrained RandomForest
    (fall_rf_v2.joblib) expects, in the exact names it was trained with.
    Do not add/remove/rename features here without retraining the model.
    """
    ax, ay, az = df["accel_x_g"].to_numpy(), df["accel_y_g"].to_numpy(), df["accel_z_g"].to_numpy()
    gx, gy, gz = df["gyro_x_dps"].to_numpy(), df["gyro_y_dps"].to_numpy(), df["gyro_z_dps"].to_numpy()
    dt = df["t"].iloc[1] - df["t"].iloc[0] if len(df) > 1 else 1 / 200.0

    gvec = np.asarray(gravity_vec, dtype=float)
    gvec = gvec / np.linalg.norm(gvec)

    a_svm = _svm(ax, ay, az)
    g_svm = _svm(gx, gy, gz)
    a_jerk = _jerk(a_svm, dt)

    # Vertical = projection onto the calibrated gravity axis; horizontal =
    # everything perpendicular to it. Orientation-invariant by construction.
    vertical = ax * gvec[0] + ay * gvec[1] + az * gvec[2]
    horizontal = np.sqrt(np.clip((ax ** 2 + ay ** 2 + az ** 2) - vertical ** 2, 0, None))
    tilt = np.degrees(np.arccos(np.clip(vertical / (a_svm + 1e-9), -1, 1)))

    # Free-fall dip: lowest SVM point BEFORE the impact peak (the near-
    # weightlessness signature that precedes a real impact), and how long
    # after that dip the impact peak occurs.
    peak_idx = int(np.argmax(a_svm))
    pre_peak = a_svm[:peak_idx + 1] if peak_idx > 0 else a_svm[:1]
    free_fall_dip = float(np.min(pre_peak))
    dip_idx = int(np.argmin(pre_peak))
    time_dip_to_peak = float((peak_idx - dip_idx) * dt)

    # Post-impact stillness within THIS window (cheap early signal; the
    # dedicated check_stillness() below does the authoritative check on the
    # dedicated post-trigger window).
    post_tail_std = float(np.std(a_svm[int(len(a_svm) * 0.75):]))

    return {
        "accel_svm_max": float(np.max(a_svm)),
        "accel_svm_min": float(np.min(a_svm)),
        "accel_svm_std": float(np.std(a_svm)),
        "accel_svm_mean": float(np.mean(a_svm)),
        "accel_svm_range": float(np.max(a_svm) - np.min(a_svm)),
        "jerk_max": float(np.max(np.abs(a_jerk))),
        "jerk_std": float(np.std(a_jerk)),
        "gyro_svm_max": float(np.max(g_svm)),
        "gyro_svm_std": float(np.std(g_svm)),
        "gyro_svm_mean": float(np.mean(g_svm)),
        "tilt_max": float(np.max(tilt)),
        "tilt_range": float(np.max(tilt) - np.min(tilt)),
        "tilt_std": float(np.std(tilt)),
        "vertical_min": float(np.min(vertical)),
        "vertical_max": float(np.max(vertical)),
        "vertical_range": float(np.max(vertical) - np.min(vertical)),
        "horizontal_max": float(np.max(horizontal)),
        "horizontal_std": float(np.std(horizontal)),
        "accel_energy": float(np.sum(a_svm ** 2) / len(a_svm)),
        "free_fall_dip": free_fall_dip,
        "time_dip_to_peak": time_dip_to_peak,
        "post_tail_std": post_tail_std,
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
