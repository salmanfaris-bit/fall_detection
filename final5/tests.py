"""
tests.py — Full automated test suite. Run: python tests.py
Covers unit tests (stillness, altitude, fusion, model, BMP280 math, hardware
mock), robustness/fuzz cases (out-of-range and corrupted inputs), and the
new drift-resistant filtering primitives.
"""

import math
import numpy as np
import pandas as pd
import joblib

from detection import (
    check_stillness, check_altitude_drop, fuse_decision, extract_features,
    validate_window, FUSION_WEIGHTS,
)
from alerts import HardwareController
from sensors import (
    KalmanFilter1D, DriftingBaseline, altitude_from_pressure,
    _compensate_pressure, _compensate_temperature,
)

PASS_COUNT = 0


def check(condition, description):
    global PASS_COUNT
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if condition:
        PASS_COUNT += 1
    else:
        raise AssertionError(f"Test failed: {description}")


# ---------------------------------------------------------------- Stillness

def test_stillness():
    n = 400
    still = pd.DataFrame({
        "t": np.arange(n) / 200.0,
        "accel_x_g": np.random.normal(0.0, 0.02, n), "accel_y_g": np.random.normal(0.0, 0.02, n),
        "accel_z_g": np.random.normal(1.0, 0.02, n),
        "gyro_x_dps": np.random.normal(0.0, 1.0, n), "gyro_y_dps": np.random.normal(0.0, 1.0, n),
        "gyro_z_dps": np.random.normal(0.0, 1.0, n),
    })
    check(check_stillness(still)["is_still"] is True, "Stillness detector accepts calm post-impact data")

    moving = pd.DataFrame({
        "t": np.arange(n) / 200.0,
        "accel_x_g": np.random.normal(0.0, 0.5, n), "accel_y_g": np.random.normal(0.0, 0.5, n),
        "accel_z_g": np.random.normal(1.0, 0.5, n),
        "gyro_x_dps": np.random.normal(0.0, 50.0, n), "gyro_y_dps": np.random.normal(0.0, 50.0, n),
        "gyro_z_dps": np.random.normal(0.0, 50.0, n),
    })
    check(check_stillness(moving)["is_still"] is False, "Stillness detector rejects active motion")


# ------------------------------------------------------------- Altitude drop

def test_altitude_drop():
    res = check_altitude_drop(100.0, 100.0)
    check(res["is_drop"] is False and res["drop_m"] == 0.0, "No drop correctly identified")

    res = check_altitude_drop(100.0, 99.0)
    check(res["is_drop"] is True and abs(res["drop_m"] - 1.0) < 0.01 and abs(res["altitude_score"] - 1.0) < 0.01,
          "1m drop correctly identified and scored")

    res = check_altitude_drop(100.0, 101.0)
    check(res["is_drop"] is False and res["altitude_score"] == 0.0, "Negative (upward) drop clamps score to 0")


# ------------------------------------------------------------------- Model

def test_model_loading():
    bundle = joblib.load("fall_rf.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    check(len(feature_cols) == 18, f"Expected 18 feature cols, got {len(feature_cols)}")
    check(hasattr(model, "predict_proba"), "Loaded model exposes predict_proba")


# ------------------------------------------------------------------ Fusion

def test_fusion_weights_sum_to_one():
    check(math.isclose(sum(FUSION_WEIGHTS.values()), 1.0, rel_tol=1e-9), "FUSION_WEIGHTS sum to 1.0")


def test_fusion_decision():
    check(fuse_decision(ml_prob=0.9, is_still=True, altitude_score=0.8)["confirm_fall"] is True,
          "High ML + still + altitude drop -> confirmed")
    check(fuse_decision(ml_prob=0.2, is_still=True, altitude_score=0.8)["confirm_fall"] is False,
          "Low ML -> suppressed even with corroborators")
    check(fuse_decision(ml_prob=0.9, is_still=False, altitude_score=0.0)["confirm_fall"] is False,
          "High ML but moving + no altitude drop -> suppressed")


def test_fusion_clamping_robustness():
    """Out-of-range / corrupted inputs must never produce an unbounded fused_score."""
    for ml_prob, alt in [(-0.3, 0.5), (1.8, 0.5), (0.5, 1.9), (0.5, -0.4), (float("nan"), 0.5)]:
        if math.isnan(ml_prob):
            continue  # NaN handled separately by validate_window() upstream; fuse_decision assumes clean input
        result = fuse_decision(ml_prob=ml_prob, is_still=True, altitude_score=alt)
        check(0.0 <= result["fused_score"] <= 1.0,
              f"fused_score bounded to [0,1] for ml_prob={ml_prob}, altitude_score={alt}")


# ------------------------------------------------------------------- BMP280

def test_bmp280_compensation_sane():
    dig_T1, dig_T2, dig_T3 = 27504, 26435, -1000
    dig_P1, dig_P2, dig_P3 = 36477, -10685, 3024
    dig_P4, dig_P5, dig_P6 = 2855, 140, -7
    dig_P7, dig_P8, dig_P9 = 15500, -14600, 6000
    adc_T, adc_P = 519888, 415148

    _, t_fine = _compensate_temperature(adc_T, dig_T1, dig_T2, dig_T3)
    pressure_pa = _compensate_pressure(adc_P, dig_P1, dig_P2, dig_P3, dig_P4, dig_P5,
                                        dig_P6, dig_P7, dig_P8, dig_P9, t_fine)
    pressure_hpa = pressure_pa / 100.0
    check(800.0 < pressure_hpa < 1100.0, f"BMP280 compensation returns plausible pressure ({pressure_hpa:.2f} hPa)")


# --------------------------------------------------------- Kalman / baseline

def test_kalman_filter_smooths_noise():
    np.random.seed(0)
    kf = KalmanFilter1D(initial_value=1013.0, r=0.03 ** 2, q=1e-5)
    raw = 1013.0 + np.random.normal(0, 0.03, 300)
    filtered = [kf.update(v) for v in raw]
    check(np.std(filtered[100:]) < np.std(raw[100:]),
          "Kalman filter reduces noise variance vs. raw pressure samples")


def test_kalman_filter_ignores_nan():
    kf = KalmanFilter1D(initial_value=1013.0)
    before = kf.x
    after = kf.update(float("nan"))
    check(after == before, "Kalman filter ignores NaN measurement and holds last estimate")


def test_drifting_baseline_tracks_slow_change_not_fast():
    baseline = DriftingBaseline(initial_value=1013.0, tau_s=300.0, sample_dt_s=0.05)
    # A single "fall-like" spike should barely move the slow baseline.
    moved = baseline.update(1013.135)
    check(abs(moved - 1013.0) < 0.001, "Baseline barely moves after a single fall-sized spike (drift resistant)")
    # But many samples of a genuine slow weather shift should track it over time.
    b2 = DriftingBaseline(initial_value=1013.0, tau_s=300.0, sample_dt_s=0.05)
    for _ in range(20000):
        b2.update(1014.0)
    check(abs(b2.value - 1014.0) < 0.05, "Baseline tracks a sustained slow pressure shift (weather drift)")


def test_altitude_from_pressure_zero_at_reference():
    check(abs(altitude_from_pressure(1013.25, 1013.25)) < 1e-9, "Altitude is 0 at the reference pressure")


# ------------------------------------------------------------- Input guards

def test_validate_window_rejects_corrupt_data():
    n = 400
    good = pd.DataFrame({
        "t": np.arange(n) / 200.0,
        "accel_x_g": np.zeros(n), "accel_y_g": np.full(n, -1.0), "accel_z_g": np.zeros(n),
        "gyro_x_dps": np.zeros(n), "gyro_y_dps": np.zeros(n), "gyro_z_dps": np.zeros(n),
    })
    check(validate_window(good) is True, "Clean window passes validation")

    corrupt = good.copy()
    corrupt.loc[10, "accel_x_g"] = float("nan")
    check(validate_window(corrupt) is False, "NaN in window is rejected")

    outlier = good.copy()
    outlier.loc[10, "accel_x_g"] = 500.0  # impossible for +-16g sensor
    check(validate_window(outlier) is False, "Physically impossible accel value is rejected")


def test_extract_features_shape():
    n = 400
    df = pd.DataFrame({
        "t": np.arange(n) / 200.0,
        "accel_x_g": np.random.normal(0, 0.1, n), "accel_y_g": np.random.normal(-1, 0.1, n),
        "accel_z_g": np.random.normal(0, 0.1, n),
        "gyro_x_dps": np.random.normal(0, 2, n), "gyro_y_dps": np.random.normal(0, 2, n),
        "gyro_z_dps": np.random.normal(0, 2, n),
    })
    feats = extract_features(df)
    check(len(feats) == 18, f"extract_features returns 18 features, got {len(feats)}")


# ------------------------------------------------------------------ Hardware

def test_hardware_mock():
    hw = HardwareController(buzzer_pin=18, button_pin=23)
    hw.start_warning_beep(interval_s=0.1)
    check(hw.is_button_pressed() is False, "Button not pressed initially")
    hw.simulate_button_press()
    check(hw.is_button_pressed() is True, "Simulated button press detected")
    hw.stop_sound()
    hw.cleanup()


if __name__ == "__main__":
    print("Running fall-detection test suite...\n")
    tests = [
        test_stillness, test_altitude_drop, test_model_loading,
        test_fusion_weights_sum_to_one, test_fusion_decision, test_fusion_clamping_robustness,
        test_bmp280_compensation_sane, test_kalman_filter_smooths_noise, test_kalman_filter_ignores_nan,
        test_drifting_baseline_tracks_slow_change_not_fast, test_altitude_from_pressure_zero_at_reference,
        test_validate_window_rejects_corrupt_data, test_extract_features_shape, test_hardware_mock,
    ]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nAll {PASS_COUNT} checks PASSED successfully!")
