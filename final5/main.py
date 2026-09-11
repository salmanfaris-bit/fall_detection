"""
main.py — Live fall-detection pipeline. Run this on the Raspberry Pi 4.

Pipeline:
  1. BNO055 IMU @ 200Hz -> 2.0s sliding window
  2. RandomForest classifier -> fall probability (must stay elevated for
     ML_CONSECUTIVE_TRIGGERS_REQUIRED windows in a row before verification
     starts, to reject single-window sensor glitches)
  3. Parallel verification: post-impact stillness + Kalman-filtered
     barometric altitude drop
  4. Fusion decision (clamped, bounded) -> confirm or suppress
  5. 30s cancellable warning countdown -> final alarm

Barometer handling: pressure is smoothed with KalmanFilter1D (fast, removes
sensor noise) and the "resting" reference is tracked with DriftingBaseline
(slow, absorbs weather drift). The baseline is only updated during normal
MONITORING and is frozen the instant a candidate fall starts verifying, so a
real fall is never averaged away.

Every confirmed/cancelled/false-alarm event is appended to events.csv for
post-incident review.
"""

import time
import csv
import os
import collections

import numpy as np
import pandas as pd
import joblib

from sensors import (
    BNO055Reader, MockBNO055Reader,
    BMP280Reader, MockBMP280Reader,
    KalmanFilter1D, DriftingBaseline, altitude_from_pressure,
)
from detection import (
    extract_features, check_stillness, check_altitude_drop, fuse_decision,
    validate_window, FUSION_THRESHOLD, ML_CONSECUTIVE_TRIGGERS_REQUIRED,
)
from alerts import HardwareController

MODEL_PATH = "fall_rf.joblib"
TARGET_HZ = 200
WINDOW_SECONDS = 2.0
STEP_SECONDS = 0.5
STILLNESS_WINDOW_S = 2.0
WARNING_COUNTDOWN_S = 30.0
FALL_PROB_THRESHOLD = 0.65
ALT_DROP_THRESHOLD_M = 0.3
ALT_SAMPLE_RATE_DIVISOR = 10   # sample barometer at ~20Hz effective
EVENTS_LOG_PATH = "events.csv"

WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_HZ)
STEP_SAMPLES = int(STEP_SECONDS * TARGET_HZ)
STILLNESS_SAMPLES = int(STILLNESS_WINDOW_S * TARGET_HZ)

STATE_MONITORING = "MONITORING"
STATE_VERIFYING = "VERIFYING"
STATE_WARNING_COUNTDOWN = "WARNING_COUNTDOWN"
STATE_ALARM_ACTIVE = "ALARM_ACTIVE"


def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def buffer_to_df(buffer, target_hz):
    df = pd.DataFrame(buffer)
    df["t"] = np.arange(len(buffer)) / target_hz
    return df


def log_event(kind: str, detail: str):
    is_new = not os.path.exists(EVENTS_LOG_PATH)
    with open(EVENTS_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "event", "detail"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), kind, detail])


def init_sensors():
    """Real hardware first, mock fallback second — never silently pretend
    mock data is real."""
    shared_counter = [0]

    try:
        imu = BNO055Reader(address=0x28)
        imu.configure()
        print("Connected to BNO055 IMU over I2C.")
    except Exception as e:
        print(f"[Notice] BNO055 unavailable ({e}). Using MockBNO055Reader.")
        imu = MockBNO055Reader()
        imu.configure()
        imu.set_counter(shared_counter)

    try:
        baro = BMP280Reader(address=0x76)
        baro.configure()
        print("Connected to BMP280 over I2C.")
    except Exception as e:
        print(f"[Notice] BMP280 unavailable ({e}). Using MockBMP280Reader.")
        baro = MockBMP280Reader()
        baro.configure()
    baro.set_counter(shared_counter) if hasattr(baro, "set_counter") else None

    return imu, baro


def calibrate_barometer(baro, duration_s: float = 5.0):
    """Average pressure over duration_s (wearer should stand still) to seed
    the Kalman filter and the slow drift baseline."""
    print(f"Calibrating barometer reference over {duration_s:.0f}s (stand still)...")
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        p, _ = baro.read_raw()
        samples.append(p)
        time.sleep(0.05)
    ref = float(np.mean(samples)) if samples else 1013.25
    print(f"Reference pressure: {ref:.3f} hPa")
    return ref


def main():
    print("==========================================================")
    print(" Wearable Fall Detection Pipeline")
    print(" ML (RandomForest) + Stillness + Kalman-filtered Altitude")
    print(" 30s cancellable warning countdown")
    print("==========================================================")

    print("\nLoading model...")
    model, feature_cols = load_model(MODEL_PATH)
    print(f"Model loaded. Features ({len(feature_cols)}): {feature_cols}")

    hw = HardwareController(buzzer_pin=18, button_pin=23)
    imu, baro = init_sensors()

    reference_hpa = calibrate_barometer(baro)
    kalman = KalmanFilter1D(initial_value=reference_hpa)
    baseline = DriftingBaseline(initial_value=reference_hpa, tau_s=300.0,
                                 sample_dt_s=ALT_SAMPLE_RATE_DIVISOR / TARGET_HZ)

    buffer = collections.deque(maxlen=WINDOW_SAMPLES)
    stillness_buffer = []
    altitude_deque = collections.deque(maxlen=500)

    state = STATE_MONITORING
    samples_since_eval = 0
    consecutive_triggers = 0
    countdown_start = 0.0
    last_print = 0.0
    fall_prob = 0.0
    altitude_baseline_m = 0.0
    verify_tick = 0

    interval = 1.0 / TARGET_HZ
    next_t = time.time()

    print(f"\n[System Active] Monitoring at {TARGET_HZ}Hz. Ctrl+C to stop.\n")

    try:
        while True:
            try:
                accel_g, gyro_dps = imu.read_raw()
            except Exception as e:
                print(f"[WARN] IMU read failed, skipping sample: {e}")
                next_t += interval
                time.sleep(max(0, next_t - time.time()))
                continue

            sample = {
                "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
                "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
            }

            # ------------------------------------------------- MONITORING
            if state == STATE_MONITORING:
                buffer.append(sample)
                samples_since_eval += 1

                if samples_since_eval % ALT_SAMPLE_RATE_DIVISOR == 0:
                    try:
                        p_raw, _ = baro.read_raw()
                        p_filtered = kalman.update(p_raw)
                        reference_hpa = baseline.update(p_filtered)  # drift tracking, MONITORING only
                        alt_m = altitude_from_pressure(p_filtered, reference_hpa)
                        altitude_deque.append(alt_m)
                    except Exception:
                        pass

                if len(buffer) == WINDOW_SAMPLES and samples_since_eval >= STEP_SAMPLES:
                    samples_since_eval = 0
                    window_df = buffer_to_df(list(buffer), TARGET_HZ)

                    if not validate_window(window_df):
                        print("[WARN] Corrupted sensor window skipped.        ", end="\r")
                    else:
                        feats = extract_features(window_df)
                        x = np.array([[feats[c] for c in feature_cols]])
                        pred = model.predict(x)[0]
                        prob = model.predict_proba(x)[0]
                        fall_prob = float(prob[1] if len(prob) > 1 else prob[0])

                        if pred == 1 and fall_prob >= FALL_PROB_THRESHOLD:
                            consecutive_triggers += 1
                        else:
                            consecutive_triggers = 0

                        if consecutive_triggers >= ML_CONSECUTIVE_TRIGGERS_REQUIRED:
                            alt_window = list(altitude_deque)[-20:]  # ~1s at 20Hz
                            altitude_baseline_m = float(np.median(alt_window)) if alt_window else 0.0

                            print(f"\n[STAGE 1 TRIGGER] Fall candidate (ML Prob: {fall_prob:.2f}, "
                                  f"{consecutive_triggers} consecutive windows). Verifying...")
                            stillness_buffer = []
                            altitude_deque.clear()
                            altitude_deque.append(altitude_baseline_m)
                            state = STATE_VERIFYING
                            verify_tick = 0
                            consecutive_triggers = 0
                        else:
                            now = time.time()
                            if now - last_print >= 1.0:
                                print(f"Status: Normal (ML Fall Prob: {fall_prob:.2f})   ", end="\r")
                                last_print = now

            # ------------------------------------------------- VERIFYING
            elif state == STATE_VERIFYING:
                stillness_buffer.append(sample)
                verify_tick += 1
                if verify_tick % ALT_SAMPLE_RATE_DIVISOR == 0:
                    try:
                        p_raw, _ = baro.read_raw()
                        p_filtered = kalman.update(p_raw)
                        # NOTE: baseline (reference_hpa) is NOT updated here —
                        # frozen during verification so a real fall can't drift away.
                        alt_m = altitude_from_pressure(p_filtered, reference_hpa)
                        altitude_deque.append(alt_m)
                    except Exception:
                        pass

                if len(stillness_buffer) >= STILLNESS_SAMPLES:
                    stillness_df = buffer_to_df(stillness_buffer, TARGET_HZ)
                    stillness_res = check_stillness(stillness_df, accel_std_threshold=0.15,
                                                     gyro_std_threshold=15.0)

                    alt_samples = list(altitude_deque)
                    altitude_after_m = float(np.median(alt_samples[-20:])) if alt_samples else 0.0
                    altitude_res = check_altitude_drop(altitude_baseline_m, altitude_after_m,
                                                        drop_threshold_m=ALT_DROP_THRESHOLD_M)

                    fuse_result = fuse_decision(ml_prob=fall_prob, is_still=stillness_res["is_still"],
                                                 altitude_score=altitude_res["altitude_score"])

                    print(f"[FUSION] ML={fuse_result['ml_component']:.2f} "
                          f"Stillness={fuse_result['stillness_component']:.1f} "
                          f"Altitude={fuse_result['altitude_component']:.2f} "
                          f"-> Fused={fuse_result['fused_score']:.2f} (threshold {FUSION_THRESHOLD}) "
                          f"=> {'CONFIRMED' if fuse_result['confirm_fall'] else 'SUPPRESSED'}")

                    if fuse_result["confirm_fall"]:
                        log_event("FALL_CONFIRMED",
                                   f"fused={fuse_result['fused_score']:.2f} ml={fall_prob:.2f} "
                                   f"still={stillness_res['is_still']} drop_m={altitude_res['drop_m']:.2f}")
                        state = STATE_WARNING_COUNTDOWN
                        countdown_start = time.time()
                        hw.start_warning_beep(interval_s=0.5)
                    else:
                        log_event("FALSE_ALARM_SUPPRESSED",
                                   f"fused={fuse_result['fused_score']:.2f} ml={fall_prob:.2f} "
                                   f"still={stillness_res['is_still']} drop_m={altitude_res['drop_m']:.2f}")
                        state = STATE_MONITORING
                        buffer.clear()
                        altitude_deque.clear()
                        stillness_buffer = []

            # ------------------------------------------------- WARNING COUNTDOWN
            elif state == STATE_WARNING_COUNTDOWN:
                elapsed = time.time() - countdown_start
                remaining = int(WARNING_COUNTDOWN_S - elapsed)
                now = time.time()
                if now - last_print >= 1.0:
                    print(f"\r  WARNING: FALL DETECTED! Alarm in {remaining}s. Press Cancel! ", end="\r")
                    last_print = now

                if hw.is_button_pressed():
                    print("\n[CANCELLED] Alarm dismissed by user.")
                    log_event("CANCELLED_BY_USER", "warning countdown")
                    hw.stop_sound()
                    state = STATE_MONITORING
                    buffer.clear()
                    time.sleep(1.0)
                elif remaining <= 0:
                    print("\n[FINAL ALARM] No response in 30 seconds!")
                    log_event("FINAL_ALARM", "no response in 30s")
                    hw.start_alarm_sound()
                    state = STATE_ALARM_ACTIVE

            # ------------------------------------------------- ALARM ACTIVE
            elif state == STATE_ALARM_ACTIVE:
                if hw.is_button_pressed():
                    print("\n[RESET] Alarm stopped by user.")
                    log_event("ALARM_RESET_BY_USER", "")
                    hw.stop_sound()
                    state = STATE_MONITORING
                    buffer.clear()

            next_t += interval
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        hw.cleanup()
        print("Hardware cleaned up. System exited safely.")


if __name__ == "__main__":
    main()
