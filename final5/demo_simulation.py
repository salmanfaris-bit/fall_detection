"""
demo_simulation.py — End-to-end simulation using mock sensors. No hardware
required. Demonstrates the full state machine: MONITORING -> VERIFYING ->
WARNING_COUNTDOWN -> (cancelled | ALARM_ACTIVE).
"""

import time
import numpy as np
import pandas as pd

from sensors import (
    MockBNO055Reader, MockBMP280Reader,
    KalmanFilter1D, DriftingBaseline, altitude_from_pressure,
)
from detection import check_stillness, check_altitude_drop, fuse_decision, FUSION_THRESHOLD
from alerts import HardwareController

TARGET_HZ = 200
STEP_SECONDS = 0.5
STILLNESS_WINDOW_S = 2.0
STILLNESS_SAMPLES = int(STILLNESS_WINDOW_S * TARGET_HZ)
STEP_SAMPLES = int(STEP_SECONDS * TARGET_HZ)


def run_scenario(name: str, auto_cancel: bool):
    print(f"\n{'=' * 60}\nSCENARIO: {name}\n{'=' * 60}")

    shared_counter = [0]
    imu = MockBNO055Reader()
    baro = MockBMP280Reader()
    imu.set_counter(shared_counter)
    baro.set_counter(shared_counter)
    imu.configure()
    baro.configure()

    kalman = KalmanFilter1D(initial_value=1013.0)
    baseline = DriftingBaseline(initial_value=1013.0, tau_s=300.0, sample_dt_s=0.05)
    hw = HardwareController(buzzer_pin=18, button_pin=23)

    state = "MONITORING"
    buffer, altitude_deque = [], []
    fall_prob, altitude_baseline_m = 0.0, 0.0
    countdown_start = None

    try:
        while True:
            shared_counter[0] += 1
            accel_g, gyro_dps = imu.read_raw()
            p_raw, _ = baro.read_raw()

            if state == "MONITORING":
                if shared_counter[0] % 10 == 0:
                    p_f = kalman.update(p_raw)
                    ref = baseline.update(p_f)
                    altitude_deque.append(altitude_from_pressure(p_f, ref))
                if shared_counter[0] > 25 * TARGET_HZ and fall_prob == 0.0:
                    fall_prob = 0.85  # simulated ML trigger
                    altitude_baseline_m = float(np.median(altitude_deque[-20:])) if altitude_deque else 0.0
                    print(f"\n[STAGE 1] Candidate fall (ML Prob: {fall_prob:.2f}). Verifying...")
                    buffer, altitude_deque = [], [altitude_baseline_m]
                    state = "VERIFYING"

            elif state == "VERIFYING":
                buffer.append({
                    "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
                    "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
                })
                if shared_counter[0] % 10 == 0:
                    p_f = kalman.update(p_raw)
                    ref = baseline.value  # frozen during verification
                    altitude_deque.append(altitude_from_pressure(p_f, ref))

                if len(buffer) >= STILLNESS_SAMPLES:
                    df = pd.DataFrame(buffer)
                    df["t"] = np.arange(len(df)) / TARGET_HZ
                    still_res = check_stillness(df, accel_std_threshold=0.15, gyro_std_threshold=15.0)
                    alt_after = float(np.median(altitude_deque[-20:])) if altitude_deque else 0.0
                    alt_res = check_altitude_drop(altitude_baseline_m, alt_after, drop_threshold_m=0.3)
                    fuse = fuse_decision(ml_prob=fall_prob, is_still=still_res["is_still"],
                                          altitude_score=alt_res["altitude_score"])

                    print(f"[FUSION] ML={fuse['ml_component']:.2f} Stillness={fuse['stillness_component']:.1f} "
                          f"Altitude={fuse['altitude_component']:.2f} -> Fused={fuse['fused_score']:.2f} "
                          f"(threshold {FUSION_THRESHOLD})")

                    if fuse["confirm_fall"]:
                        print("  -> CONFIRMED: entering WARNING_COUNTDOWN")
                        state = "WARNING_COUNTDOWN"
                        countdown_start = time.time()
                        hw.start_warning_beep(interval_s=0.5)
                    else:
                        print("  -> SUPPRESSED: false alarm, returning to monitoring")
                        state, fall_prob = "MONITORING", 0.0

            elif state == "WARNING_COUNTDOWN":
                elapsed = time.time() - countdown_start
                remaining = 30.0 - elapsed
                if auto_cancel and elapsed > 1.0:
                    hw.simulate_button_press()
                if hw.is_button_pressed():
                    print("  [CANCELLED] User pressed the cancel button.")
                    hw.stop_sound()
                    state = "MONITORING"
                    break
                if remaining <= 0:
                    print("  [FINAL ALARM] No response within 30s.")
                    hw.start_alarm_sound()
                    state = "ALARM_ACTIVE"
                    break

            time.sleep(0.0005)  # accelerated for demo purposes
    finally:
        hw.cleanup()

    print(f"Scenario '{name}' finished in state: {state}")
    return state


if __name__ == "__main__":
    state_a = run_scenario("Fall detected, user cancels in time", auto_cancel=True)
    state_b = run_scenario("Fall detected, user does NOT respond", auto_cancel=False)
    print(f"\nSummary: cancel-scenario -> {state_a} | no-response-scenario -> {state_b}")
