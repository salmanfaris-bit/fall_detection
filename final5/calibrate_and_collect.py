"""
calibrate_and_collect.py — Two field utilities in one CLI:

  python calibrate_and_collect.py noise
      One-time barometer noise measurement (sensor stationary). Prints the
      variance to use as KalmanFilter1D(r=...) in sensors.py.

  python calibrate_and_collect.py collect
      Record a labeled fall/ADL trial for future model retraining. Refuses
      to silently save mock data as if it were real hardware data.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from sensors import BNO055Reader, MockBNO055Reader, BMP280Reader, MockBMP280Reader

DEFAULT_OUTPUT_DIR = "recorded_trials"
DEFAULT_DURATION_S = 10.0


def measure_barometer_noise(n_samples: int = 200):
    print(f"Measuring static barometer noise over {n_samples} samples (keep sensor still)...")
    try:
        baro = BMP280Reader(address=0x76)
        baro.configure()
    except Exception as e:
        print(f"[ERROR] Real BMP280 not available ({e}). Noise measurement requires real hardware.")
        return

    samples = []
    for _ in range(n_samples):
        p, _ = baro.read_raw()
        samples.append(p)
        time.sleep(0.05)

    variance = float(np.var(samples))
    print(f"\nMeasured pressure variance (r): {variance:.6f} hPa^2")
    print(f"Update sensors.py: KalmanFilter1D(initial_value=..., r={variance:.6f})")


def _make_readers():
    """Real hardware first, mock fallback second, tagged accordingly."""
    try:
        imu = BNO055Reader(address=0x28)
        imu.configure()
        imu_is_real = True
    except Exception as e:
        print(f"[WARNING] Could not connect to real BNO055 hardware: {e}")
        imu = MockBNO055Reader()
        imu.configure()
        imu_is_real = False

    try:
        baro = BMP280Reader(address=0x76)
        baro.configure()
        baro_is_real = True
    except Exception as e:
        print(f"[WARNING] Could not connect to real BMP280 hardware: {e}")
        baro = MockBMP280Reader()
        baro.configure()
        baro_is_real = False

    return imu, imu_is_real, baro, baro_is_real


def collect_trial(duration_s: float, label: str, trial_num: int, target_hz: int = 200):
    imu, imu_is_real, baro, baro_is_real = _make_readers()

    if not imu_is_real:
        print("\n" + "!" * 60)
        print("WARNING: No real BNO055 hardware detected.")
        print("Mock data is for pipeline-logic testing only, NOT calibration.")
        print("!" * 60)
        response = input("Proceed and save a MARKED-MOCK trial anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborting — no file saved.")
            return None
        print("User confirmed: saving mock-simulated data with explicit marker.")
    else:
        print(f"\nRecording trial #{trial_num}: {label} for {duration_s}s. Starting in 3...")
        time.sleep(3)

    interval = 1.0 / target_hz
    n_samples = int(duration_s * target_hz)
    rows = []
    next_t = time.time()

    for i in range(n_samples):
        accel_g, gyro_dps = imu.read_raw()
        try:
            p, _ = baro.read_raw()
            alt_m = None if not baro_is_real and not hasattr(baro, "altitude_from_pressure") else p
        except Exception:
            alt_m = None
        rows.append({
            "t": i / target_hz,
            "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
            "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
            "pressure_hpa": alt_m,
        })
        next_t += interval
        sleep_time = next_t - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    df = pd.DataFrame(rows)
    df["label"] = label
    df["trial_num"] = trial_num
    df["source"] = "real_hardware" if imu_is_real else "mock_simulation"

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    marker = "real" if imu_is_real else "mock"
    filename = os.path.join(DEFAULT_OUTPUT_DIR, f"trial_{trial_num:03d}_{label.replace(' ', '_')}_{marker}.csv")
    df.to_csv(filename, index=False)

    accel_mag = np.sqrt(df["accel_x_g"] ** 2 + df["accel_y_g"] ** 2 + df["accel_z_g"] ** 2).mean()
    print(f"Saved {len(df)} samples to {filename}")
    print(f"  Accel magnitude at rest: {accel_mag:.3f}g | Source: {'REAL' if imu_is_real else 'MOCK'}")
    if not imu_is_real:
        print("  NOTE: mock_simulation data — do NOT use for threshold calibration.")
    return filename


def collect_interactive():
    label = input("Enter label (e.g., 'fall', 'walk', 'sit'): ").strip() or "unknown"
    try:
        trial_num = int(input("Enter trial number: ").strip())
    except ValueError:
        trial_num = 0
    try:
        duration = float(input(f"Duration in seconds (default {DEFAULT_DURATION_S}): ").strip() or DEFAULT_DURATION_S)
    except ValueError:
        duration = DEFAULT_DURATION_S
    collect_trial(duration_s=duration, label=label, trial_num=trial_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barometer calibration + labeled trial collection")
    parser.add_argument("mode", choices=["noise", "collect"], help="'noise' to measure barometer noise, 'collect' to record a labeled trial")
    args = parser.parse_args()

    if args.mode == "noise":
        measure_barometer_noise()
    else:
        collect_interactive()
