"""
debug_validate_window.py — Drop-in diagnostic for the "Corrupted sensor
window skipped" message. Instead of guessing, this prints EXACTLY why each
failing window was rejected: which column, how many bad samples, and the
actual min/max values seen.

Usage options:
  1) Live, on your actual sensors (real Pi or mock fallback — same as main.py):
       python3 debug_validate_window.py --live --seconds 10

  2) Against a CSV you already recorded (e.g. from calibrate_and_collect.py
     or the scenario CSVs from earlier):
       python3 debug_validate_window.py --csv scenario_csvs/forward_trip.csv
"""

import argparse
import numpy as np
import pandas as pd

TARGET_HZ = 200
WINDOW_N = int(2.0 * TARGET_HZ)
STEP_N = int(0.5 * TARGET_HZ)

ACCEL_COLS = ["accel_x_g", "accel_y_g", "accel_z_g"]
GYRO_COLS = ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]
ACCEL_LIMIT_G = 20.0
GYRO_LIMIT_DPS = 2200.0


def explain_window(df: pd.DataFrame, window_idx: int):
    cols = ACCEL_COLS + GYRO_COLS
    values = df[cols].to_numpy()

    if not np.all(np.isfinite(values)):
        bad_mask = ~np.isfinite(values)
        bad_cols = [cols[j] for j in range(len(cols)) if bad_mask[:, j].any()]
        n_bad = int(bad_mask.sum())
        print(f"  Window @ idx {window_idx}: REJECTED — {n_bad} NaN/Inf value(s) in {bad_cols}")
        return False

    accel_vals = df[ACCEL_COLS].to_numpy()
    if np.any(np.abs(accel_vals) > ACCEL_LIMIT_G):
        worst_col = ACCEL_COLS[np.argmax(np.abs(accel_vals).max(axis=0))]
        worst_val = accel_vals[np.abs(accel_vals) > ACCEL_LIMIT_G]
        print(f"  Window @ idx {window_idx}: REJECTED — accel exceeds +-{ACCEL_LIMIT_G}g "
              f"(worst column: {worst_col}, offending values: {np.round(worst_val[:5], 2)}"
              f"{'...' if len(worst_val) > 5 else ''}, count={len(worst_val)})")
        return False

    gyro_vals = df[GYRO_COLS].to_numpy()
    if np.any(np.abs(gyro_vals) > GYRO_LIMIT_DPS):
        worst_col = GYRO_COLS[np.argmax(np.abs(gyro_vals).max(axis=0))]
        worst_val = gyro_vals[np.abs(gyro_vals) > GYRO_LIMIT_DPS]
        print(f"  Window @ idx {window_idx}: REJECTED — gyro exceeds +-{GYRO_LIMIT_DPS} dps "
              f"(worst column: {worst_col}, offending values: {np.round(worst_val[:5], 1)}"
              f"{'...' if len(worst_val) > 5 else ''}, count={len(worst_val)})")
        return False

    return True


def run_on_dataframe(df: pd.DataFrame):
    n = len(df)
    if n < WINDOW_N:
        print(f"Only {n} samples provided, need at least {WINDOW_N} (2s @ {TARGET_HZ}Hz). "
              f"Showing whole thing as one window.")
        explain_window(df, 0)
        return

    idx = WINDOW_N
    total, rejected = 0, 0
    while idx <= n:
        window = df.iloc[idx - WINDOW_N: idx]
        total += 1
        ok = explain_window(window, idx)
        if not ok:
            rejected += 1
        idx += STEP_N

    print(f"\n{rejected}/{total} windows rejected.")
    if rejected == total and total > 0:
        print("EVERY window failed — this points to a systematic problem (wrong scale factor, "
              "bad wiring/address, or sensor not fully initialized) rather than occasional glitches.")
    elif rejected > 0:
        print("Some windows failed — likely occasional I2C glitches. If this happens rarely, "
              "it's the validator doing its job; main.py just moves on to the next window.")


def run_live(seconds: float):
    from sensors import BNO055Reader, MockBNO055Reader
    import time

    imu = None
    try:
        imu = BNO055Reader(address=0x28)
        imu.configure()
        print("Using REAL BNO055 over I2C.\n")
    except Exception as e:
        print(f"Real BNO055 unavailable ({e}) — using MockBNO055Reader.\n")
        imu = MockBNO055Reader()
        imu.configure()

    n_samples = int(seconds * TARGET_HZ)
    rows = []
    interval = 1.0 / TARGET_HZ
    next_t = time.time()
    for i in range(n_samples):
        accel_g, gyro_dps = imu.read_raw()
        rows.append({
            "accel_x_g": accel_g[0], "accel_y_g": accel_g[1], "accel_z_g": accel_g[2],
            "gyro_x_dps": gyro_dps[0], "gyro_y_dps": gyro_dps[1], "gyro_z_dps": gyro_dps[2],
        })
        next_t += interval
        sleep_time = next_t - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    df = pd.DataFrame(rows)
    df["t"] = np.arange(len(df)) / TARGET_HZ
    print(f"Captured {len(df)} live samples ({seconds:.1f}s). Raw range check:")
    for c in ACCEL_COLS + GYRO_COLS:
        print(f"  {c:<14} min={df[c].min():8.2f}  max={df[c].max():8.2f}")

    if hasattr(imu, "glitch_stats"):
        stats = imu.glitch_stats()
        print(f"\nDriver-level sample-and-hold stats: {stats['glitches_total']}/{stats['reads_total']} "
              f"reads needed holding ({stats['glitch_rate']*100:.1f}%).")

    print()
    run_on_dataframe(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to a recorded CSV (t, accel_*, gyro_* columns)")
    parser.add_argument("--live", action="store_true", help="Read live from sensors.py (real or mock)")
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration for --live mode")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        run_on_dataframe(df)
    elif args.live:
        run_live(args.seconds)
    else:
        parser.error("Provide either --csv PATH or --live")
