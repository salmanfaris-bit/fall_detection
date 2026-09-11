"""
SisFall dataset loader and unit converter.

Handles:
  - filename parsing (<CODE>_<SUBJECT>_<TRIAL>.txt)
  - raw bit -> physical unit conversion (ADXL345 accel in g, ITG3200 gyro in deg/s)
  - dropping the MMA8451Q columns (not needed for BNO055 matching)
  - resampling from SisFall's native 200 Hz to a target rate

Usage:
    from sisfall_loader import load_file, walk_dataset

    df = load_file("SisFall_dataset/SA01/F05_SA01_R04.txt")
    for meta, df in walk_dataset("SisFall_dataset"):
        ...
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

# ---- Sensor constants from the SisFall readme ----
ADXL345_RESOLUTION_BITS = 13
ADXL345_RANGE_G = 16          # +-16g

ITG3200_RESOLUTION_BITS = 16
ITG3200_RANGE_DPS = 2000      # +-2000 deg/s

SISFALL_NATIVE_HZ = 200

FILENAME_RE = re.compile(r"^([A-Z]\d{2})_([A-Z]{2}\d{2})_R(\d{2})\.txt$", re.IGNORECASE)


def _bits_to_physical(raw: np.ndarray, resolution_bits: int, full_scale_range) -> np.ndarray:
    """
    SisFall's documented conversion:
        value_physical = [(2 * Range) / (2 ** Resolution)] * raw_bits
    Works for both acceleration (g) and angular velocity (deg/s) --
    same formula, different Range/Resolution constants.
    """
    scale = (2.0 * full_scale_range) / (2 ** resolution_bits)
    return raw.astype(np.float64) * scale


def parse_filename(filename: str) -> dict:
    """
    Parses e.g. 'F05_SA01_R04.txt' ->
        {'activity_code': 'F05', 'subject_id': 'SA01', 'trial': 4,
         'is_fall': True, 'age_group': 'adult'}
    """
    base = os.path.basename(filename)
    m = FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Filename doesn't match SisFall pattern: {base}")

    activity_code, subject_id, trial = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
    is_fall = activity_code.startswith("F")
    age_group = "adult" if subject_id.startswith("SA") else "elderly"

    return {
        "activity_code": activity_code,
        "subject_id": subject_id,
        "trial": trial,
        "is_fall": is_fall,
        "label": 1 if is_fall else 0,
        "age_group": age_group,
    }


def load_file(filepath: str, target_hz: float = None) -> pd.DataFrame:
    """
    Loads a single SisFall .txt file, converts to physical units,
    drops the MMA8451Q columns, optionally resamples.

    Returns a DataFrame with columns:
        t, accel_x_g, accel_y_g, accel_z_g, gyro_x_dps, gyro_y_dps, gyro_z_dps
    plus metadata columns (subject_id, activity_code, label, age_group) broadcast
    to every row for easy downstream filtering/grouping.
    """
    # Rows look like: "17,-179,-99,-18,-504,-352, 76,-697,-279;"
    # comma-separated, semicolon-terminated, no header.
    raw = pd.read_csv(
        filepath,
        header=None,
        sep=",",
        engine="python",
        skip_blank_lines=True,
    )

    # last column has trailing ';' stuck to the number -> strip and convert
    last_col = raw.columns[-1]
    raw[last_col] = (
        raw[last_col].astype(str).str.replace(";", "", regex=False).astype(float)
    )
    raw = raw.astype(float)

    if raw.shape[1] != 9:
        raise ValueError(f"Expected 9 columns, got {raw.shape[1]} in {filepath}")

    accel_adxl_raw = raw.iloc[:, 0:3].to_numpy()
    gyro_raw = raw.iloc[:, 3:6].to_numpy()
    # columns 6:9 are MMA8451Q -- intentionally dropped

    accel_g = _bits_to_physical(accel_adxl_raw, ADXL345_RESOLUTION_BITS, ADXL345_RANGE_G)
    gyro_dps = _bits_to_physical(gyro_raw, ITG3200_RESOLUTION_BITS, ITG3200_RANGE_DPS)

    n = len(raw)
    t = np.arange(n) / SISFALL_NATIVE_HZ

    df = pd.DataFrame({
        "t": t,
        "accel_x_g": accel_g[:, 0],
        "accel_y_g": accel_g[:, 1],
        "accel_z_g": accel_g[:, 2],
        "gyro_x_dps": gyro_dps[:, 0],
        "gyro_y_dps": gyro_dps[:, 1],
        "gyro_z_dps": gyro_dps[:, 2],
    })

    if target_hz is not None and target_hz != SISFALL_NATIVE_HZ:
        df = resample_dataframe(df, SISFALL_NATIVE_HZ, target_hz)

    meta = parse_filename(filepath)
    for k, v in meta.items():
        df[k] = v

    return df


def resample_dataframe(df: pd.DataFrame, orig_hz: float, target_hz: float) -> pd.DataFrame:
    """
    Polyphase resampling (scipy.signal.resample_poly) on the sensor columns only.
    Use this to align SisFall's 200 Hz to whatever rate you run BNO055 at.
    """
    from fractions import Fraction

    frac = Fraction(target_hz / orig_hz).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    sensor_cols = ["accel_x_g", "accel_y_g", "accel_z_g",
                   "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]

    resampled = {
        col: resample_poly(df[col].to_numpy(), up, down) for col in sensor_cols
    }
    n_new = len(next(iter(resampled.values())))
    t_new = np.arange(n_new) / target_hz

    out = pd.DataFrame({"t": t_new, **resampled})
    return out


def walk_dataset(dataset_root: str, target_hz: float = None, subjects=None, activities=None):
    """
    Generator over every file in the SisFall dataset tree.
    dataset_root should point at the folder containing SA01/, SA02/, ... SE15/.

    subjects: optional iterable of subject_ids to include, e.g. {'SA01','SA02'}
    activities: optional iterable of activity codes to include, e.g. {'F01','F02','D01'}

    Yields (meta_dict, dataframe) per file.
    """
    for subject_folder in sorted(os.listdir(dataset_root)):
        subj_path = os.path.join(dataset_root, subject_folder)
        if not os.path.isdir(subj_path):
            continue
        if subjects is not None and subject_folder.upper() not in subjects:
            continue

        for fname in sorted(os.listdir(subj_path)):
            if not fname.lower().endswith(".txt"):
                continue
            try:
                meta = parse_filename(fname)
            except ValueError:
                continue  # skip README.txt etc if present in folder

            if activities is not None and meta["activity_code"] not in activities:
                continue

            fpath = os.path.join(subj_path, fname)
            try:
                df = load_file(fpath, target_hz=target_hz)
            except Exception as e:
                print(f"[WARN] failed to load {fpath}: {e}")
                continue

            yield meta, df


if __name__ == "__main__":
    # quick smoke test - point this at your actual dataset folder before running
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "SisFall_dataset"
    count = 0
    for meta, df in walk_dataset(root):
        count += 1
        if count <= 3:
            print(meta, "-> shape", df.shape, "accel range g:",
                  df["accel_x_g"].min(), df["accel_x_g"].max())
        if count >= 20:
            break
    print(f"Loaded {count} files OK (smoke test capped at 20).")
