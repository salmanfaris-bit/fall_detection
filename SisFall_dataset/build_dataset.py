"""
Builds the full per-trial feature table from the SisFall dataset and saves it
as a CSV. Run this once; everything downstream (training, evaluation) reads
from the CSV instead of re-parsing 4510 raw text files every time.

Usage:
    python build_dataset.py "F:\\amigosia\\SisFall_dataset" features.csv
"""

import sys
import pandas as pd
from sisfall_loader import walk_dataset
from feature_extraction import build_feature_row


def main(dataset_root: str, out_csv: str, target_hz: float = None):
    rows = []
    n_ok, n_fail = 0, 0

    for meta, df in walk_dataset(dataset_root, target_hz=target_hz):
        try:
            rows.append(build_feature_row(meta, df))
            n_ok += 1
        except Exception as e:
            print(f"[WARN] feature extraction failed for {meta}: {e}")
            n_fail += 1

        if n_ok % 500 == 0 and n_ok > 0:
            print(f"...{n_ok} trials processed")

    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    print(f"Done. {n_ok} trials OK, {n_fail} failed. Saved to {out_csv}")
    print(f"Class balance: {table['label'].value_counts().to_dict()}")
    print(f"Subjects: {table['subject_id'].nunique()}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_dataset.py <dataset_root> <out_csv> [target_hz]")
        sys.exit(1)

    dataset_root = sys.argv[1]
    out_csv = sys.argv[2]
    target_hz = float(sys.argv[3]) if len(sys.argv) > 3 else None

    main(dataset_root, out_csv, target_hz)
