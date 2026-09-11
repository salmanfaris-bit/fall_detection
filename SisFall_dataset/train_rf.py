"""
Trains a Random Forest fall detector on the feature table produced by
build_dataset.py, using a SUBJECT-LEVEL train/test split (entire subjects
held out, never individual trials) to avoid leakage.

Usage:
    python train_rf.py features.csv
"""

import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

META_COLS = ["activity_code", "subject_id", "trial", "is_fall", "label", "age_group"]


def main(csv_path: str, model_out: str = "fall_rf.joblib", test_size: float = 0.25, random_state: int = 42):
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()

    # subject-level split -- GroupShuffleSplit keeps every trial from a given
    # subject entirely in train OR entirely in test, never split across both
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Test subjects  ({len(test_subjects)}): {sorted(test_subjects)}")
    print(f"Train trials: {len(X_train)}, Test trials: {len(X_test)}")
    print(f"Train class balance: {np.bincount(y_train)}")
    print(f"Test class balance:  {np.bincount(y_test)}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Test set results (held-out subjects) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["ADL", "Fall"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(10))

    joblib.dump({"model": clf, "feature_cols": feature_cols}, model_out)
    print(f"\nModel saved to {model_out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_rf.py <features_csv> [model_out.joblib]")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_out = sys.argv[2] if len(sys.argv) > 2 else "fall_rf.joblib"
    main(csv_path, model_out)
