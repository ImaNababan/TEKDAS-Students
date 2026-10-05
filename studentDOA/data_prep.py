"""Clean student outcome data and create stratified train/test files.

Run from this directory with: python data_prep.py
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "student_doa.csv"
TARGET = "Target"

# These columns contain numeric codes for categories, not measured quantities.
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def prepare_data():
    df = pd.read_csv(SOURCE_FILE)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")
    missing_columns = set(CATEGORICAL_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Kolom kategori tidak ditemukan: {sorted(missing_columns)}")

    original_rows = len(df)
    missing_before = df.isna().sum().to_dict()
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df.dropna(subset=[TARGET])
    df = df.loc[df[TARGET].ne("")].copy()

    feature_columns = [column for column in df.columns if column != TARGET]
    numeric_columns = [column for column in feature_columns if column not in CATEGORICAL_COLUMNS]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("string")

    X = df[feature_columns]
    y = df[TARGET]
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    train_df = train_X.copy()
    train_df[TARGET] = train_y
    test_df = test_X.copy()
    test_df[TARGET] = test_y

    df.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)
    train_df.to_csv(BASE_DIR / "student_train.csv", index=False)
    test_df.to_csv(BASE_DIR / "student_test.csv", index=False)

    report = {
        "source_file": SOURCE_FILE.name,
        "original_rows": int(original_rows),
        "duplicate_rows_removed": duplicates_removed,
        "prepared_rows": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "categorical_features": CATEGORICAL_COLUMNS,
        "numeric_features": numeric_columns,
        "missing_values_before_cleaning": {
            key: int(value) for key, value in missing_before.items() if value
        },
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target_distribution": {
            str(key): int(value) for key, value in y.value_counts().sort_index().items()
        },
    }
    (BASE_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Data preparation selesai.")
    print(f"Train / test: {len(train_df):,} / {len(test_df):,}")
    print("Dibuat: student_ml_dataset.csv, student_train.csv, student_test.csv")
    print("Dibuat: data_quality_report.json")


if __name__ == "__main__":
    prepare_data()
