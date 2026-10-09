"""Train and evaluate a student outcome classifier.

Run after data_prep.py with: python ml_model.py
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import CATEGORICAL_COLUMNS, TARGET


BASE_DIR = Path(__file__).resolve().parent


def train_and_evaluate():
    train_df = pd.read_csv(BASE_DIR / "student_train.csv")
    test_df = pd.read_csv(BASE_DIR / "student_test.csv")
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET].astype(str)
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET].astype(str)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in X_train.columns]
    numeric = [column for column in X_train.columns if column not in categorical]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=80,
                    min_samples_leaf=5,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    classes = model.named_steps["classifier"].classes_
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    prep = model.named_steps["preprocessor"]
    feature_names = prep.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(BASE_DIR / "feature_importance.csv", index=False)
    joblib.dump(model, BASE_DIR / "student_outcome_pipeline.pkl", compress=3)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "classes": [str(label) for label in classes],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "test_rows": int(len(y_test)),
    }
    (BASE_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Pelatihan model selesai.")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print("Dibuat: student_outcome_pipeline.pkl, model_metrics.json, feature_importance.csv")


if __name__ == "__main__":
    train_and_evaluate()
