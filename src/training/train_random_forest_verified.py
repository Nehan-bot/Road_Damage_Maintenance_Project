
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

REVIEW_FILE = (
    ROOT / "data/processed/priority/final_review_sample_50_assessed.csv"
)

FEATURE_FILE = (
    ROOT / "data/processed/vision/yolo_train_image_features.csv"
)

OUTPUT_DIR = ROOT / "experiments/random_forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "random_forest_priority.joblib"
METRICS_FILE = OUTPUT_DIR / "random_forest_metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "random_forest_test_predictions.csv"

TARGET = "independent_human_verified_label"

FEATURES = [
    "total_detections",
    "longitudinal crack_count",
    "longitudinal crack_mean_confidence",
    "transverse crack_count",
    "transverse crack_mean_confidence",
    "alligator crack_count",
    "alligator crack_mean_confidence",
    "pothole_count",
    "pothole_mean_confidence",
]

ALLOWED_LABELS = ["Low", "Medium", "High"]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading review and feature datasets...")

review = pd.read_csv(REVIEW_FILE, dtype={TARGET: "string"})
features = pd.read_csv(FEATURE_FILE)

review.columns = review.columns.str.strip()
features.columns = features.columns.str.strip()

review["image_name"] = review["image_name"].astype(str).str.strip()
features["image_name"] = features["image_name"].astype(str).str.strip()

review[TARGET] = review[TARGET].fillna("").str.strip()

print("Review images:", len(review))
print("Feature images:", len(features))

# --------------------------------------------------
# VALIDATE LABELS
# --------------------------------------------------

valid_mask = review[TARGET].isin(ALLOWED_LABELS)

invalid = set(
    review.loc[
        (review[TARGET] != "") & (~valid_mask),
        TARGET
    ].dropna()
)

if invalid:
    raise ValueError(f"Invalid labels found: {invalid}")

review = review[valid_mask].copy()

print("\nCLASS DISTRIBUTION")
print(review[TARGET].value_counts())

if review[TARGET].nunique() != 3:
    raise ValueError("All three priority classes are required.")

if review[TARGET].value_counts().min() < 5:
    raise ValueError("Each class requires at least 5 verified labels.")

# --------------------------------------------------
# MERGE FEATURES AND LABELS
# --------------------------------------------------

data = review[["image_name", TARGET]].merge(
    features[["image_name"] + FEATURES],
    on="image_name",
    how="inner",
    validate="one_to_one"
)

if len(data) != len(review):
    missing = set(review["image_name"]) - set(data["image_name"])
    raise ValueError(f"Missing image features for: {missing}")

X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
y = data[TARGET]

if X.isna().all(axis=0).any():
    raise ValueError("At least one feature column contains no numeric values.")

# --------------------------------------------------
# SAME SPLIT AS DECISION TREE
# --------------------------------------------------

X_train, X_test, y_train, y_test, train_names, test_names = (
    train_test_split(
        X,
        y,
        data["image_name"],
        test_size=0.25,
        random_state=42,
        stratify=y
    )
)

print("\nDATA SPLIT")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# TRAIN RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest...")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=ALLOWED_LABELS,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=ALLOWED_LABELS
)

print("\nRANDOM FOREST RESULTS")
print("=====================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=ALLOWED_LABELS,
    zero_division=0
))

print("\nConfusion Matrix:")
print(matrix)

# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

metrics = {
    "model": "Random Forest",
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "class_distribution": {
        str(k): int(v)
        for k, v in review[TARGET].value_counts().items()
    },
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "confusion_matrix_labels": ALLOWED_LABELS,
    "features": FEATURES,
    "random_state": 42,
    "test_size": 0.25,
    "limitation": (
        "Exploratory evaluation on a small manually labelled sample. "
        "YOLO training features are in-sample predictions."
    )
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

prediction_df = pd.DataFrame({
    "image_name": test_names.to_numpy(),
    "actual_priority": y_test.to_numpy(),
    "predicted_priority": predictions
})

prediction_df.to_csv(PREDICTIONS_FILE, index=False)

print("\nRANDOM FOREST PIPELINE COMPLETED")
print("================================")
print("Model:", MODEL_FILE)
print("Metrics:", METRICS_FILE)
print("Predictions:", PREDICTIONS_FILE)
