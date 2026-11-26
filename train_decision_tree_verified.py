
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# 1. PROJECT PATHS
# =====================================================

REVIEW_FILE = Path(
    "data/processed/priority/final_review_sample_50_assessed.csv"
)

FEATURE_FILE = Path(
    "data/processed/vision/yolo_train_image_features.csv"
)

OUTPUT_DIR = Path("experiments/decision_tree")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "independent_human_verified_label"

ALLOWED = {"Low", "Medium", "High"}

# =====================================================
# 2. LOAD DATASETS
# =====================================================

print("Loading review and feature datasets...")

if not REVIEW_FILE.exists():
    raise FileNotFoundError(REVIEW_FILE)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(FEATURE_FILE)

review = pd.read_csv(
    REVIEW_FILE,
    dtype={TARGET: "string"}
)

features = pd.read_csv(FEATURE_FILE)

# Handle missing target values safely
review[TARGET] = (
    review[TARGET]
    .fillna("")
    .str.strip()
)

print("Review images:", len(review))
print("Feature images:", len(features))

# =====================================================
# 3. VALIDATE DATA
# =====================================================

required_columns = {"image_name", TARGET}

missing_columns = required_columns - set(review.columns)

if missing_columns:
    raise ValueError(
        f"Missing review columns: {missing_columns}"
    )

if review["image_name"].duplicated().any():
    raise ValueError(
        "Duplicate image names in review CSV."
    )

if features["image_name"].duplicated().any():
    raise ValueError(
        "Duplicate image names in feature CSV."
    )

if not set(review["image_name"]).issubset(
    set(features["image_name"])
):
    raise ValueError(
        "Some review images are missing from training features."
    )

# =====================================================
# 4. VALIDATE INDEPENDENT HUMAN LABELS
# =====================================================

valid_mask = review[TARGET].isin(ALLOWED)

invalid = set(
    review.loc[
        (review[TARGET] != "") & (~valid_mask),
        TARGET
    ].dropna().tolist()
)

if invalid:
    raise ValueError(
        f"Invalid labels found: {invalid}. "
        "Allowed values: Low, Medium, High."
    )

verified = review[
    review[TARGET].isin(ALLOWED)
].copy()

pending = review[
    ~review[TARGET].isin(ALLOWED)
]

print("\nLABEL VALIDATION")
print("----------------")
print("Total review images:", len(review))
print("Independently verified:", len(verified))
print("Pending:", len(pending))

# =====================================================
# 5. REQUIRE SUFFICIENT VERIFIED DATA
# =====================================================

MIN_VERIFIED = 30
MIN_PER_CLASS = 5

if len(verified) < MIN_VERIFIED:
    raise ValueError(
        f"Only {len(verified)} independently verified labels. "
        f"At least {MIN_VERIFIED} are required. "
        "Review the original full-resolution images and enter "
        "independent labels. Do not copy provisional labels."
    )

class_counts = verified[TARGET].value_counts()

print("\nCLASS DISTRIBUTION")
print("------------------")
print(class_counts)

if set(class_counts.index) != ALLOWED:
    raise ValueError(
        "All three classes must be represented: "
        "Low, Medium and High."
    )

if (class_counts < MIN_PER_CLASS).any():
    raise ValueError(
        f"Each class needs at least {MIN_PER_CLASS} "
        "independently verified examples."
    )

# =====================================================
# 6. MERGE FEATURES AND VERIFIED LABELS
# =====================================================

data = verified[
    ["image_name", TARGET]
].merge(
    features,
    on="image_name",
    how="inner",
    validate="one_to_one"
)

if len(data) != len(verified):
    raise ValueError(
        "Some verified records could not be matched to features."
    )

# =====================================================
# 7. REMOVE LEAKAGE COLUMNS
# =====================================================

excluded_columns = [
    "image_name",
    TARGET,
    "provisional_priority",
    "priority_label",
    "risk_score",
    "provisional_score",
    "label_source",
    "review_reason",
    "reviewed_priority",
    "review_status",
    "reviewer_notes",
    "provisional_visual_assessment",
    "verification_status",
    "independent_human_verified_label"
]

X = data.drop(
    columns=[
        col for col in excluded_columns
        if col in data.columns
    ]
)

# Keep numerical model features only
X = X.select_dtypes(include="number")

y = data[TARGET]

if X.empty:
    raise ValueError(
        "No numerical features available for training."
    )

print("\nFEATURES USED")
print("-------------")

for column in X.columns:
    print(column)

# =====================================================
# 8. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nDATA SPLIT")
print("----------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =====================================================
# 9. DECISION TREE MODEL
# =====================================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
    )
])

# =====================================================
# 10. TRAIN MODEL
# =====================================================

print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Training completed.")

# =====================================================
# 11. PREDICTION
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# 12. EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"],
    zero_division=0,
    output_dict=True
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"]
)

print("\nDECISION TREE RESULTS")
print("=====================")

print("Accuracy:", round(accuracy, 4))

print(
    "Balanced Accuracy:",
    round(balanced_accuracy, 4)
)

print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=["Low", "Medium", "High"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# 13. SAVE MODEL
# =====================================================

model_path = (
    OUTPUT_DIR / "decision_tree_priority.joblib"
)

joblib.dump(model, model_path)

# =====================================================
# 14. SAVE METRICS
# =====================================================

metrics = {
    "model": "Decision Tree",
    "target": TARGET,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "class_distribution": {
        str(k): int(v)
        for k, v in class_counts.items()
    },
    "confusion_matrix": cm.tolist(),
    "class_order": [
        "Low",
        "Medium",
        "High"
    ],
    "classification_report": report,
    "features": list(X.columns),
    "evaluation_note": (
        "Exploratory evaluation using a small independently "
        "reviewed subset. A separate larger labeled holdout "
        "is needed for reliable generalization claims."
    )
}

metrics_path = (
    OUTPUT_DIR / "decision_tree_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

# =====================================================
# 15. SAVE PREDICTIONS
# =====================================================

prediction_df = pd.DataFrame({
    "actual_priority": y_test.values,
    "predicted_priority": predictions
})

prediction_df.to_csv(
    OUTPUT_DIR / "decision_tree_test_predictions.csv",
    index=False
)

# =====================================================
# 16. FINAL SUMMARY
# =====================================================

print("\nDECISION TREE PIPELINE COMPLETED")
print("================================")

print("Model:", model_path)
print("Metrics:", metrics_path)

print(
    "Predictions:",
    OUTPUT_DIR / "decision_tree_test_predictions.csv"
)

print("\nIMPORTANT:")
print(
    "This evaluation is exploratory and should not "
    "be presented as definitive performance."
)
