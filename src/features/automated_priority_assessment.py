
from pathlib import Path
import pandas as pd
import cv2

# Project paths
ROOT = Path(".")
FEATURES = ROOT / "data/processed/vision/yolo_train_image_features.csv"
DETECTIONS = ROOT / "data/processed/vision/yolo_train_detections.csv"

IMAGES = (
    ROOT / "data/raw/road_images/"
    "RDD2022_India_4Class_Corrected_Balanced600/train/images"
)

OUT = ROOT / "data/processed/priority"
OUT.mkdir(parents=True, exist_ok=True)

# Load data
features = pd.read_csv(FEATURES)
detections = pd.read_csv(DETECTIONS)

if features["image_name"].duplicated().any():
    raise ValueError("Duplicate image names in features CSV.")

if not set(detections["image_name"]).issubset(
    set(features["image_name"])
):
    raise ValueError("Detection records contain unknown images.")

# Calculate bounding-box area relative to image area
area_records = []

for image_name, group in detections.groupby("image_name"):
    image_path = IMAGES / image_name
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]
    image_area = width * height

    box_areas = (
        (group["x2"] - group["x1"]).clip(lower=0)
        * (group["y2"] - group["y1"]).clip(lower=0)
    )

    area_records.append({
        "image_name": image_name,
        "total_box_area_ratio": min(
            box_areas.sum() / image_area, 1.0
        ),
        "largest_box_area_ratio": (
            box_areas.max() / image_area
        ),
        "mean_box_area_ratio": (
            box_areas.mean() / image_area
        )
    })

area_df = pd.DataFrame(area_records)

df = features.merge(
    area_df, on="image_name", how="left"
)

area_columns = [
    "total_box_area_ratio",
    "largest_box_area_ratio",
    "mean_box_area_ratio"
]

df[area_columns] = df[area_columns].fillna(0)

# Count columns
classes = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "pothole"
]

count_columns = [f"{name}_count" for name in classes]

df["total_detections"] = df[count_columns].sum(axis=1)
df["number_of_damage_classes"] = (
    df[count_columns] > 0
).sum(axis=1)

# Provisional severity score
# This is a heuristic triage score, NOT ground truth.
df["provisional_score"] = (
    3 * df["pothole_count"]
    + 3 * df["alligator crack_count"]
    + 2 * df["longitudinal crack_count"]
    + 2 * df["transverse crack_count"]
    + 10 * df["largest_box_area_ratio"]
    + 5 * df["total_box_area_ratio"]
)

def assign_priority(row):
    if row["total_detections"] == 0:
        return "Low"

    if row["provisional_score"] >= 8:
        return "High"

    if row["provisional_score"] >= 3:
        return "Medium"

    return "Low"

df["provisional_priority"] = df.apply(
    assign_priority, axis=1
)

# Identify uncertain cases for human review
def identify_uncertainty(row):
    reasons = []

    if row["total_detections"] == 0:
        reasons.append("No YOLO detections")

    if row["total_detections"] > 0 and (
        row["total_box_area_ratio"] < 0.005
    ):
        reasons.append("Very small detected area")

    confidence_columns = [
        f"{name}_mean_confidence" for name in classes
    ]

    active = [
        row[col] for col in confidence_columns
        if row[col] > 0
    ]

    if active and max(active) < 0.40:
        reasons.append("Low detection confidence")

    if active and min(active) < 0.30:
        reasons.append("At least one weak detection")

    if row["provisional_score"] in range(2, 9):
        reasons.append("Near priority threshold")

    return "; ".join(reasons)

df["review_reason"] = df.apply(
    identify_uncertainty, axis=1
)

df["human_review_required"] = (
    df["review_reason"] != ""
)

# Human verification fields remain blank
df["reviewed_priority"] = ""
df["review_status"] = "Pending"
df["reviewer_notes"] = ""

# Save complete provisional assessment
df["label_source"] = "Automated heuristic; not ground truth"

output_file = OUT / "automated_priority_assessment.csv"
review_file = OUT / "priority_uncertain_cases.csv"
summary_file = OUT / "priority_assessment_summary.csv"

df.to_csv(output_file, index=False)

review_df = df[df["human_review_required"]].copy()
review_df.to_csv(review_file, index=False)

summary = (
    df["provisional_priority"]
    .value_counts()
    .rename_axis("provisional_priority")
    .reset_index(name="image_count")
)
summary.to_csv(summary_file, index=False)

# Validation and summary
assert len(df) == len(features)
assert df["image_name"].nunique() == len(df)
assert df["reviewed_priority"].eq("").all()

print("\nAUTOMATED ASSESSMENT COMPLETED")
print("--------------------------------")
print("Total images:", len(df))
print("Images requiring human review:", len(review_df))
print("Provisional priority distribution:")
print(summary.to_string(index=False))

print("\nSaved:")
print(output_file)
print(review_file)
print(summary_file)

print("\nWARNING:")
print("Provisional labels are rule-generated.")
print("They are NOT independently verified ground truth.")
