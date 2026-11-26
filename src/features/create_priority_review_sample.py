
from pathlib import Path
import pandas as pd

# Input and output
INPUT = Path("data/processed/vision/yolo_train_image_features.csv")
OUTPUT_DIR = Path("data/processed/priority")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "priority_manual_review_200.csv"

# Load training features
df = pd.read_csv(INPUT)

# Validate input
if len(df) < 200:
    raise ValueError("At least 200 training images are required.")

if df["image_name"].duplicated().any():
    raise ValueError("Duplicate image names found.")

# Identify damage classes
classes = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "pothole"
]

count_cols = [f"{name}_count" for name in classes]
confidence_cols = [
    f"{name}_mean_confidence" for name in classes
]

# Group images by detection pattern
df["detected_classes"] = (
    df[count_cols] > 0
).sum(axis=1)

df["review_group"] = "Multiple damage classes"
df.loc[df["detected_classes"] == 0, "review_group"] = "No detections"
df.loc[df["detected_classes"] == 1, "review_group"] = "Single damage class"

# Confidence score for sampling coverage only
total = df[count_cols].sum(axis=1)

weighted_confidence = sum(
    df[count_cols[i]] * df[confidence_cols[i]]
    for i in range(4)
)

df["mean_detection_confidence"] = (
    weighted_confidence / total.replace(0, 1)
)

# Select a representative sample from each group
# The sample is for human review, not automatic labeling.
sample_parts = []

group_sizes = {
    "No detections": 40,
    "Single damage class": 100,
    "Multiple damage classes": 60
}

for group, size in group_sizes.items():
    group_df = df[df["review_group"] == group]

    if len(group_df) == 0:
        continue

    take = min(size, len(group_df))

    # Spread selection across confidence levels
    if group == "No detections":
        selected = group_df.sample(
            n=take, random_state=42
        )
    else:
        group_df = group_df.copy()
        group_df["confidence_band"] = pd.qcut(
            group_df["mean_detection_confidence"],
            q=min(4, group_df["mean_detection_confidence"].nunique()),
            labels=False,
            duplicates="drop"
        )

        selected = (
            group_df.groupby(
                "confidence_band", group_keys=False
            )
            .apply(
                lambda x: x.sample(
                    n=max(1, round(take * len(x) / len(group_df))),
                    random_state=42
                )
            )
        )

        selected = selected.drop_duplicates("image_name")

        if len(selected) < take:
            remaining = group_df[
                ~group_df["image_name"].isin(selected["image_name"])
            ]
            selected = pd.concat([
                selected,
                remaining.sample(
                    n=min(take - len(selected), len(remaining)),
                    random_state=42
                )
            ])

        selected = selected.head(take)

    sample_parts.append(selected)

sample = pd.concat(sample_parts).drop_duplicates("image_name")

# Fill any shortfall from remaining training images
if len(sample) < 200:
    remaining = df[
        ~df["image_name"].isin(sample["image_name"])
    ]
    sample = pd.concat([
        sample,
        remaining.sample(
            n=200 - len(sample), random_state=42
        )
    ])

sample = sample.sample(frac=1, random_state=42).head(200)

# Human-review fields: deliberately blank labels
sample["reviewed_priority"] = ""
sample["review_status"] = "Pending"
sample["reviewer_notes"] = ""

# Save useful columns
sample = sample.drop(
    columns=["detected_classes", "review_group",
             "confidence_band"],
    errors="ignore"
)

sample.to_csv(OUTPUT, index=False)

# Validation
check = pd.read_csv(OUTPUT)

assert len(check) == 200
assert check["image_name"].nunique() == 200
assert check["reviewed_priority"].fillna("").eq("").all()
assert check["review_status"].eq("Pending").all()

print("\nREVIEW SAMPLE CREATED")
print("Images:", len(check))
print("Unique images:", check["image_name"].nunique())
print("\nReview status:")
print(check["review_status"].value_counts())
print("\nSaved:", OUTPUT)
