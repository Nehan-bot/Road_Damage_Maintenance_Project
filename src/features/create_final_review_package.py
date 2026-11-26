
from pathlib import Path
import pandas as pd
import cv2
import math

ROOT = Path(".")
INPUT = ROOT / "data/processed/priority/priority_uncertain_cases.csv"

IMAGE_DIR = (
    ROOT / "data/raw/road_images/"
    "RDD2022_India_4Class_Corrected_Balanced600/train/images"
)

OUT = ROOT / "data/processed/priority"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT).fillna({
    "review_reason": "",
    "provisional_priority": ""
})

# Build mutually exclusive sampling groups
def category(row):
    reason = str(row["review_reason"]).lower()

    if "no yolo detections" in reason:
        return "No detections"
    if "low detection confidence" in reason:
        return "Low confidence"
    if "very small detected area" in reason:
        return "Small detected area"
    return "Other uncertainty"

df["review_group"] = df.apply(category, axis=1)

# Target distribution: 20 + 15 + 10 + 5
targets = {
    "No detections": 20,
    "Low confidence": 15,
    "Small detected area": 10,
    "Other uncertainty": 5
}

parts = []

for group, count in targets.items():
    candidates = df[df["review_group"] == group]

    if len(candidates) == 0:
        continue

    selected = candidates.sample(
        n=min(count, len(candidates)),
        random_state=42
    )
    parts.append(selected)

sample = pd.concat(parts).drop_duplicates("image_name")

# Fill any shortfall from remaining uncertain cases
if len(sample) < 50:
    remaining = df[
        ~df["image_name"].isin(sample["image_name"])
    ]
    extra = remaining.sample(
        n=min(50 - len(sample), len(remaining)),
        random_state=42
    )
    sample = pd.concat([sample, extra])

sample = sample.head(50).copy()

sample["image_path"] = sample["image_name"].apply(
    lambda name: str(IMAGE_DIR / name)
)
sample["reviewed_priority"] = ""
sample["review_status"] = "Pending"
sample["reviewer_notes"] = ""

csv_path = OUT / "final_review_sample_50.csv"
sample.to_csv(csv_path, index=False)

# Create contact sheet
tiles = []
tile_width, tile_height = 240, 180

for _, row in sample.iterrows():
    path = Path(row["image_path"])
    image = cv2.imread(str(path))

    if image is None:
        continue

    image = cv2.resize(image, (tile_width, 140))
    tile = 255 * (image * 0 + 1)
    tile = tile.astype("uint8")

    tile[:140] = image
    cv2.putText(
        tile,
        row["image_name"],
        (5, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 0, 0), 1
    )
    cv2.putText(
        tile,
        row["review_group"][:30],
        (5, 174),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 0, 0), 1
    )
    tiles.append(tile)

columns = 5
rows = math.ceil(len(tiles) / columns)

import numpy as np

sheet = np.ones(
    (rows * tile_height, columns * tile_width, 3),
    dtype="uint8"
) * 255

for i, tile in enumerate(tiles):
    y = (i // columns) * tile_height
    x = (i % columns) * tile_width

    sheet[
        y:y + tile.shape[0],
        x:x + tile.shape[1]
    ] = tile
contact_path = OUT / "final_review_contact_sheet.jpg"
cv2.imwrite(str(contact_path), sheet)

assert len(sample) == 50
assert sample["image_name"].nunique() == 50

print("REVIEW PACKAGE CREATED")
print("Images:", len(sample))
print("\nReview groups:")
print(sample["review_group"].value_counts())
print("\nCSV:", csv_path)
print("Contact sheet:", contact_path)
print("\nOnly assign Low, Medium or High after inspecting the image.")
