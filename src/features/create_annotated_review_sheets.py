
from pathlib import Path
import pandas as pd
import cv2
import math

ROOT = Path(".")

REVIEW_CSV = (
    ROOT / "data/processed/priority/final_review_sample_50.csv"
)

DETECTIONS_CSV = (
    ROOT / "data/processed/vision/yolo_train_detections.csv"
)

IMAGE_DIR = (
    ROOT / "data/raw/road_images/"
    "RDD2022_India_4Class_Corrected_Balanced600/train/images"
)

OUTPUT_DIR = (
    ROOT / "data/processed/priority/annotated_review_sheets"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

review = pd.read_csv(REVIEW_CSV).fillna("")
detections = pd.read_csv(DETECTIONS_CSV)

# Validate
required = {"image_name", "provisional_priority", "review_reason"}

if not required.issubset(review.columns):
    raise ValueError(f"Review CSV needs columns: {required}")

if len(review) != 50 or review["image_name"].nunique() != 50:
    raise ValueError("Expected exactly 50 unique review images.")

# Prepare detection lookup
detection_groups = {
    name: group
    for name, group in detections.groupby("image_name")
}

# Layout: 2 columns x 5 rows per page
COLS = 2
ROWS = 5
TILE_W = 640
TILE_H = 390
PER_PAGE = COLS * ROWS

def draw_image(row):
    name = row["image_name"]
    path = IMAGE_DIR / name

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    h, w = image.shape[:2]
    scale = min(600 / w, 280 / h)

    image = cv2.resize(
        image,
        (int(w * scale), int(h * scale))
    )

    canvas = 255 * (
        __import__("numpy").ones(
            (TILE_H, TILE_W, 3), dtype="uint8"
        )
    )

    ih, iw = image.shape[:2]
    x_offset = (TILE_W - iw) // 2
    y_offset = 55

    canvas[
        y_offset:y_offset + ih,
        x_offset:x_offset + iw
    ] = image

    # Draw detections
    group = detection_groups.get(name)

    if group is not None:
        for _, det in group.iterrows():
            x1 = int(det["x1"] * scale) + x_offset
            y1 = int(det["y1"] * scale) + y_offset
            x2 = int(det["x2"] * scale) + x_offset
            y2 = int(det["y2"] * scale) + y_offset

            cv2.rectangle(
                canvas, (x1, y1), (x2, y2),
                (0, 0, 255), 2
            )

            label = (
                f'{det["damage_class"]} '
                f'{float(det["confidence"]):.2f}'
            )

            cv2.putText(
                canvas, label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 255), 1
            )

    priority = str(row["provisional_priority"])
    reason = str(row["review_reason"])

    cv2.putText(
        canvas, name,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 2
    )

    cv2.putText(
        canvas, f"Provisional: {priority}",
        (10, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 0, 0), 2
    )

    cv2.putText(
        canvas, reason[:75],
        (10, 365),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 0, 0), 1
    )

    return canvas


# Generate five sheets
for page in range(math.ceil(len(review) / PER_PAGE)):
    subset = review.iloc[
        page * PER_PAGE:(page + 1) * PER_PAGE
    ]

    sheet = 255 * (
        __import__("numpy").ones(
            (ROWS * TILE_H, COLS * TILE_W, 3),
            dtype="uint8"
        )
    )

    for index, (_, row) in enumerate(subset.iterrows()):
        tile = draw_image(row)

        x = (index % COLS) * TILE_W
        y = (index // COLS) * TILE_H

        sheet[y:y + TILE_H, x:x + TILE_W] = tile

    output = OUTPUT_DIR / f"review_page_{page + 1}.jpg"
    cv2.imwrite(str(output), sheet)

    print("Created:", output)

print("\nCOMPLETED")
print("Images reviewed:", len(review))
print("Pages generated:", math.ceil(len(review) / PER_PAGE))
