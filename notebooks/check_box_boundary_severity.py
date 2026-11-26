
from pathlib import Path
import csv

DATASET = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class"
)

OUTPUT = Path(
    r"C:\College\M.tech\Datasets\RDD2022\Final_Dataset_Audit"
)

rows = []

for split in ["train", "val", "test"]:

    label_dir = DATASET / split / "labels"

    for label_file in label_dir.glob("*.txt"):

        for line_no, line in enumerate(
            label_file.read_text().splitlines(), start=1
        ):

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                continue

            try:
                class_id = int(values[0])
                xc, yc, bw, bh = map(float, values[1:])
            except ValueError:
                continue

            # YOLO box boundaries
            left = xc - bw / 2
            right = xc + bw / 2
            top = yc - bh / 2
            bottom = yc + bh / 2

            # Amount outside image, in normalized units
            excess = max(
                0,
                -left,
                right - 1,
                -top,
                bottom - 1
            )

            if excess > 0:
                rows.append({
                    "split": split,
                    "image": label_file.stem,
                    "line": line_no,
                    "class_id": class_id,
                    "x_center": xc,
                    "y_center": yc,
                    "width": bw,
                    "height": bh,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                    "maximum_overflow": excess
                })

output_file = OUTPUT / "box_boundary_severity.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()) if rows else [
            "split", "image", "line", "class_id",
            "x_center", "y_center", "width", "height",
            "left", "right", "top", "bottom",
            "maximum_overflow"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("Bounding-box boundary analysis completed.")
print("Flagged final annotation rows:", len(rows))

if rows:
    values = [r["maximum_overflow"] for r in rows]

    print("Maximum overflow:", max(values))
    print("Average overflow:", sum(values) / len(values))

    print("\nOverflow categories:")

    print("Small (<= 0.005):", sum(v <= 0.005 for v in values))
    print("Moderate (0.005-0.02):", sum(0.005 < v <= 0.02 for v in values))
    print("Large (> 0.02):", sum(v > 0.02 for v in values))

print("\nReport:", output_file)
