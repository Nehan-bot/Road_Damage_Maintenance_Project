
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# 1. DATASET PATHS
# --------------------------------------------------

DATASET = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Balanced"
)

IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"

# --------------------------------------------------
# 2. FIND IMAGES AND LABEL FILES
# --------------------------------------------------

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
]

labels = list(LABEL_DIR.glob("*.txt"))

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

missing_labels = image_stems - label_stems
orphan_labels = label_stems - image_stems

# --------------------------------------------------
# 3. INITIALIZE COUNTERS
# --------------------------------------------------

class_images = Counter()
class_instances = Counter()
invalid_boxes = []

# --------------------------------------------------
# 4. VALIDATE YOLO ANNOTATIONS
# --------------------------------------------------

for label in labels:

    present_classes = set()

    for line_number, line in enumerate(
        label.read_text(encoding="utf-8").splitlines(),
        start=1
    ):

        parts = line.split()

        if len(parts) != 5:
            invalid_boxes.append(
                (label.name, line_number, "Invalid columns", line)
            )
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

        except ValueError:
            invalid_boxes.append(
                (label.name, line_number, "Invalid values", line)
            )
            continue

        if class_id not in range(4):
            invalid_boxes.append(
                (label.name, line_number, "Unknown class", parts[0])
            )
            continue

        if not all(
            0 <= v <= 1 for v in (x, y, w, h)
        ):
            invalid_boxes.append(
                (label.name, line_number, "Coordinates out of range",
                 class_id, x, y, w, h)
            )
            continue

        if (
            w <= 0 or h <= 0
            or x - w / 2 < 0
            or x + w / 2 > 1
            or y - h / 2 < 0
            or y + h / 2 > 1
        ):
            invalid_boxes.append(
                (label.name, line_number, "Invalid box geometry",
                 class_id, x, y, w, h)
            )
            continue

        # Count only valid annotations
        class_instances[class_id] += 1
        present_classes.add(class_id)

    # Count each class once per image
    for class_id in present_classes:
        class_images[class_id] += 1

# --------------------------------------------------
# 5. DISPLAY VALIDATION RESULTS
# --------------------------------------------------

print("\n========== DATASET VALIDATION ==========")

print("Images:", len(images))
print("Label files:", len(labels))
print("Images missing labels:", len(missing_labels))
print("Labels without images:", len(orphan_labels))
print("Invalid annotations:", len(invalid_boxes))

print("\n========== CLASS DISTRIBUTION ==========")

names = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}

for class_id, name in names.items():

    print(
        f"{name}: "
        f"{class_instances[class_id]} annotations, "
        f"{class_images[class_id]} images"
    )

# --------------------------------------------------
# 6. DISPLAY INVALID ANNOTATION DETAILS
# --------------------------------------------------

if invalid_boxes:

    print("\n========== INVALID ANNOTATION DETAILS ==========")

    for issue in invalid_boxes[:100]:
        print(issue)

# --------------------------------------------------
# 7. FINAL RESULT
# --------------------------------------------------

if missing_labels or orphan_labels or invalid_boxes:

    print("\nVALIDATION FAILED: inspect the reported issues.")

else:

    print("\nVALIDATION PASSED.")
