
from pathlib import Path
import csv
import random
import cv2

# ==================================================
# CONFIGURATION
# ==================================================

DATASET_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class"
)

OUTPUT_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\Annotation_Quality_Review"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "annotation_review.csv"

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}

SPLIT_SAMPLE_COUNTS = {
    "train": 60,
    "val": 20,
    "test": 20
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RANDOM_SEED = 42

# ==================================================
# LOAD EXISTING REVIEW RECORDS
# ==================================================

reviewed = {}

if CSV_FILE.exists():
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviewed[(row["split"], row["image_name"])] = row

fieldnames = [
    "image_name",
    "split",
    "annotation_present",
    "review_status",
    "remarks"
]

# ==================================================
# READ YOLO ANNOTATIONS
# ==================================================

def draw_boxes(image, label_path):

    height, width = image.shape[:2]
    annotation_present = False

    if not label_path.exists():
        return image, "MISSING LABEL FILE"

    lines = [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    if not lines:
        return image, "EMPTY LABEL FILE"

    annotation_present = True

    for line in lines:

        values = line.split()

        if len(values) != 5:
            cv2.putText(
                image, "INVALID ANNOTATION LINE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
            continue

        try:
            class_id = int(values[0])
            xc, yc, bw, bh = map(float, values[1:])
        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        label = CLASS_NAMES[class_id]

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            image, label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

    return image, "ANNOTATIONS PRESENT"

# ==================================================
# SELECT REVIEW SAMPLES
# ==================================================

random.seed(RANDOM_SEED)

samples = []

for split, sample_count in SPLIT_SAMPLE_COUNTS.items():

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print("Missing directories:", split)
        continue

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Separate non-empty and empty annotations.
    nonempty = []
    empty = []

    for image_path in image_files:

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        if label_path.stat().st_size > 0:
            nonempty.append(image_path)
        else:
            empty.append(image_path)

    # Include both annotated and empty-label images.
    candidates = nonempty + empty

    available = [
        p for p in candidates
        if (split, p.name) not in reviewed
    ]

    selected_count = min(sample_count, len(available))

    selected = random.sample(available, selected_count)

    samples.extend([(split, p) for p in selected])

    print(
        f"{split}: selected {selected_count} "
        f"images for review"
    )

# ==================================================
# INTERACTIVE REVIEW
# ==================================================

print("\nANNOTATION REVIEW CONTROLS")
print("C = Correct")
print("L = Incorrect class")
print("B = Incorrect bounding box")
print("M = Missing annotation")
print("U = Uncertain")
print("N = Skip")
print("Q = Save and quit")

for index, (split, image_path) in enumerate(samples, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    label_path = (
        DATASET_DIR / split / "labels" /
        (image_path.stem + ".txt")
    )

    image, annotation_status = draw_boxes(
        image, label_path
    )

    display = image.copy()

    cv2.putText(
        display,
        f"{split.upper()} | {index}/{len(samples)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 0, 0), 2
    )

    cv2.putText(
        display,
        annotation_status,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 0, 0), 2
    )

    cv2.putText(
        display,
        "C:Correct L:Wrong class B:Wrong box M:Missing U:Uncertain",
        (10, image.shape[0] - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 2
    )

    cv2.putText(
        display,
        "N:Skip Q:Quit",
        (10, image.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 2
    )

    window_name = "Road Damage Annotation Review"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 750)
    cv2.imshow(window_name, display)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        print("\nReview stopped by user.")
        break

    decisions = {
        ord("c"): "Correct",
        ord("l"): "Incorrect class",
        ord("b"): "Incorrect bounding box",
        ord("m"): "Missing annotation",
        ord("u"): "Uncertain"
    }

    if key == ord("n"):
        continue

    if key not in decisions:
        print("Unrecognized key. Image skipped.")
        continue

    status = decisions[key]

    row = {
        "image_name": image_path.name,
        "split": split,
        "annotation_present": (
            "Yes" if label_path.exists()
            and label_path.stat().st_size > 0
            else "No"
        ),
        "review_status": status,
        "remarks": ""
    }

    reviewed[(split, image_path.name)] = row

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviewed.values())

    print(image_path.name, "->", status)

cv2.destroyAllWindows()

print("\nReview session completed.")
print("Review records saved to:")
print(CSV_FILE)
