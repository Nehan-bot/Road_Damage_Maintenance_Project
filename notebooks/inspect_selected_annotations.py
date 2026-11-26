
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

DATASET_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India"
)

OUTPUT_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\selected_annotation_review"
)

SPLIT = "train"

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Other corruption",
    4: "Pothole"
}

FILENAMES = [
    "India_001751",
    "India_000416",
    "India_003760",
    "India_003433",
    "India_002121",
    "India_001584"
]

image_dir = DATASET_DIR / SPLIT / "images"
label_dir = DATASET_DIR / SPLIT / "labels"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename in FILENAMES:

    image_path = image_dir / f"{filename}.jpg"
    label_path = label_dir / f"{filename}.txt"

    if not image_path.exists():
        print("Image not found:", image_path)
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("Unable to read:", image_path)
        continue

    height, width = image.shape[:2]

    if label_path.exists():

        lines = label_path.read_text().strip().splitlines()

        for line in lines:

            if not line.strip():
                continue

            class_id, xc, yc, bw, bh = map(
                float, line.split()
            )

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            label = CLASS_NAMES.get(
                int(class_id), "Unknown"
            )

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

    output_path = OUTPUT_DIR / f"{filename}_review.jpg"

    cv2.imwrite(str(output_path), image)

    print("Saved:", output_path)

print("\nSelected annotation inspection completed.")
