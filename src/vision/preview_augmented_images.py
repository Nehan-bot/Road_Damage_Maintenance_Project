
from pathlib import Path
import random
import cv2

DATASET = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Balanced"
)

IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"

OUTPUT_DIR = DATASET / "visual_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Longitudinal",
    1: "Transverse",
    2: "Alligator",
    3: "Pothole"
}

COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 255),
    2: (0, 165, 255),
    3: (0, 255, 0)
}

random.seed(42)

augmented_images = sorted(
    p for p in IMAGE_DIR.glob("*_aug_*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)

selected = random.sample(
    augmented_images,
    min(20, len(augmented_images))
)

for image_path in selected:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None or not label_path.exists():
        print("Skipped:", image_path.name)
        continue

    height, width = image.shape[:2]

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            color = COLORS[class_id]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

print("Visual inspection images saved:", len(selected))
print("Output folder:", OUTPUT_DIR)
