
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class"
)

OUTPUT_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\visual_inspection"
)

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}

SPLITS = ["train", "val", "test"]

SAMPLES_PER_SPLIT = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# --------------------------------------------------
# DRAW YOLO BOUNDING BOXES
# --------------------------------------------------

def draw_annotations(image_path, label_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    height, width = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as file:

            for line_number, line in enumerate(file, start=1):

                values = line.strip().split()

                if len(values) != 5:
                    print(
                        f"Invalid line in {label_path}, "
                        f"line {line_number}"
                    )
                    continue

                try:
                    class_id = int(values[0])
                    x_center, y_center, box_width, box_height = (
                        map(float, values[1:])
                    )
                except ValueError:
                    print(f"Invalid values in {label_path}")
                    continue

                if class_id not in CLASS_NAMES:
                    print(
                        f"Unexpected class ID {class_id} "
                        f"in {label_path}"
                    )
                    continue

                # Convert normalized coordinates to pixels
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Clip box coordinates to image boundaries
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    print(f"Degenerate box in {label_path}")
                    continue

                label = CLASS_NAMES[class_id]

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# SAMPLE AND SAVE IMAGES
# --------------------------------------------------

for split in SPLITS:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    output_split_dir = OUTPUT_DIR / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not label_dir.exists():
        print(f"Missing image or label directory for {split}")
        continue

    image_files = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )

    if not image_files:
        print(f"No images found in {split}")
        continue

    sample_count = min(SAMPLES_PER_SPLIT, len(image_files))

    selected_images = random.sample(image_files, sample_count)

    print(f"\nInspecting {split}: {sample_count} images")

    for image_path in selected_images:

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            print(f"Missing label file: {label_path}")
            continue

        annotated_image = draw_annotations(
            image_path,
            label_path
        )

        if annotated_image is None:
            continue

        output_path = output_split_dir / (
            image_path.stem + "_annotated.jpg"
        )

        plt.imsave(output_path, annotated_image)

        print(f"Saved: {output_path}")

print("\nVisual inspection images generated.")
print(f"Output directory: {OUTPUT_DIR}")
