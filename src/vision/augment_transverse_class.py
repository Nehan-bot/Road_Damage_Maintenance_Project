
from pathlib import Path
from collections import Counter
import csv
import random
import shutil

import cv2
import numpy as np


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

SOURCE = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Corrected\RDD2022_India_4Class_cvat_Corrected"
)

OUTPUT = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Corrected_Balanced600"
)

TARGET_TRANSVERSE_IMAGES = 600
TRANSVERSE_CLASS_ID = 1
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# 2. READ YOLO LABELS
# --------------------------------------------------

def read_labels(label_path):
    annotations = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}: {line}"
                )

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class {class_id} in {label_path}"
                )

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                raise ValueError(
                    f"Invalid normalized box in {label_path}: {line}"
                )

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Non-positive box dimensions in {label_path}"
                )

            annotations.append([class_id, x, y, w, h])

    return annotations


def write_labels(label_path, annotations):
    with open(label_path, "w", encoding="utf-8") as file:
        for class_id, x, y, w, h in annotations:
            file.write(
                f"{class_id} {x:.8f} {y:.8f} "
                f"{w:.8f} {h:.8f}\n"
            )


# --------------------------------------------------
# 3. IMAGE AUGMENTATION
# --------------------------------------------------

def augment_image(image, annotations):
    result = image.copy()

    # Random horizontal flip
    flipped = random.random() < 0.5

    if flipped:
        result = cv2.flip(result, 1)

        for annotation in annotations:
            annotation[1] = 1.0 - annotation[1]

    # Brightness and contrast
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=beta
    )

    # Mild gamma adjustment
    gamma = random.uniform(0.85, 1.15)
    lookup = np.array([
        min(255, max(0, ((i / 255.0) ** (1.0 / gamma)) * 255))
        for i in range(256)
    ]).astype("uint8")

    result = cv2.LUT(result, lookup)

    # Mild blur occasionally
    blurred = random.random() < 0.15

    if blurred:
        result = cv2.GaussianBlur(result, (3, 3), 0)

    return result, annotations, {
        "horizontal_flip": flipped,
        "brightness_contrast_alpha": round(alpha, 4),
        "brightness_beta": beta,
        "gamma": round(gamma, 4),
        "mild_blur": blurred,
    }


# --------------------------------------------------
# 4. COUNT CLASS DISTRIBUTION
# --------------------------------------------------

def get_distribution(labels_dir):
    instance_counts = Counter()
    image_counts = Counter()

    for label_file in labels_dir.glob("*.txt"):
        annotations = read_labels(label_file)

        classes_in_image = set()

        for annotation in annotations:
            class_id = annotation[0]
            instance_counts[class_id] += 1
            classes_in_image.add(class_id)

        for class_id in classes_in_image:
            image_counts[class_id] += 1

    return instance_counts, image_counts


# --------------------------------------------------
# 5. MAIN PROCESS
# --------------------------------------------------

def main():

    source_images = SOURCE / "train" / "images"
    source_labels = SOURCE / "train" / "labels"

    output_images = OUTPUT / "train" / "images"
    output_labels = OUTPUT / "train" / "labels"

    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(
            "Original training images or labels folder not found."
        )

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT}\n"
            "Use a new empty output folder. Do not delete original data."
        )

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    image_files = sorted(
        p for p in source_images.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    # Find transverse-crack source images
    transverse_sources = []

    for image_path in image_files:
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label file: {label_path}"
            )

        annotations = read_labels(label_path)

        if any(
            a[0] == TRANSVERSE_CLASS_ID
            for a in annotations
        ):
            transverse_sources.append(image_path)

    print("Original training images:", len(image_files))
    print(
        "Original transverse-crack images:",
        len(transverse_sources)
    )

    if len(transverse_sources) == 0:
        raise ValueError("No transverse-crack images found.")

    if len(transverse_sources) > TARGET_TRANSVERSE_IMAGES:
        raise ValueError(
            "Original transverse image count exceeds target."
        )

    # Copy original training data
    print("\nCopying original training images and labels...")

    for image_path in image_files:
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        label_path = source_labels / f"{image_path.stem}.txt"

        shutil.copy2(
            label_path,
            output_labels / label_path.name
        )

    # Generate augmented images
    required_augmented = (
        TARGET_TRANSVERSE_IMAGES - len(transverse_sources)
    )

    print("\nAugmented images to generate:", required_augmented)

    audit_path = OUTPUT / "augmentation_audit.csv"

    with open(
        audit_path, "w", newline="", encoding="utf-8"
    ) as audit_file:

        writer = csv.DictWriter(
            audit_file,
            fieldnames=[
                "generated_image",
                "source_image",
                "horizontal_flip",
                "brightness_contrast_alpha",
                "brightness_beta",
                "gamma",
                "mild_blur",
            ],
        )

        writer.writeheader()

        for index in range(required_augmented):

            source_path = transverse_sources[
                index % len(transverse_sources)
            ]

            source_label_path = (
                source_labels / f"{source_path.stem}.txt"
            )

            image = cv2.imread(str(source_path))

            if image is None:
                raise ValueError(
                    f"Could not read image: {source_path}"
                )

            annotations = read_labels(source_label_path)

            augmented, new_annotations, params = augment_image(
                image,
                [a.copy() for a in annotations]
            )

            new_stem = f"{source_path.stem}_aug_{index + 1:04d}"

            new_image_path = (
                output_images / f"{new_stem}{source_path.suffix}"
            )

            new_label_path = (
                output_labels / f"{new_stem}.txt"
            )

            success = cv2.imwrite(
                str(new_image_path),
                augmented
            )

            if not success:
                raise IOError(
                    f"Could not save image: {new_image_path}"
                )

            write_labels(
                new_label_path,
                new_annotations
            )

            writer.writerow({
                "generated_image": new_image_path.name,
                "source_image": source_path.name,
                **params,
            })

            if (index + 1) % 100 == 0:
                print(
                    f"Generated {index + 1} / "
                    f"{required_augmented}"
                )

    # Final distribution
    instance_counts, image_counts = get_distribution(
        output_labels
    )

    print("\n========== FINAL TRAINING DISTRIBUTION ==========")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"\n{class_name}")
        print("Annotation instances:", instance_counts[class_id])
        print("Unique images:", image_counts[class_id])

    print("\nTotal training images:", len(list(output_images.iterdir())))
    print("Audit report:", audit_path)
    print("\nAugmentation completed.")


if __name__ == "__main__":
    main()
