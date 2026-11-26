
from pathlib import Path
from collections import defaultdict
import random
import shutil
import csv
import zipfile

DATASET = Path(
    r"data\raw\road_images\RDD2022_India_4Class_Corrected_Balanced600"
)

OUTPUT = Path(r"experiments\manual_review\batch_01")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}

SAMPLES_PER_CLASS = 25
EMPTY_LABEL_SAMPLES = 25
SEED = 42


def main():
    random.seed(SEED)

    image_dir = DATASET / "train" / "images"
    label_dir = DATASET / "train" / "labels"

    output_images = OUTPUT / "images" / "train"
    output_labels = OUTPUT / "labels" / "train"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = {
        p.stem.lower(): p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        p.stem.lower(): p
        for p in label_dir.glob("*.txt")
    }

    class_candidates = defaultdict(list)
    empty_candidates = []

    for stem, image_path in images.items():
        label_path = labels.get(stem)

        if label_path is None:
            continue

        class_ids = set()

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.strip().split()

            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    if class_id in CLASS_NAMES:
                        class_ids.add(class_id)
                except ValueError:
                    pass

        if not class_ids:
            empty_candidates.append(stem)

        for class_id in class_ids:
            class_candidates[class_id].append(stem)

    selected = {}
    reasons = defaultdict(set)

    # Sample images for each damage class.
    for class_id, class_name in CLASS_NAMES.items():
        candidates = class_candidates[class_id].copy()
        random.shuffle(candidates)

        chosen = candidates[:SAMPLES_PER_CLASS]

        selected[class_id] = chosen

        for stem in chosen:
            reasons[stem].add(class_name)

        print(
            f"{class_name}: selected {len(chosen)} "
            f"from {len(candidates)} candidates"
        )

    # Sample genuinely empty-label images.
    random.shuffle(empty_candidates)
    chosen_empty = empty_candidates[:EMPTY_LABEL_SAMPLES]

    for stem in chosen_empty:
        reasons[stem].add("empty_label")

    print(f"Empty-label images selected: {len(chosen_empty)}")

    # Copy selected images and labels without modifying originals.
    manifest_rows = []

    for stem, reason_set in sorted(reasons.items()):
        source_image = images[stem]
        source_label = labels[stem]

        destination_image = output_images / source_image.name
        destination_label = output_labels / source_label.name

        shutil.copy2(source_image, destination_image)
        shutil.copy2(source_label, destination_label)

        manifest_rows.append({
            "image": source_image.name,
            "review_categories": ";".join(sorted(reason_set)),
            "original_split": "train",
        })

    manifest_path = OUTPUT / "manifest.csv"

    with manifest_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "review_categories",
                "original_split",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # YOLO dataset config for CVAT import.
    yaml_text = """path: .
train: images/train
val: images/train
nc: 4
names:
  0: longitudinal crack
  1: transverse crack
  2: alligator crack
  3: pothole
"""

    (OUTPUT / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    zip_path = OUTPUT.parent / "manual_review_batch_01.zip"

    with zipfile.ZipFile(
        zip_path, "w", zipfile.ZIP_DEFLATED
    ) as z:
        for file_path in OUTPUT.rglob("*"):
            if file_path.is_file():
                z.write(
                    file_path,
                    file_path.relative_to(OUTPUT.parent),
                )

    print("\nReview batch created.")
    print(f"Unique images: {len(manifest_rows)}")
    print(f"Folder: {OUTPUT.resolve()}")
    print(f"ZIP: {zip_path.resolve()}")
    print("Original dataset was not changed.")


if __name__ == "__main__":
    main()
