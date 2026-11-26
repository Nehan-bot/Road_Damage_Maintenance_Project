from pathlib import Path
import random
import shutil
import re
from collections import Counter

# Direct in-place merge into the current project dataset.
PROJECT_DATASET = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Corrected_Balanced600"
)
SOURCE_ROOT = Path(
    r"C:\College\M.tech\Datasets\RDD2022Full\RDD_SPLIT"
)

# Existing training annotation counts. Script stops if these don't match.
EXPECTED_EXISTING = {0: 1533, 1: 682, 2: 1580, 3: 2770}
TARGET_TRANSVERSE = max(EXPECTED_EXISTING.values())
NEEDED_TRANSVERSE = TARGET_TRANSVERSE - EXPECTED_EXISTING[1]

VALID_CLASSES = {"0", "1", "2", "3"}  # exclude source class 4 ("other")
TRANSVERSE_CLASS = "1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
PREFIX = "India_any_"


def parse_label(label_path):
    kept_rows = []
    counts = Counter()

    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        row = raw.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO row in {label_path}: {row}")

        class_id = parts[0]
        if class_id not in VALID_CLASSES:
            continue

        kept_rows.append(row)
        counts[int(class_id)] += 1

    return kept_rows, counts


def locate_image(image_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    for candidate in image_dir.glob(stem + ".*"):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.is_file():
            return candidate
    return None


def count_project_boxes(label_dir):
    counts = Counter()
    for path in label_dir.glob("*.txt"):
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = raw.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) == 5 and parts[0] in VALID_CLASSES:
                counts[int(parts[0])] += 1
    return counts


def next_filename_number(image_dir, label_dir):
    highest = 0
    pattern = re.compile(r"^India_any_(\d+)$", re.IGNORECASE)
    for folder in (image_dir, label_dir):
        for path in folder.iterdir():
            match = pattern.match(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def main():
    train_images = PROJECT_DATASET / "train" / "images"
    train_labels = PROJECT_DATASET / "train" / "labels"
    source_images = SOURCE_ROOT / "train" / "images"
    source_labels = SOURCE_ROOT / "train" / "labels"

    for path in (train_images, train_labels, source_images, source_labels):
        if not path.is_dir():
            raise FileNotFoundError(f"Folder not found: {path}")

    # Verify project dataset counts before making any changes.
    existing = count_project_boxes(train_labels)
    for class_id, expected in EXPECTED_EXISTING.items():
        if existing[class_id] != expected:
            raise RuntimeError(
                f"Expected {expected} boxes for class {class_id}, found "
                f"{existing[class_id]}. No files have been copied. "
                "Update/check the counts before proceeding."
            )

    # Build candidates from ORIGINAL source train labels, not transverse_only.
    # This preserves any co-occurring project classes 0-3 in each chosen image.
    candidates = []
    missing_source_images = 0

    for label_path in source_labels.glob("*.txt"):
        rows, class_counts = parse_label(label_path)
        transverse_boxes = class_counts[1]
        if transverse_boxes == 0:
            continue

        image_path = locate_image(source_images, label_path.stem)
        if image_path is None:
            missing_source_images += 1
            continue

        candidates.append({
            "image": image_path,
            "rows": rows,
            "counts": class_counts,
            "transverse": transverse_boxes,
        })

    random.Random(SEED).shuffle(candidates)

    chosen = []
    transverse_total = 0
    for item in candidates:
        chosen.append(item)
        transverse_total += item["transverse"]
        if transverse_total >= NEEDED_TRANSVERSE:
            break

    if transverse_total < NEEDED_TRANSVERSE:
        raise RuntimeError(
            f"Only {transverse_total} eligible transverse boxes found; "
            f"need {NEEDED_TRANSVERSE}. No files copied."
        )

    start_number = next_filename_number(train_images, train_labels)

    # Preflight all destination names before copying anything.
    planned = []
    for offset, item in enumerate(chosen):
        stem = f"{PREFIX}{start_number + offset:06d}"
        image_dest = train_images / (stem + item["image"].suffix.lower())
        label_dest = train_labels / (stem + ".txt")
        if image_dest.exists() or label_dest.exists():
            raise FileExistsError(
                f"Destination collision: {stem}. No files copied."
            )
        planned.append((item, image_dest, label_dest))

    added_counts = Counter()
    copied_paths = []

    try:
        for item, image_dest, label_dest in planned:
            shutil.copy2(item["image"], image_dest)
            copied_paths.append(image_dest)

            label_dest.write_text(
                "\n".join(item["rows"]) + "\n",
                encoding="utf-8"
            )
            copied_paths.append(label_dest)

            added_counts.update(item["counts"])

    except Exception:
        # Remove only files created by this run if a copy/write fails.
        for path in reversed(copied_paths):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    final_counts = {
        class_id: existing[class_id] + added_counts[class_id]
        for class_id in range(4)
    }

    print("\nIN-PLACE TRANSVERSE BALANCING COMPLETE")
    print(f"Images added: {len(planned)}")
    print(f"Transverse boxes added: {added_counts[1]} (needed ~{NEEDED_TRANSVERSE})")
    print(f"Source images missing: {missing_source_images}")
    print("\nAdded annotation boxes by class:")
    print(f"  0 longitudinal: {added_counts[0]}")
    print(f"  1 transverse:   {added_counts[1]}")
    print(f"  2 alligator:    {added_counts[2]}")
    print(f"  3 pothole:      {added_counts[3]}")
    print("\nFinal training annotation box counts:")
    print(f"  0 longitudinal: {final_counts[0]}")
    print(f"  1 transverse:   {final_counts[1]}")
    print(f"  2 alligator:    {final_counts[2]}")
    print(f"  3 pothole:      {final_counts[3]}")
    print(f"\nAdded files directly to:\n{train_images}\n{train_labels}")
    print("Existing validation/test folders and data.yaml were not changed.")
    print("Source RDD2022 dataset was not modified.")


if __name__ == "__main__":
    main()
