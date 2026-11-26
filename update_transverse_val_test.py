
from pathlib import Path
import argparse
import hashlib
import math
import shutil

SOURCE_ROOT = Path(
    r"C:\College\M.tech\Datasets\RDD2022Full\RDD_SPLIT"
)

PROJECT_ROOT = Path(
    r"data\raw\road_images\RDD2022_India_4Class_Corrected_Balanced600"
)

VALID_CLASSES = {0, 1, 2, 3}
TRANSVERSE_CLASS = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_valid_labels(label_path):
    """Read valid YOLO rows for project classes 0-3."""
    rows = []

    if not label_path.exists():
        return rows

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO row: {label_path}, line {line_number}"
            )

        class_id = int(parts[0])

        # Exclude class 4 ('other corruption') and any unknown classes.
        if class_id not in VALID_CLASSES:
            continue

        values = [float(x) for x in parts[1:]]

        if not all(math.isfinite(x) for x in values):
            raise ValueError(
                f"Non-finite coordinates: {label_path}, line {line_number}"
            )

        x, y, width, height = values

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(
                f"Out-of-range YOLO box: {label_path}, line {line_number}"
            )

        rows.append(
            f"{class_id} "
            + " ".join(f"{value:.6f}" for value in values)
        )

    return rows


def box_key(row):
    parts = row.split()
    return tuple([int(parts[0])] + [round(float(x), 6) for x in parts[1:]])


def get_image_map(folder):
    image_map = {}

    for image in folder.iterdir():
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            key = image.stem.lower()

            if key in image_map:
                raise RuntimeError(
                    f"Duplicate image stem in {folder}: {image.stem}"
                )

            image_map[key] = image

    return image_map


def find_source_image(image_map, label_stem):
    image = image_map.get(label_stem.lower())

    if image is None:
        raise FileNotFoundError(
            f"No matching source image for label: {label_stem}"
        )

    return image


def unique_target_name(source_image, split, project_images):
    """
    Make a new name if the source filename exists but is a different image.
    """
    base_name = source_image.name
    target = project_images / base_name

    if not target.exists():
        return target

    if sha256_file(target) == sha256_file(source_image):
        return target

    file_hash = sha256_file(source_image)[:8]
    new_name = f"RDD{split.upper()}_{source_image.stem}_{file_hash}"
    target = project_images / f"{new_name}{source_image.suffix}"

    if target.exists():
        if sha256_file(target) == sha256_file(source_image):
            return target
        raise RuntimeError(f"Unexpected filename collision: {target}")

    return target


def main(apply_changes):
    totals = {
        "val": {"selected": 0, "added": 0, "updated": 0, "boxes": 0},
        "test": {"selected": 0, "added": 0, "updated": 0, "boxes": 0},
    }

    for split in ("val", "test"):
        source_images_dir = SOURCE_ROOT / split / "images"
        source_labels_dir = SOURCE_ROOT / split / "labels"

        project_images_dir = PROJECT_ROOT / split / "images"
        project_labels_dir = PROJECT_ROOT / split / "labels"

        if not source_images_dir.exists() or not source_labels_dir.exists():
            raise FileNotFoundError(
                f"Source images/labels missing for split: {split}"
            )

        if not project_images_dir.exists() or not project_labels_dir.exists():
            raise FileNotFoundError(
                f"Project images/labels missing for split: {split}"
            )

        source_image_map = get_image_map(source_images_dir)
        project_image_map = get_image_map(project_images_dir)

        print(f"\n--- {split.upper()} ---")

        for source_label in sorted(source_labels_dir.glob("*.txt")):
            source_rows = read_valid_labels(source_label)

            # Only select images with at least one transverse box.
            transverse_rows = [
                row for row in source_rows
                if int(row.split()[0]) == TRANSVERSE_CLASS
            ]

            if not transverse_rows:
                continue

            source_image = find_source_image(
                source_image_map, source_label.stem
            )

            totals[split]["selected"] += 1

            target_image = unique_target_name(
                source_image, split, project_images_dir
            )
            target_label = project_labels_dir / (
                target_image.stem + ".txt"
            )

            source_hash = sha256_file(source_image)

            # Image already exists and is exactly the same file:
            # preserve current labels and add missing transverse boxes.
            if target_image.exists():
                if sha256_file(target_image) != source_hash:
                    raise RuntimeError(
                        f"Target image differs unexpectedly: {target_image}"
                    )

                existing_rows = read_valid_labels(target_label)
                existing_keys = {box_key(row) for row in existing_rows}

                new_transverse_rows = [
                    row for row in transverse_rows
                    if box_key(row) not in existing_keys
                ]

                if new_transverse_rows:
                    totals[split]["updated"] += 1
                    totals[split]["boxes"] += len(new_transverse_rows)

                    if apply_changes:
                        merged_rows = existing_rows + new_transverse_rows
                        target_label.write_text(
                            "\n".join(merged_rows) + "\n",
                            encoding="utf-8"
                        )

                continue

            # New image: copy it and all valid class 0-3 boxes.
            totals[split]["added"] += 1
            totals[split]["boxes"] += len(source_rows)

            if apply_changes:
                shutil.copy2(source_image, target_image)
                target_label.write_text(
                    "\n".join(source_rows) + "\n",
                    encoding="utf-8"
                )

        print(f"Transverse-positive source images: {totals[split]['selected']}")
        print(f"New images to add:                {totals[split]['added']}")
        print(f"Existing labels to update:        {totals[split]['updated']}")
        print(f"Annotation boxes to add:          {totals[split]['boxes']}")

    print("\n--- SUMMARY ---")
    for split in ("val", "test"):
        print(
            f"{split}: selected={totals[split]['selected']}, "
            f"new_images={totals[split]['added']}, "
            f"updated_labels={totals[split]['updated']}"
        )

    if apply_changes:
        print("\nChanges applied.")
    else:
        print("\nDRY RUN ONLY. No files were changed.")
        print("If the counts look correct, run again with --apply.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy images and update labels. Default is dry-run."
    )
    args = parser.parse_args()

    main(apply_changes=args.apply)
