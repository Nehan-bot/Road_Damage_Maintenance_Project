
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import math

DATASET = Path(
    r"data\raw\road_images\RDD2022_India_4Class_Corrected_Balanced600"
)

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "pothole",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    hashes_by_split = defaultdict(dict)
    total_errors = 0

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        images = {
            p.stem.lower(): p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels = {
            p.stem.lower(): p
            for p in label_dir.glob("*.txt")
            if p.is_file()
        }

        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))

        class_counts = Counter()
        transverse_images = 0
        empty_labels = 0
        invalid_rows = []

        for stem, label_path in labels.items():
            has_transverse = False
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            if not any(line.strip() for line in lines):
                empty_labels += 1

            for line_number, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    invalid_rows.append(
                        f"{label_path.name}:{line_number} wrong field count"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_rows.append(
                        f"{label_path.name}:{line_number} nonnumeric value"
                    )
                    continue

                if class_id not in CLASS_NAMES:
                    invalid_rows.append(
                        f"{label_path.name}:{line_number} invalid class {class_id}"
                    )
                    continue

                if not all(math.isfinite(x) for x in coords):
                    invalid_rows.append(
                        f"{label_path.name}:{line_number} non-finite coordinate"
                    )
                    continue

                x, y, width, height = coords

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):
                    invalid_rows.append(
                        f"{label_path.name}:{line_number} invalid box range"
                    )
                    continue

                class_counts[class_id] += 1

                if class_id == 1:
                    has_transverse = True

            if has_transverse:
                transverse_images += 1

        for stem, image_path in images.items():
            hashes_by_split[split][sha256(image_path)] = image_path.name

        print(f"\n===== {split.upper()} =====")
        print(f"Images: {len(images)}")
        print(f"Label files: {len(labels)}")
        print(f"Missing labels: {len(missing_labels)}")
        print(f"Orphan labels: {len(orphan_labels)}")
        print(f"Empty label files: {empty_labels}")
        print(f"Invalid label rows: {len(invalid_rows)}")
        print(f"Transverse-positive images: {transverse_images}")

        print("Annotation counts:")
        for class_id, name in CLASS_NAMES.items():
            print(f"  {class_id} {name}: {class_counts[class_id]}")

        if missing_labels:
            print("First missing label stems:", missing_labels[:10])

        if orphan_labels:
            print("First orphan label stems:", orphan_labels[:10])

        if invalid_rows:
            print("First invalid rows:")
            for item in invalid_rows[:10]:
                print(" ", item)

        total_errors += (
            len(missing_labels)
            + len(orphan_labels)
            + len(invalid_rows)
        )

    print("\n===== CROSS-SPLIT IMAGE DUPLICATES =====")

    duplicate_found = False

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            shared_hashes = (
                set(hashes_by_split[split_a])
                & set(hashes_by_split[split_b])
            )

            print(
                f"{split_a} vs {split_b}: "
                f"{len(shared_hashes)} identical image files"
            )

            if shared_hashes:
                duplicate_found = True

    print("\n===== FINAL STATUS =====")
    print(f"Missing/orphan/invalid label issues: {total_errors}")

    if duplicate_found:
        print("WARNING: Identical images occur across splits.")
    else:
        print("No identical image files found across splits.")

    print("Audit complete. No files were changed.")


if __name__ == "__main__":
    main()
