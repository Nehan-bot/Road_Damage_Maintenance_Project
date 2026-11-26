
from pathlib import Path
from collections import Counter
import csv
import math

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022"
)

ORIGINAL_DIR = BASE_DIR / "RDD2022_India"

FINAL_DIR = BASE_DIR / "RDD2022_India_4Class"

REVIEW_CSV = BASE_DIR / "Annotation_Quality_Review" / "annotation_review.csv"

REPORT_DIR = BASE_DIR / "Final_Dataset_Audit"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

ORIGINAL_CLASSES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Other corruption",
    4: "Pothole"
}

FINAL_CLASSES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}

# Original-to-final class mapping.
CLASS_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    4: 3
}

issues = []
comparison_rows = []
summary = {}

# ============================================================
# HELPERS
# ============================================================

def add_issue(split, image, issue, details=""):
    issues.append({
        "split": split,
        "image": image,
        "issue": issue,
        "details": details
    })


def get_images(folder):
    if not folder.exists():
        return {}

    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def get_labels(folder):
    if not folder.exists():
        return {}

    return {
        p.stem: p
        for p in folder.glob("*.txt")
    }


def read_labels(path, split, image_name, dataset_type):

    if not path.exists():
        return None

    annotations = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        add_issue(split, image_name, "LABEL_READ_ERROR", str(exc))
        return None

    for line_number, line in enumerate(lines, start=1):

        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            add_issue(
                split, image_name,
                "INVALID_FIELD_COUNT",
                f"{dataset_type}: line {line_number}"
            )
            continue

        try:
            class_value = float(values[0])

            if not class_value.is_integer():
                raise ValueError("Class ID is not an integer")

            class_id = int(class_value)

            coords = [float(v) for v in values[1:]]

            if not all(math.isfinite(v) for v in coords):
                raise ValueError("Non-finite coordinate")

            xc, yc, width, height = coords

        except ValueError as exc:
            add_issue(
                split, image_name,
                "INVALID_ANNOTATION_VALUE",
                f"{dataset_type}: line {line_number}: {exc}"
            )
            continue

        allowed = (
            ORIGINAL_CLASSES
            if dataset_type == "original"
            else FINAL_CLASSES
        )

        if class_id not in allowed:
            add_issue(
                split, image_name,
                "INVALID_CLASS_ID",
                f"{dataset_type}: class {class_id}"
            )
            continue

        if not (0 <= xc <= 1 and 0 <= yc <= 1):
            add_issue(
                split, image_name,
                "INVALID_CENTER_COORDINATE",
                f"{dataset_type}: line {line_number}"
            )

        if not (0 < width <= 1 and 0 < height <= 1):
            add_issue(
                split, image_name,
                "INVALID_BOX_SIZE",
                f"{dataset_type}: line {line_number}"
            )

        if (
            width > 0 and height > 0
            and not (
                0 <= xc - width / 2
                and xc + width / 2 <= 1
                and 0 <= yc - height / 2
                and yc + height / 2 <= 1
            )
        ):
            add_issue(
                split, image_name,
                "BOX_OUTSIDE_IMAGE",
                f"{dataset_type}: line {line_number}"
            )

        annotations.append((class_id, *coords))

    return annotations


def annotations_match(original, final):

    if original is None or final is None:
        return False

    expected = []

    for annotation in original:

        old_class = annotation[0]

        # Exclude original class 3.
        if old_class == 3:
            continue

        new_class = CLASS_MAPPING[old_class]

        expected.append((new_class, *annotation[1:]))

    if len(expected) != len(final):
        return False

    # Compare annotation rows independent of their order.
    unmatched = list(final)

    for expected_row in expected:

        match_index = None

        for i, final_row in enumerate(unmatched):

            if final_row[0] != expected_row[0]:
                continue

            if all(
                abs(a - b) <= 1e-6
                for a, b in zip(expected_row[1:], final_row[1:])
            ):
                match_index = i
                break

        if match_index is None:
            return False

        unmatched.pop(match_index)

    return len(unmatched) == 0


# ============================================================
# DATASET AUDIT
# ============================================================

print("\n" + "=" * 65)
print("COMPLETE ROAD DAMAGE DATASET AUDIT")
print("=" * 65)

for split in SPLITS:

    print(f"\nChecking {split.upper()}...")

    original_image_dir = ORIGINAL_DIR / split / "images"
    original_label_dir = ORIGINAL_DIR / split / "labels"

    final_image_dir = FINAL_DIR / split / "images"
    final_label_dir = FINAL_DIR / split / "labels"

    original_images = get_images(original_image_dir)
    original_labels = get_labels(original_label_dir)

    final_images = get_images(final_image_dir)
    final_labels = get_labels(final_label_dir)

    stats = {
        "original_images": len(original_images),
        "final_images": len(final_images),
        "original_labels": len(original_labels),
        "final_labels": len(final_labels),
        "empty_labels": 0,
        "retained_annotations": 0,
        "excluded_annotations": 0,
        "class_counts": Counter(),
        "comparison_matches": 0,
        "comparison_mismatches": 0
    }

    for name in sorted(set(original_images) | set(final_images)):

        if name not in original_images:
            add_issue(split, name, "IMAGE_MISSING_FROM_ORIGINAL")
            continue

        if name not in final_images:
            add_issue(split, name, "IMAGE_MISSING_FROM_FINAL")
            continue

        original_image = original_images[name]
        final_image = final_images[name]

        if original_image.suffix.lower() != final_image.suffix.lower():
            add_issue(split, name, "IMAGE_EXTENSION_MISMATCH")

        if name not in original_labels:
            add_issue(split, name, "ORIGINAL_LABEL_FILE_MISSING")
            continue

        if name not in final_labels:
            add_issue(split, name, "FINAL_LABEL_FILE_MISSING")
            continue

        original_annotations = read_labels(
            original_labels[name],
            split, name, "original"
        )

        final_annotations = read_labels(
            final_labels[name],
            split, name, "final"
        )

        if original_annotations is None or final_annotations is None:
            continue

        if len(final_annotations) == 0:
            stats["empty_labels"] += 1

        for class_id, *_ in original_annotations:

            if class_id == 3:
                stats["excluded_annotations"] += 1
            else:
                stats["retained_annotations"] += 1

        for class_id, *_ in final_annotations:
            stats["class_counts"][class_id] += 1

        match = annotations_match(
            original_annotations,
            final_annotations
        )

        if match:
            stats["comparison_matches"] += 1
        else:
            stats["comparison_mismatches"] += 1

            add_issue(
                split, name,
                "ORIGINAL_FINAL_ANNOTATION_MISMATCH",
                "Final labels do not match the expected four-class conversion."
            )

        comparison_rows.append({
            "split": split,
            "image": name,
            "original_annotations": len(original_annotations),
            "final_annotations": len(final_annotations),
            "conversion_match": match
        })

    summary[split] = stats

    print("Original images:", stats["original_images"])
    print("Final images:", stats["final_images"])
    print("Original label files:", stats["original_labels"])
    print("Final label files:", stats["final_labels"])
    print("Empty final labels:", stats["empty_labels"])
    print("Retained original annotations:", stats["retained_annotations"])
    print("Excluded original annotations:", stats["excluded_annotations"])
    print("Conversion matches:", stats["comparison_matches"])
    print("Conversion mismatches:", stats["comparison_mismatches"])
    print("Final class distribution:")

    for class_id, class_name in FINAL_CLASSES.items():
        print(
            f"  {class_name}: "
            f"{stats['class_counts'][class_id]}"
        )


# ============================================================
# REVIEW CSV AUDIT
# ============================================================

review_summary = Counter()
review_rows = []

print("\n" + "=" * 65)
print("ANNOTATION REVIEW CSV")
print("=" * 65)

if REVIEW_CSV.exists():

    with open(REVIEW_CSV, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            status = row.get("review_status", "").strip()
            review_summary[status] += 1
            review_rows.append(row)

    for status, count in review_summary.items():
        print(f"{status}: {count}")

    # Flag logically inconsistent records.
    for row in review_rows:

        if (
            row.get("annotation_present", "").strip().lower() == "no"
            and row.get("review_status", "").strip() == "Incorrect class"
        ):
            add_issue(
                row.get("split", ""),
                row.get("image_name", ""),
                "REVIEW_LOG_INCONSISTENCY",
                "Empty annotation cannot have an incorrect class in its label file."
            )

else:
    print("Review CSV not found:", REVIEW_CSV)


# ============================================================
# SAVE REPORTS
# ============================================================

issue_file = REPORT_DIR / "dataset_issues.csv"
comparison_file = REPORT_DIR / "conversion_comparison.csv"
summary_file = REPORT_DIR / "audit_summary.txt"

with open(issue_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["split", "image", "issue", "details"]
    )

    writer.writeheader()
    writer.writerows(issues)


with open(comparison_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "image",
            "original_annotations",
            "final_annotations",
            "conversion_match"
        ]
    )

    writer.writeheader()
    writer.writerows(comparison_rows)


with open(summary_file, "w", encoding="utf-8") as f:

    f.write("ROAD DAMAGE DATASET AUDIT SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    for split, stats in summary.items():

        f.write(f"{split.upper()}\n")
        f.write("-" * 30 + "\n")

        for key, value in stats.items():

            if key == "class_counts":
                for class_id, count in value.items():
                    f.write(
                        f"{FINAL_CLASSES[class_id]}: {count}\n"
                    )
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n")

    f.write(f"Total audit issues: {len(issues)}\n")
    f.write(f"Review records: {len(review_rows)}\n")
    f.write("\nThis audit does not determine semantic correctness visually.\n")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("AUDIT COMPLETED")
print("=" * 65)

print("Total conversion comparisons:", len(comparison_rows))
print(
    "Conversion mismatches:",
    sum(
        1 for row in comparison_rows
        if not row["conversion_match"]
    )
)

print("Total structural and review issues:", len(issues))

print("\nReports saved in:")
print(REPORT_DIR)

print("\n1. audit_summary.txt")
print("2. dataset_issues.csv")
print("3. conversion_comparison.csv")

print("\nIMPORTANT:")
print("This script checks structure and conversion consistency.")
print("It cannot determine whether a box visually covers the correct damage.")
