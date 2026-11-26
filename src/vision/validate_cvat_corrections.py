
from pathlib import Path
from collections import Counter
import math

# --------------------------------------------------
# 1. DATASET PATHS
# --------------------------------------------------

ORIGINAL = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class"
)

CORRECTED = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Corrected\RDD2022_India_4Class_cvat_Corrected"
)

NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}


# --------------------------------------------------
# 2. VALIDATION FUNCTION
# --------------------------------------------------

def validate():

    original_labels = ORIGINAL / "train" / "labels"
    corrected_labels = CORRECTED / "train" / "labels"
    images = CORRECTED / "train" / "images"

    image_files = [
        p for p in images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    label_files = list(corrected_labels.glob("*.txt"))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    missing = image_stems - label_stems
    orphan = label_stems - image_stems

    invalid = []
    counts = Counter()
    changed = []

    # --------------------------------------------------
    # 3. CHECK LABEL FILES
    # --------------------------------------------------

    for label in label_files:

        original = original_labels / label.name

        if not original.exists():
            invalid.append((label.name, "No original label"))
            continue

        if label.read_bytes() != original.read_bytes():
            changed.append(label.name)

        for line_no, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(),
            start=1
        ):

            parts = line.split()

            if len(parts) != 5:
                invalid.append(
                    (label.name, line_no, "Column count")
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

            except ValueError:
                invalid.append(
                    (label.name, line_no, "Invalid number")
                )
                continue

            values = (x, y, w, h)

            if class_id not in NAMES:
                invalid.append(
                    (label.name, line_no, "Invalid class")
                )
                continue

            if not all(math.isfinite(v) for v in values):
                invalid.append(
                    (label.name, line_no, "Non-finite value")
                )
                continue

            if not all(0 <= v <= 1 for v in values):
                invalid.append(
                    (label.name, line_no, "Coordinate outside range")
                )
                continue

            if (
                w <= 0 or h <= 0
                or x - w / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6
                or y - h / 2 < -1e-6
                or y + h / 2 > 1 + 1e-6
            ):
                invalid.append(
                    (label.name, line_no, "Invalid box geometry")
                )
                continue

            counts[class_id] += 1

    # --------------------------------------------------
    # 4. DISPLAY AUDIT RESULTS
    # --------------------------------------------------

    print("\n========== CORRECTED DATASET AUDIT ==========")

    print("Images:", len(image_files))
    print("Labels:", len(label_files))
    print("Missing labels:", len(missing))
    print("Orphan labels:", len(orphan))
    print("Invalid annotations:", len(invalid))
    print("Changed label files:", len(changed))

    print("\n========== CLASS DISTRIBUTION ==========")

    for class_id, name in NAMES.items():
        print(
            f"{name}: {counts[class_id]} annotations"
        )

    print("\n========== CHANGE SUMMARY ==========")

    print("Changed files:", len(changed))
    print("Unchanged files:", 44 - len(changed))

    if changed:
        print("\nChanged annotation files:")

        for name in changed:
            print(name)

    # --------------------------------------------------
    # 5. FINAL VALIDATION
    # --------------------------------------------------

    if (
        len(image_files) == 5368
        and len(label_files) == 5368
        and not missing
        and not orphan
        and not invalid
        and len(changed) <= 44
    ):

        print("\nSTRUCTURAL VALIDATION PASSED")

        print(
            "Changed annotation files:",
            len(changed)
        )

        print(
            "Unchanged annotation files:",
            44 - len(changed)
        )

    else:
        print("\nVALIDATION FAILED")
        print("Please inspect the reported issues.")


# --------------------------------------------------
# 6. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    validate()
