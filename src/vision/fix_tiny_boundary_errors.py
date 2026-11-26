
from pathlib import Path
import csv

DATASET = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Balanced"
)

LABEL_DIR = DATASET / "train" / "labels"

TOLERANCE = 0.00001

audit_file = DATASET / "boundary_correction_audit.csv"

correction_count = 0
large_error_count = 0

with open(audit_file, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "label_file",
        "line_number",
        "class_id",
        "original_values",
        "corrected_values",
        "correction"
    ])

    for label_path in LABEL_DIR.glob("*.txt"):

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        updated_lines = []

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            if len(parts) != 5:
                updated_lines.append(line)
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                updated_lines.append(line)
                continue

            original = (x, y, w, h)

            left = x - w / 2
            right = x + w / 2
            top = y - h / 2
            bottom = y + h / 2

            overflow = max(
                0,
                -left,
                right - 1,
                -top,
                bottom - 1
            )

            if overflow > 0:

                if overflow <= TOLERANCE:

                    # Clamp the box edges to image boundaries.
                    left = max(0.0, left)
                    right = min(1.0, right)
                    top = max(0.0, top)
                    bottom = min(1.0, bottom)

                    x = (left + right) / 2
                    y = (top + bottom) / 2
                    w = right - left
                    h = bottom - top

                    correction_count += 1

                    writer.writerow([
                        label_path.name,
                        line_number,
                        class_id,
                        original,
                        (x, y, w, h),
                        "Tiny boundary correction"
                    ])

                else:
                    large_error_count += 1

            updated_lines.append(
                f"{class_id} {x:.8f} {y:.8f} "
                f"{w:.8f} {h:.8f}"
            )

        label_path.write_text(
            "\n".join(updated_lines) + (
                "\n" if updated_lines else ""
            ),
            encoding="utf-8"
        )

print("\n========== CORRECTION SUMMARY ==========")
print("Corrected annotations:", correction_count)
print("Larger issues left unchanged:", large_error_count)
print("Audit report:", audit_file)
