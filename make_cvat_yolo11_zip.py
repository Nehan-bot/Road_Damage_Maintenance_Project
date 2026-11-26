
from pathlib import Path
import zipfile

ROOT = Path("experiments/manual_review/batch_01")
IMAGE_DIR = ROOT / "images" / "train"
LABEL_DIR = ROOT / "labels" / "train"

OUTPUT_ZIP = Path(
    "experiments/manual_review/"
    "manual_review_batch_01_CVAT_YOLO11.zip"
)

CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "pothole",
]

# CVAT YOLO 1.1 annotation archive structure.
obj_data = """classes = 4
names = obj.names
train = train.txt
backup = backup/
"""

train_entries = []
label_files = []

for image_path in sorted(IMAGE_DIR.iterdir()):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        continue

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file for image: {image_path.name}"
        )

    # CVAT matches annotation txt to task frame by image basename.
    train_entries.append(
        f"obj_train_data/{image_path.name}"
    )

    label_files.append(label_path)

if len(train_entries) != 125:
    print(f"WARNING: Found {len(train_entries)} images, expected 125.")

OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(
    OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED
) as archive:
    archive.writestr("obj.data", obj_data)

    archive.writestr(
        "obj.names",
        "\n".join(CLASS_NAMES) + "\n"
    )

    archive.writestr(
        "train.txt",
        "\n".join(train_entries) + "\n"
    )

    for label_path in label_files:
        archive.write(
            label_path,
            f"obj_train_data/{label_path.name}"
        )

print("Created CVAT YOLO 1.1 ZIP:")
print(OUTPUT_ZIP.resolve())
print("Images listed:", len(train_entries))
print("Label files included:", len(label_files))

with zipfile.ZipFile(OUTPUT_ZIP, "r") as archive:
    names = archive.namelist()
    print("obj.data present:", "obj.data" in names)
    print("obj.names present:", "obj.names" in names)
    print("train.txt present:", "train.txt" in names)
    print("Label txt files:", sum(
        name.startswith("obj_train_data/")
        and name.endswith(".txt")
        for name in names
    ))
