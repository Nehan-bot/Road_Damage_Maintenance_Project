
from pathlib import Path
from collections import Counter
import shutil
import zipfile

# Original four-class training dataset
DATASET = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class"
)

IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"

# CVAT preparation output
OUTPUT = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\annotations\cvat_transverse_review"
)

IMAGES_OUT = OUTPUT / "obj_train_data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)

    selected = []

    # Find images containing transverse-crack class ID 1
    for label_path in sorted(LABEL_DIR.glob("*.txt")):

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        if any(
            line.split() and line.split()[0] == "1"
            for line in lines
        ):
            selected.append(label_path)

    print("Selected transverse-positive images:", len(selected))

    if len(selected) != 44:
        raise ValueError(
            f"Expected 44 images, found {len(selected)}. "
            "Check the source dataset before continuing."
        )

    class_counts = Counter()

    image_list = []

    for label_path in selected:

        image_path = next(
            (
                IMAGE_DIR / (label_path.stem + ext)
                for ext in IMAGE_EXTENSIONS
                if (IMAGE_DIR / (label_path.stem + ext)).exists()
            ),
            None
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Image missing: {label_path.stem}"
            )

        # Copy original image and corresponding label
        shutil.copy2(
            image_path,
            IMAGES_OUT / image_path.name
        )

        shutil.copy2(
            label_path,
            IMAGES_OUT / label_path.name
        )

        image_list.append(
            f"obj_train_data/{image_path.name}"
        )

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.split()

            if len(parts) == 5:
                class_counts[int(parts[0])] += 1

    # CVAT YOLO 1.1 metadata
    (OUTPUT / "obj.names").write_text(
        "longitudinal crack\n"
        "transverse crack\n"
        "alligator crack\n"
        "pothole\n",
        encoding="utf-8"
    )

    (OUTPUT / "obj.data").write_text(
        "classes = 4\n"
        "names = obj.names\n"
        "train = train.txt\n",
        encoding="utf-8"
    )

    (OUTPUT / "train.txt").write_text(
        "\n".join(image_list) + "\n",
        encoding="utf-8"
    )

    # Create ZIP archive
    zip_path = OUTPUT / "cvat_transverse_train_review.zip"

    with zipfile.ZipFile(
        zip_path, "w", zipfile.ZIP_DEFLATED
    ) as archive:

        for file in OUTPUT.rglob("*"):

            if not file.is_file() or file == zip_path:
                continue

            archive.write(
                file,
                file.relative_to(OUTPUT)
            )

    print("\n========== CVAT PACKAGE ==========")
    print("Images:", len(image_list))
    print("Class annotation counts:", dict(class_counts))
    print("ZIP:", zip_path.resolve())
    print("ZIP size MB:", round(zip_path.stat().st_size / 1024**2, 2))


if __name__ == "__main__":
    main()
