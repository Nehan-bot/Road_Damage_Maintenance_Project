
from pathlib import Path
import zipfile
import shutil

# Original, unaugmented four-class dataset
SOURCE = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class"
)

# New corrected dataset - keep the old datasets untouched
OUTPUT = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India_4Class_Corrected\RDD2022_India_4Class_cvat_Corrected"
)

# CVAT export ZIP - change this to your actual local ZIP location
CVAT_ZIP = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\annotations\cvat_transverse_corrected\corrected_annotations.zip"
)

def main():

    source_images = SOURCE / "train" / "images"
    source_labels = SOURCE / "train" / "labels"

    output_images = OUTPUT / "train" / "images"
    output_labels = OUTPUT / "train" / "labels"

    if not source_images.exists():
        raise FileNotFoundError(source_images)

    if not source_labels.exists():
        raise FileNotFoundError(source_labels)

    if not CVAT_ZIP.exists():
        raise FileNotFoundError(CVAT_ZIP)

    # Prevent accidental overwriting
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "Rename or remove it only after checking its contents."
        )

    print("Copying original training images and labels...")

    shutil.copytree(source_images, output_images)
    shutil.copytree(source_labels, output_labels)

    print("Applying CVAT corrections...")

    corrected = 0

    with zipfile.ZipFile(CVAT_ZIP, "r") as archive:

        for member in archive.namelist():

            path = Path(member)

            if (
                path.parent.name != "obj_train_data"
                or path.suffix.lower() != ".txt"
            ):
                continue

            destination = output_labels / path.name

            if not destination.exists():
                raise FileNotFoundError(
                    f"Original label not found: {path.name}"
                )

            content = archive.read(member)

            # Verify that the exported annotation is readable
            text = content.decode("utf-8-sig")

            destination.write_text(
                text,
                encoding="utf-8"
            )

            corrected += 1

    if corrected != 44:
        raise ValueError(
            f"Expected 44 corrected labels, found {corrected}"
        )

    print("\n========== INTEGRATION SUMMARY ==========")
    print("Original training images:", len(list(output_images.iterdir())))
    print("Corrected labels integrated:", corrected)
    print("Output dataset:", OUTPUT)
    print("\nIntegration completed.")

if __name__ == "__main__":
    main()
