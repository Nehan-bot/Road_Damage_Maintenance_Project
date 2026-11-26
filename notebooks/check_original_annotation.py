
from pathlib import Path

FILENAME = "India_002311"

DATASETS = {
    "Original India dataset": Path(
        r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India"
    ),
    "Final four-class dataset": Path(
        r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India"
    )
}

for dataset_name, root in DATASETS.items():

    print(f"\n{'=' * 55}")
    print(dataset_name)
    print("=" * 55)

    if not root.exists():
        print("Dataset folder not found:", root)
        continue

    image_files = list(root.rglob(FILENAME + ".*"))

    for image_path in image_files:

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        label_path = (
            image_path.parent.parent
            / "labels"
            / (FILENAME + ".txt")
        )

        print("\nImage:", image_path)
        print("Label:", label_path)

        if not label_path.exists():
            print("LABEL FILE NOT FOUND")
            continue

        content = label_path.read_text().strip()

        if content:
            print("ANNOTATIONS:")
            print(content)
        else:
            print("EMPTY LABEL FILE")

print("\nInvestigation completed.")
