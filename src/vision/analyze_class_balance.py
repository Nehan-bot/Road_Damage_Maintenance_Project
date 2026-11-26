
from pathlib import Path
from collections import Counter, defaultdict

# Original four-class training labels
LABEL_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class\train\labels"
)

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}


def main():

    if not LABEL_DIR.exists():
        raise FileNotFoundError(LABEL_DIR)

    instance_counts = Counter()
    image_counts = Counter()
    images_per_class = defaultdict(list)

    label_files = list(LABEL_DIR.glob("*.txt"))

    for label_file in label_files:

        classes_in_image = set()

        with open(label_file, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                if class_id not in CLASS_NAMES:
                    continue

                instance_counts[class_id] += 1
                classes_in_image.add(class_id)

        for class_id in classes_in_image:
            image_counts[class_id] += 1
            images_per_class[class_id].append(label_file.stem)

    print("\n========== TRAINING CLASS DISTRIBUTION ==========")

    print("Total label files:", len(label_files))

    for class_id, class_name in CLASS_NAMES.items():

        print(f"\n{class_name}")
        print("Annotation instances:", instance_counts[class_id])
        print("Unique images:", image_counts[class_id])

    print("\n========== TRANSVERSE CRACK DETAILS ==========")

    transverse_images = images_per_class[1]

    print("Transverse crack images:", len(transverse_images))

    for image_name in transverse_images:
        print(image_name)

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
