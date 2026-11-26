
from pathlib import Path
import cv2
import csv

DATASET = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class"
)

OUTPUT = Path(
    r"C:\College\M.tech\Datasets\RDD2022\Final_Annotation_Review"
)

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}

REVIEW_ITEMS = {
    "train": [
        "India_007811",
        "India_007689",
        "India_000905",
        "India_003722",
        "India_007769",
        "India_000283",
        "India_003507",
        "India_004468",
        "India_006779",
        "India_003969",
        "India_009863",
    ],
    "val": [
        "India_003245",
        "India_006811",
        "India_008677",
    ]
}

OUTPUT.mkdir(parents=True, exist_ok=True)

report = []

for split, filenames in REVIEW_ITEMS.items():

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    output_dir = OUTPUT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:

        image_path = image_dir / f"{name}.jpg"
        label_path = label_dir / f"{name}.txt"

        if not image_path.exists():
            print("IMAGE NOT FOUND:", image_path)
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("IMAGE READ ERROR:", image_path)
            continue

        height, width = image.shape[:2]

        if label_path.exists():
            lines = label_path.read_text().splitlines()
        else:
            lines = []

        annotation_count = 0

        for line in lines:

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            xc, yc, bw, bh = map(float, values[1:])

            if class_id not in CLASS_NAMES:
                continue

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            annotation_count += 1

        output_path = output_dir / f"{name}_review.jpg"

        cv2.imwrite(str(output_path), image)

        report.append({
            "split": split,
            "image": name + ".jpg",
            "annotation_count": annotation_count,
            "label_file_exists": label_path.exists(),
            "review_image": str(output_path)
        })

        print("Generated:", output_path)

with open(
    OUTPUT / "review_manifest.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "image",
            "annotation_count",
            "label_file_exists",
            "review_image"
        ]
    )

    writer.writeheader()
    writer.writerows(report)

print("\nFINAL REVIEW IMAGES GENERATED")
print("Images:", len(report))
print("Output:", OUTPUT)
