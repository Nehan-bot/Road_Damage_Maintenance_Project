
from pathlib import Path

ROOT = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India"
)

FILENAMES = [
    "India_007011",
    "India_008170",
    "India_003763",
    "India_008489",
    "India_007424",
    "India_009769"
]

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Other corruption",
    4: "Pothole"
}

for filename in FILENAMES:

    print("\n" + "=" * 55)
    print(filename)

    found = False

    for split in ["train", "val", "test"]:

        label_path = (
            ROOT / split / "labels" / f"{filename}.txt"
        )

        if not label_path.exists():
            continue

        found = True
        content = label_path.read_text().strip()

        print("Split:", split)

        if not content:
            print("Original annotation: EMPTY")
            continue

        for line in content.splitlines():

            values = line.split()

            if len(values) != 5:
                print("Invalid line:", line)
                continue

            class_id = int(values[0])

            print(
                "Class ID:", class_id,
                "| Category:", CLASS_NAMES.get(
                    class_id, "Unknown"
                ),
                "| Coordinates:", " ".join(values[1:])
            )

    if not found:
        print("Original annotation file not found.")
