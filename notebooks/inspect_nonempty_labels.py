
from pathlib import Path
import random

DATASET_DIR = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India"
)

LABEL_DIR = DATASET_DIR / "train" / "labels"

random.seed(42)

label_files = [
    p for p in LABEL_DIR.glob("*.txt")
    if p.stat().st_size > 0
]

samples = random.sample(label_files, min(10, len(label_files)))

print("Non-empty label files available:", len(label_files))

for label_path in samples:

    print("\nImage:", label_path.stem + ".jpg")
    print("Label file:", label_path.name)

    lines = label_path.read_text().strip().splitlines()

    for line in lines[:5]:
        print(" ", line)

    if len(lines) > 5:
        print("  ... additional annotations:", len(lines) - 5)
