
import pandas as pd
import shutil
from pathlib import Path

project = Path(".")

csv_file = (
    project / "data/processed/priority/manual_review_sample.csv"
)

image_dir = (
    project / "data/raw/road_images/"
    "RDD2022_India_4Class_Corrected_Balanced600/test/images"
)

output_dir = (
    project / "data/processed/priority/review_images"
)

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_file)

copied = 0
missing = []

for _, row in df.iterrows():

    image_name = row["image_name"]
    source = image_dir / image_name

    if source.exists():

        destination = output_dir / image_name
        shutil.copy2(source, destination)
        copied += 1

    else:
        missing.append(image_name)

print("REVIEW IMAGE PREPARATION")
print("------------------------")
print("Images selected:", len(df))
print("Images copied:", copied)
print("Missing images:", len(missing))

if missing:
    print("Missing filenames:", missing)

print("Output folder:", output_dir)
