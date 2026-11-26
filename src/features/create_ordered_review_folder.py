
from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "priority"
    / "final_review_sample_50_assessed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "priority"
    / "ordered_review_images"
)

if not CSV_FILE.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

required = ["image_name", "image_path"]

for col in required:
    if col not in df.columns:
        raise ValueError(
            f"Missing column: {col}\n"
            f"Available columns: {df.columns.tolist()}"
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

index_rows = []
missing = []

for number, row in enumerate(df.to_dict("records"), start=1):
    image_name = str(row["image_name"]).strip()
    relative_path = str(row["image_path"]).strip()

    source = PROJECT_ROOT / Path(relative_path)

    if not source.exists():
        missing.append(image_name)
        continue

    destination_name = f"{number:02d}_{image_name}"
    destination = OUTPUT_DIR / destination_name

    shutil.copy2(source, destination)

    index_rows.append({
        "sequence": number,
        "image_name": image_name,
        "review_image": destination_name,
        "provisional_visual_assessment":
            row.get("provisional_visual_assessment", ""),
        "independent_human_verified_label": ""
    })

pd.DataFrame(index_rows).to_csv(
    OUTPUT_DIR / "review_index.csv",
    index=False
)

print("\nREVIEW FOLDER CREATED")
print("Folder:", OUTPUT_DIR)
print("Images copied:", len(index_rows))
print("Missing images:", len(missing))

if missing:
    print("Missing filenames:")
    for name in missing:
        print(name)
