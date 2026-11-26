
import pandas as pd
from pathlib import Path

file_path = Path(
    "data/processed/vision/yolo_test_image_features.csv"
)

df = pd.read_csv(file_path)

print("VISION FEATURE VALIDATION")
print("-------------------------")

print("Number of images:", len(df))
print("Number of columns:", len(df.columns))

print("\nMissing values:")
print(df.isna().sum())

print("\nTotal detections:")
print(df["total_detections"].sum())

print("\nDamage count summary:")
count_columns = [
    col for col in df.columns
    if col.endswith("_count")
]

print(df[count_columns].sum())

print("\nImages with no detections:")
print((df["total_detections"] == 0).sum())

print("\nSample records:")
print(df.head().to_string(index=False))

print("\nValidation completed.")
