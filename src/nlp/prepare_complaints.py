
import pandas as pd
import re
from pathlib import Path

# Input dataset
input_file = Path(
    "data/raw/complaints/nyc_311_road_complaints.csv"
)

# Output directory
output_dir = Path("data/processed/complaints")
output_dir.mkdir(parents=True, exist_ok=True)

# Load original data
df = pd.read_csv(input_file, low_memory=False)

# Select useful columns
columns = [
    "unique_key",
    "created_date",
    "descriptor",
    "resolution_description",
    "status",
    "borough",
    "latitude",
    "longitude"
]

df = df[columns].copy()

# Remove missing descriptors
df = df.dropna(subset=["descriptor"])

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["complaint_text"] = df["descriptor"].apply(clean_text)

# Remove empty text
df = df[df["complaint_text"] != ""]

# Preserve category separately
df["complaint_category"] = df["descriptor"]

# Add source information
df["source_country"] = "USA"
df["source_city"] = "New York"

# Save processed dataset
output_file = output_dir / "nyc_complaints_cleaned.csv"
df.to_csv(output_file, index=False)

print("Cleaning completed.")
print("Records:", len(df))
print("Columns:", list(df.columns))
print("Saved to:", output_file)
print("\nSample:")
print(df[["complaint_text", "complaint_category"]].head(10))

