
import pandas as pd
from pathlib import Path

input_file = Path(
    "data/processed/priority/maintenance_priority_dataset.csv"
)

output_dir = Path("data/processed/priority")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

sample_sizes = {
    "High": 15,
    "Medium": 15,
    "Low": 10,
    "Unassessed": 10
}

samples = []

for priority, size in sample_sizes.items():

    group = df[df["priority_label"] == priority]

    selected = group.sample(
        n=min(size, len(group)),
        random_state=42
    ).copy()

    samples.append(selected)

review_df = pd.concat(samples, ignore_index=True)

# Add manual review fields
review_df["reviewed_priority"] = ""
review_df["reviewer_notes"] = ""
review_df["verification_status"] = "Pending"

output_file = output_dir / "manual_review_sample.csv"

review_df.to_csv(output_file, index=False)

print("MANUAL REVIEW SAMPLE CREATED")
print("----------------------------")
print("Total selected:", len(review_df))
print("\nPriority distribution:")
print(review_df["priority_label"].value_counts())
print("\nSaved to:", output_file)
