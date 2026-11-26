
import pandas as pd
from pathlib import Path

# Input
input_file = Path(
    "data/processed/vision/yolo_test_image_features.csv"
)

# Output
output_dir = Path("data/processed/priority")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "maintenance_priority_dataset.csv"

# Load vision features
df = pd.read_csv(input_file)

# Damage severity weights
weights = {
    "pothole_count": 3,
    "alligator crack_count": 3,
    "longitudinal crack_count": 2,
    "transverse crack_count": 2
}

# Calculate preliminary risk score
df["risk_score"] = 0

for column, weight in weights.items():
    df["risk_score"] += df[column] * weight

# Assign prototype priority labels
def assign_priority(score):
    if score == 0:
        return "Unassessed"
    elif score <= 2:
        return "Low"
    elif score <= 5:
        return "Medium"
    else:
        return "High"

df["priority_label"] = df["risk_score"].apply(
    assign_priority
)

# Document label origin
df["label_source"] = "Rule-based prototype"
df["manual_review_required"] = True

# Save dataset
df.to_csv(output_file, index=False)

# Display results
print("MAINTENANCE PRIORITY DATASET CREATED")
print("------------------------------------")

print("Total images:", len(df))

print("\nPriority distribution:")
print(df["priority_label"].value_counts())

print("\nRisk score summary:")
print(df["risk_score"].describe())

print("\nSaved to:", output_file)
