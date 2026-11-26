
import pandas as pd

file_path = "data/raw/complaints/nyc_311_road_complaints.csv"

df = pd.read_csv(file_path, low_memory=False)

print("\nDATASET SHAPE:", df.shape)

print("\nCOMPLAINT TYPE COUNTS:")
print(df["complaint_type"].value_counts(dropna=False))

print("\nDESCRIPTOR COUNTS:")
print(df["descriptor"].value_counts(dropna=False).head(30))

print("\nDESCRIPTOR_2 COUNTS:")
print(df["descriptor_2"].value_counts(dropna=False).head(20))

print("\nRESOLUTION DESCRIPTION SAMPLE:")
print(
    df["resolution_description"]
    .dropna()
    .head(10)
    .to_string(index=False)
)

print("\nMISSING VALUES:")
print(df[[
    "complaint_type",
    "descriptor",
    "descriptor_2",
    "resolution_description"
]].isna().sum())
