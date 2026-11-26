
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

params = {
    "$where": "complaint_type = 'Street Condition'",
    "$limit": 50000,
    "$order": "created_date DESC"
}

output_dir = Path("data/raw/complaints")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "nyc_311_road_complaints.csv"

print("Downloading NYC road complaints...")

response = requests.get(url, params=params, timeout=120)
response.raise_for_status()

df = pd.read_csv(StringIO(response.text))

df.to_csv(output_file, index=False)

print("\nDownload successful!")
print("Number of records:", len(df))
print("Columns:", list(df.columns))
print("Saved to:", output_file)
print("\nSample records:")
print(df.head(5).to_string(index=False))
