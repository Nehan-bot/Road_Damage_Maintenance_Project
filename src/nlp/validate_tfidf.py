
import pandas as pd
import joblib
from pathlib import Path

data_dir = Path("data/processed/complaints")

features_file = data_dir / "nyc_tfidf_features.csv"
vectorizer_file = data_dir / "tfidf_vectorizer.joblib"
cleaned_file = data_dir / "nyc_complaints_cleaned.csv"

features = pd.read_csv(features_file)
complaints = pd.read_csv(cleaned_file)
vectorizer = joblib.load(vectorizer_file)

print("TF-IDF VALIDATION")
print("-----------------")

print("Complaint records:", len(complaints))
print("Feature matrix shape:", features.shape)
print("Vectorizer vocabulary:", len(vectorizer.vocabulary_))

print("Missing complaint text:",
      complaints["complaint_text"].isna().sum())

print("Duplicate complaint descriptions:",
      complaints["complaint_text"].duplicated().sum())

print("Non-zero TF-IDF values:",
      (features != 0).sum().sum())

print("Validation completed.")
