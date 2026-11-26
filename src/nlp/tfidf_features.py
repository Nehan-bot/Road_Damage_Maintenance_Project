
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Input and output paths
input_file = Path(
    "data/processed/complaints/nyc_complaints_cleaned.csv"
)

output_dir = Path("data/processed/complaints")
output_dir.mkdir(parents=True, exist_ok=True)

# Load cleaned complaints
df = pd.read_csv(input_file)

# Initialize TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2
)

# Transform complaint text
tfidf_matrix = vectorizer.fit_transform(
    df["complaint_text"].fillna("")
)

# Save TF-IDF features
feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)

tfidf_file = output_dir / "nyc_tfidf_features.csv"
tfidf_df.to_csv(tfidf_file, index=False)

# Save fitted vectorizer
vectorizer_file = output_dir / "tfidf_vectorizer.joblib"
joblib.dump(vectorizer, vectorizer_file)

print("TF-IDF extraction completed.")
print("Number of records:", tfidf_matrix.shape[0])
print("Number of features:", tfidf_matrix.shape[1])
print("Feature examples:", list(feature_names[:20]))
print("Feature matrix shape:", tfidf_matrix.shape)
print("Saved:", tfidf_file)
print("Saved:", vectorizer_file)
