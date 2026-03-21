# ---------------------------
# IMPORT REQUIRED LIBRARIES
# ---------------------------
import os
import glob
import shutil
import re
import pandas as pd

# Kaggle API for dataset download
from kaggle.api.kaggle_api_extended import KaggleApi

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib  # For saving model and vectorizer


# ---------------------------
# CONFIGURATION
# ---------------------------
DATASET = "naserabdullahalam/phishing-email-dataset"
TMP_DIR = ".tmp_kaggle_download"   # Temporary folder (not committed)


# ---------------------------
# STEP 1: AUTHENTICATE KAGGLE
# ---------------------------
api = KaggleApi()
api.authenticate()


# ---------------------------
# STEP 2: PREPARE TEMP DIRECTORY
# ---------------------------
# Remove old temp folder if exists (clean state)
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

# Create fresh temp folder
os.makedirs(TMP_DIR, exist_ok=True)


# ---------------------------
# STEP 3: DOWNLOAD DATASET
# ---------------------------
api.dataset_download_files(DATASET, path=TMP_DIR, unzip=True)


# ---------------------------
# STEP 4: LOCATE CSV FILES
# ---------------------------
# Only select relevant dataset files
csv_files = [
    f for f in glob.glob(os.path.join(TMP_DIR, "*.csv"))
    if "CEAS_08" in os.path.basename(f)
]

print("CSV files downloaded:", csv_files)

if not csv_files:
    raise FileNotFoundError("No CSV found. Check dataset or Kaggle access.")


# ---------------------------
# STEP 5: LOAD DATA
# ---------------------------
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)


# ---------------------------
# DATA QUALITY CHECK (BEFORE CLEANING)
# ---------------------------
print("\n--- Data Quality BEFORE Cleaning ---")
print("Duplicate rows:", df.duplicated().sum())
print("Null counts:\n", df.isna().sum())


# ---------------------------
# STEP 6: CLEANING
# ---------------------------

# 6.1 Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print("\nDuplicates removed:", before - after)

# 6.2 Fill missing values
df["subject"] = df["subject"].fillna("")
df["receiver"] = df["receiver"].fillna("unknown")


# ---------------------------
# DATA QUALITY CHECK (AFTER CLEANING)
# ---------------------------
print("\n--- Data Quality AFTER Cleaning ---")
print("Duplicate rows:", df.duplicated().sum())
print("Null counts:\n", df.isna().sum())


# ---------------------------
# STEP 7: TEXT NORMALIZATION
# ---------------------------
def normalize_text(text):
    text = text.lower()                         # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
    text = re.sub(r'\s+', ' ', text)            # Remove extra spaces
    return text

# Apply normalization
df["subject"] = df["subject"].apply(normalize_text)
df["body"] = df["body"].apply(normalize_text)

# Remove leading/trailing whitespace
df["subject"] = df["subject"].str.strip()
df["body"] = df["body"].str.strip()


# ---------------------------
# STEP 8: COMBINE TEXT FIELDS
# ---------------------------
print("\n--- BEFORE Combining ---")
print(df[["subject", "body"]].head(3))

df["email_text"] = df["subject"] + " " + df["body"]

print("\n--- AFTER Combining ---")
print(df[["email_text"]].head(3))


# ---------------------------
# STEP 9: LABEL CONVERSION
# ---------------------------
print("\n--- BEFORE Label Conversion ---")
print(df["label"].head(5))

df["label"] = df["label"].astype(int)

print("\n--- AFTER Label Conversion ---")
print(df["label"].head(5))


# ---------------------------
# STEP 10: METADATA FEATURES
# ---------------------------
print("\n--- Creating Metadata Features ---")

df["subject_length"] = df["subject"].apply(len)
df["body_length"] = df["body"].apply(len)
df["url_count"] = df["urls"]  # already numeric

print(df[["subject_length", "body_length", "url_count"]].head(5))


# ---------------------------
# STEP 11: PHISHING FEATURES
# ---------------------------
print("\n--- Creating Phishing Features ---")

phishing_keywords = [
    "urgent", "verify", "click", "login", "password",
    "account", "bank", "security", "update", "confirm"
]

# Count keyword occurrences
df["phishing_keyword_count"] = df["email_text"].apply(
    lambda x: sum(x.count(word) for word in phishing_keywords)
)

# Count uppercase words (signal urgency)
df["uppercase_count"] = df["email_text"].apply(
    lambda x: sum(1 for word in x.split() if word.isupper())
)

# Count digits (common in malicious links)
df["digit_count"] = df["email_text"].str.count(r"\d")


# ---------------------------
# STEP 12: TRAIN / TEST SPLIT
# ---------------------------
X = df["email_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------
# STEP 13: TF-IDF FEATURE EXTRACTION
# ---------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

# Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data
X_test_tfidf = tfidf.transform(X_test)


# ---------------------------
# STEP 14: COMBINE ALL FEATURES
# ---------------------------
meta_features = [
    "subject_length",
    "body_length",
    "url_count",
    "phishing_keyword_count",
    "uppercase_count",
    "digit_count"
]

# Extract metadata aligned with indices
X_train_meta = df.loc[X_train.index, meta_features]
X_test_meta = df.loc[X_test.index, meta_features]

# Convert to sparse format
X_train_meta_sparse = csr_matrix(X_train_meta.values)
X_test_meta_sparse = csr_matrix(X_test_meta.values)

# Combine TF-IDF + metadata
X_train_final = hstack([X_train_tfidf, X_train_meta_sparse])
X_test_final = hstack([X_test_tfidf, X_test_meta_sparse])


# ---------------------------
# STEP 15: MODEL TRAINING
# ---------------------------
print("\n--- Model Training ---")

model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train_final, y_train)


# ---------------------------
# STEP 16: PREDICTION & EVALUATION
# ---------------------------
y_pred = model.predict(X_test_final)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ---------------------------
# STEP 17: SAVE MODEL
# ---------------------------
joblib.dump(model, "phishing_model_v1.pkl")
joblib.dump(tfidf, "tfidf_vectorizer_v1.pkl")

print("Model and vectorizer saved successfully.")


# ---------------------------
# STEP 18: CLEANUP TEMP FILES
# ---------------------------
shutil.rmtree(TMP_DIR)
print("Temporary dataset files deleted.")