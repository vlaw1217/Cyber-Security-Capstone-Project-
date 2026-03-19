import os
import glob 
import shutil 
import re
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET = "naserabdullahalam/phishing-email-dataset"
TMP_DIR = ".tmp_kaggle_download"   # Temporary folder (not committed)

# Authenticate with Kaggle
api = KaggleApi()
api.authenticate()

# 1) Fresh temp directory every run
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

# 2) Download & unzip into temp
api.dataset_download_files(DATASET, path=TMP_DIR, unzip=True)

# 3) Find the CSV(s) that came from this dataset
csv_files = [f for f in glob.glob(os.path.join(TMP_DIR, "*.csv"))
             if "CEAS_08" in os.path.basename(f)]
print("CSV files downloaded:", csv_files)

if not csv_files:
    raise FileNotFoundError("No CSV found after download/unzip. Check dataset contents or Kaggle access rules.")

# 4) Load into memory (DataFrame)
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# --- Data quality BEFORE cleaning ---
print("\n--- Data Quality BEFORE Cleaning ---")
print("Duplicate rows:", df.duplicated().sum())
print("Null counts:\n", df.isna().sum())

# 5) Cleaning Step 1: Remove duplicates and Fill missing values
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print("\nDuplicates removed:", before - after)

# Fill missing values 
df["subject"] = df["subject"].fillna("")
df["receiver"] = df["receiver"].fillna("unknown")

# Data quality check
print("\n--- Data Quality AFTER Cleaning ---")
print("Duplicate rows:", df.duplicated().sum())
print("Null counts:\n", df.isna().sum())

# 7) Cleaning Step 2: Normalize text
def normalize_text(text):
    text = text.lower()                 # convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text)    # remove extra spaces
    return text

# Apply normalization
df['subject'] = df['subject'].apply(normalize_text)
df['body'] = df['body'].apply(normalize_text)

# Removes leading/trailing spaces that remain after regex cleaning
df["subject"] = df["subject"].str.strip()
df["body"] = df["body"].str.strip()

# 8) Cleaning Step 3: Combine subject and body
print("\n--- BEFORE Combining ---")
print(df[["subject", "body"]].head(3))

df["email_text"] = df["subject"] + " " + df["body"]

print("\n--- AFTER Combining ---")
print(df[["email_text"]].head(3))
print("\nText normalization completed.\n")

# 9) Cleaning Step 4: Convert label to integer
print("\n--- BEFORE Label Conversion ---")
print(df["label"].head(5))

# Convert label to integer
df["label"] = df["label"].astype(int)

print("\n--- AFTER Label Conversion ---")
print(df["label"].head(5))

# 10) Cleaning Step 5: Simple metadata features
print("\n--- Creating Metadata Features ---")

# subject length
df["subject_length"] = df["subject"].apply(len)

# body length
df["body_length"] = df["body"].apply(len)

# URL count (urls column already contains numeric values)
df["url_count"] = df["urls"]

print("\nMetadata feature preview:")
print(df[["subject_length", "body_length", "url_count"]].head(5))

# 11) Step 6: Train/Test Split 
X = df['email_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 12) Step 7: TF-IDF
print("\nTF-IDF Feature Extraction:")
# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,      # limit vocabulary size
    ngram_range=(1, 2),     # unigrams + bigrams
    stop_words='english'    # remove common words
)

# Fit on training data ONLY (important!)
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data (DO NOT fit again)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Train shape:", X_train_tfidf.shape)
print("TF-IDF Test shape:", X_test_tfidf.shape)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape, "\n")

# Empty / whitespace-only
for col in ["subject", "body"]:
    if col in df.columns:
        empty = df[col].astype(str).str.strip().eq("").sum()
        print(f"Empty {col}:", empty)
        
print("\nLabel value counts:\n", df["label"].value_counts(dropna=False) if "label" in df.columns else "No label column")

print("Loaded:", csv_files[0])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(10))

# 6) Delete temp files to comply with 'no dataset file stored'
shutil.rmtree(TMP_DIR)
print("Temporary dataset files deleted.")

