import os
import glob
import shutil
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# DATASET = "zaczinho/phishing-mail-dataset-subject-body"
DATASET = "naserabdullahalam/phishing-email-dataset"
TMP_DIR = ".tmp_kaggle_download"   # temporary folder (not committed)

# authenticate with Kaggle
api = KaggleApi()
api.authenticate()

# 1) fresh temp directory every run
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

# 2) download & unzip into temp
api.dataset_download_files(DATASET, path=TMP_DIR, unzip=True)

# 3) find the CSV(s) that came from this dataset
# csv_files = glob.glob(os.path.join(TMP_DIR, "*.csv"))
csv_files = [f for f in glob.glob(os.path.join(TMP_DIR, "*.csv"))
             if "CEAS_08" in os.path.basename(f)]
print("CSV files downloaded:", csv_files)

if not csv_files:
    raise FileNotFoundError("No CSV found after download/unzip. Check dataset contents or Kaggle access rules.")

# 4) load into memory (DataFrame)
# df = pd.read_csv(csv_files[0])
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)
print("Loaded:", csv_files[0])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(10))

# 5) delete temp files to comply with 'no dataset file stored'
shutil.rmtree(TMP_DIR)
print("Temporary dataset files deleted.")

