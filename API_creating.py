import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Intialize API
api = KaggleApi()
api.authenticate()

# Dataset slug
DATASET = "zaczinho/phishing-mail-dataset-subject-body"
OUT_DIR = "data"

os.makedirs(OUT_DIR, exist_ok=True)

# Download & unzip
api.dataset_download_files(
    DATASET,
    path=OUT_DIR,
    unzip=True
)

print("Dataset downloaded to: ", OUT_DIR)