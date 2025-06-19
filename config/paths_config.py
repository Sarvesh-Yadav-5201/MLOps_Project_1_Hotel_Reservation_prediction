import os 

##...........DATA INGESTION......................##


RAW_DIR = "artifacts/raw"

RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

## CONFIIG DATA
CONFIG_PATH = "config/config.yaml"


##...........DATA PROCESSING......................##
PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DIR, "test_processed.csv")

## ...........MODEL TRAINING ......................##

MODEL_OUTPUT_PATH = "artifacts/model/lgbm_model.pkl"
