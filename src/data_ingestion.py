## ........ Importing necessary libraries ........
import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import * # All paths are imported from paths_config.py
from utils.common_functions import read_yaml


# Creating the instance for logger
logger = get_logger(__name__)

# .......................................Class for Data Ingestion..................................
class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']                          # read data_ingestion config from config.yaml
        self.bucket_name = self.config['bucket_name']                   # Google Cloud Storage bucket name
        self.file_name = self.config['bucket_file_name']                # Name of the file in the bucket
        self.train_test_ratio = self.config['train_ratio']              # Ratio for train-test split

        # Paths for local storage 
        # Make RAW directory if it doesn't exist
        if not os.path.exists(RAW_DIR):
            os.makedirs(RAW_DIR)

        ## Logging
        logger.info(f"Initialized DataIngestion with bucket: {self.bucket_name}, file: {self.file_name}, train-test ratio: {self.train_test_ratio}")



    def download_data_from_gcp(self):
        """
        Downloads data from Google Cloud Storage bucket and saves it as a CSV file.
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            # Download the file to local storage
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Raw data downloaded successfully from GCP bucket: {self.bucket_name} to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error occurred while downloading data from GCP: {e}")
            raise CustomException('Failed to download data from GCP.', e)
    
    def split_data(self):
        """
        Splits the raw data into training and testing datasets.
        """
        try:
            # Load the raw data
            df = pd.read_csv(RAW_FILE_PATH)
            logger.info(f"Raw data loaded successfully from {RAW_FILE_PATH} for Splitting into train and test datasets.")

            # Split the data into train and test sets
            train_df, test_df = train_test_split(df, test_size=1-self.train_test_ratio, random_state=42)

            # Save the train and test datasets
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Data split successfully: Train data saved to {TRAIN_FILE_PATH}, Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error occurred while splitting data")
            raise CustomException('Failed to split data.', e)
        
    def run(self):
        """
        Runs the data ingestion process: downloads data from GCP and splits it into train and test datasets.
        """
        try:
            logger.info("Starting data ingestion process...")
            self.download_data_from_gcp()
            self.split_data()
            logger.info("Data ingestion process completed successfully.")

        except CustomException as ce:
            logger.error(f"Custom Excepetion: {str(ce)}")
        
        finally:
            logger.info("Data ingestion process finished.")
            return {
                "train_file_path": TRAIN_FILE_PATH,
                "test_file_path": TEST_FILE_PATH
            }
        

if __name__ == "__main__":
    # Load configuration from YAML file
    config = read_yaml(CONFIG_PATH)

    # Create an instance of DataIngestion
    data_ingestion = DataIngestion(config)

    # Run the data ingestion process
    data_ingestion.run()