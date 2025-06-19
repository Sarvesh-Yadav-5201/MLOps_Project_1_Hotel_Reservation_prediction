## Complete Training Pipeline
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining    
from utils.common_functions import read_yaml
from config.paths_config  import *



if __name__ == "__main__":

    ## Data Ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    ## Data Preprocessing
    data_processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_PATH, PROCESSED_DIR)
    data_processor.process_data()

    ## Model Training
    Model_training  = ModelTraining(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_OUTPUT_PATH)
    Model_training.run_pipeline()


    