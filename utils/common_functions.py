## >>>>>>>>>  MAKING UTILITY FUNCTIONS >>>>>>
import os 
import pandas 
from src.logger import get_logger
from src.custom_exception import CustomException    
import yaml
import pandas  as  pd

## Creating the instance for logger
logger = get_logger(__name__)

##........ FUNCTION TO READ YAML FILE .........>>
def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Succesfully read the YAML file")
            return config
    
    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read YAMl file" , e)



## ........ FUNCTION TO READ CSV FILE .........>>

def load_data(file_path):
    try:
        logger.info(f"Loading data from {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File Not Found! @ {file_path}')

        data = pd.read_csv(file_path)

        logger.info(f"Data loaded successfully from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error occurred while loading data from {file_path}")
        raise CustomException('Failed to load data.', e)
    

    