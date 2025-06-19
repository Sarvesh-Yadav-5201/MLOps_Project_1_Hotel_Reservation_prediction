## Importing necessary libraries
import os 
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml,load_data
from config.paths_config import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

## initializing logger
logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path , test_path, config_path, processed_dir):

        self.train_path = train_path
        self.test_path = test_path      
        self.config_path = config_path
        self.processed_dir = processed_dir  

        ## Read config file
        self.config = read_yaml(self.config_path)

        ## Create directory if it does not exist    
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing...")
            
            # Droping unnecessary columns
            logger.info("Dropping unnecessary columns...")
            df.drop(columns=["Booking_ID"], inplace=True)

            ## Droping Duplicates
            logger.info("Dropping duplicate rows...")
            df.drop_duplicates(inplace=True)

            ## Defining Categorical and Numerical Columns
            logger.info("Defining categorical and numerical columns...")
            cat_col = self.config['data_processing']['categorilcal_columns']
            num_col = self.config['data_processing']['numerical_columns']

            ## Label Encoding Categorical Columns
            logger.info("Label encoding categorical columns...")
            Label_encoder = LabelEncoder()
            mappings = {}
            for col in cat_col:
                df[col] = Label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(Label_encoder.classes_, Label_encoder.transform(Label_encoder.classes_))}

            logger.info(f'Label encoding mappings: {mappings}')

            ## Skewness Handling on numerical columns
            logger.info("Handling skewness in numerical columns...")
            skewness_threshold = self.config['data_processing']['skewness_threshold']

            skewness = df[num_col].apply(lambda x: x.skew())

            ## Applying log transformation to skewed numerical columns
            for col in skewness[skewness> skewness_threshold].index:
                df[col] = np.log1p(df[col])

            logger.info('All preprocessing steps completed successfully.')  

            return df
                

        except Exception as e:
            logger.error("Error occurred during data preprocessing.", e)
            raise CustomException('Data preprocessing failed.', e)


    def balance_data(self , df):
        try:
            logger.info("Starting data balancing...")

            ## ......... USING SMOTE FOR BALANCING THE DATA .......... ##
            ## Splitting data into features and target variable
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            
            ## Balancing the data using SMOTE
            logger.info("Applying SMOTE for data balancing...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Combining resampled features and target variable into a single DataFrame
            balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["booking_status"])], axis=1)
            logger.info("SMOTE data balancing completed successfully.")

            return balanced_df
        
            # ## ......... MANUAL OVERSAMPLING FOR BALANCING THE DATA .......... ##

            # ## seperating the majority and minority classes
            # logger.info("Separating majority and minority classes...")
            # majority_class = df[df['Booking_Status'] == df['Booking_Status'].value_counts().idxmax()]
            # minority_class = df[df['Booking_Status'] == df['Booking_Status'].value_counts().idxmin()]

            # # oversampling minority class to match majority class
            # minority_oversampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)

            # # combining majority and oversampled minority class
            # balanced_df = pd.concat([majority_class, minority_oversampled], axis=0)
            # logger.info("Data balancing completed successfully.")

            # return balanced_df

        except Exception as e:
            logger.error("Error occurred while balancing data.", e)
            raise CustomException('Data balancing failed.', e)
        

    def feature_selection(self, balanced_df):
        try:
            logger.info("Starting feature selection...")
            # Splitting data into features and target variable
            X = balanced_df.drop(columns=["booking_status"])
            y = balanced_df["booking_status"]

            # Initializing RandomForestClassifier
            logger.info("Initializing RandomForestClassifier for feature selection...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Getting feature importances
            logger.info("Calculating feature importances...")
            importances = rf.feature_importances_
            feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

            # Sorting features by importance
            feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

            # Selecting top features based on importance
            n_features = self.config['data_processing']['number_of_selected_features']
            top_n = feature_importances.head(n_features)['Feature'].tolist()

            # Filtering the DataFrame to keep only the top features + target variable
            logger.info("Filtering DataFrame to keep only top features...")

            selected_df = balanced_df[top_n + ['booking_status']]
            logger.info("Feature selection completed successfully.")
            return selected_df

        except Exception as e:
            logger.error("Error occurred during feature selection.", e)
            raise CustomException('Feature selection failed.', e)

    def  save_data (self, processed_df, file_path):
        try:
            logger.info("Starting data saving in processed folder...")
            # Saving the processed DataFrame to the specified file path
            processed_df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}.")


        except Exception as e:
            logger.error("Error occurred while saving data.", e)
            raise CustomException('Data saving failed.', e) 

    def process_data(self):
        try: 
            logger.info("Starting complete data preprocessing pipeline...")

            # Load train and test data
            logger.info(f"Loading train data  and test data ...")
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)

            ## Preprocess train data
            train_df = self.preprocess_data(train_data)
            test_df = self.preprocess_data(test_data)

            ## Balancing the train data
            train_data = self.balance_data(train_df)

            ## Feature selection on balanced train data
            selected_train_data = self.feature_selection(train_data)
            test_df = test_df[selected_train_data.columns]

            ## Saving processed train and test data
            self.save_data(selected_train_data,PROCESSED_TRAIN_PATH)
            self.save_data(test_df, PROCESSED_TEST_PATH)
            logger.info("Data preprocessing pipeline completed successfully.")


        except Exception as e:
            logger.error("Error occurred during data preprocessing pipeline.", e)
            raise CustomException('Data preprocessing pipeline failed.', e)



if __name__ == "__main__":
    try:
        logger.info("Starting the main block of data preprocessing...")

        # Create an instance of DataProcessor
        data_processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_PATH, PROCESSED_DIR)

        # Call the preprocess_data method to start the preprocessing pipeline
        data_processor.process_data()

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error("Error occurred in the main block.", e)
        raise CustomException('Main block execution failed.', e)