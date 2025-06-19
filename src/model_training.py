## This script responsible for Model Training and Experiment Tracking 

import os
import pandas as pd
import numpy as np  
import joblib

from  scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from src.logger import get_logger
from src.custom_exception import CustomException

from utils.common_functions import read_yaml, load_data
from config.paths_config import *
from config.model_params import *

import mlflow
import mlflow.sklearn 

# Initializing logger
logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, TRAIN_PATH , TEST_PATH, MODEL_OUTPUT_PATH):
        self.train_path = TRAIN_PATH
        self.test_path = TEST_PATH
        self.config_path = CONFIG_PATH
        self.model_output_path = MODEL_OUTPUT_PATH

        self.lightgbm_params = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


    def load_and_split_data(self):
        try:
            logger.info("Loading training and testing data...")
            # Load training and testing data
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)


            logger.info("Data loaded successfully. Splitting features and target variable...")

            # Split features and target variable
            X_train = train_data.drop(columns=['booking_status'])
            y_train = train_data['booking_status']
            X_test = test_data.drop(columns=['booking_status'])
            y_test = test_data['booking_status']

            logger.info("Data split into features and target variable successfully.")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error occurred while loading and splitting data: {e}")
            raise CustomException('Failed to load and split data.', e)
    
    def train_lgbm(self, X_train, y_train):

        try:
            logger.info("Initializing LightGBM model training...")

            # Initialize the LightGBM classifier
            model = LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("Starting Randomized Search CV for hyperparameter tuning...")

            # Perform Randomized Search CV for hyperparameter tuning
            random_search = RandomizedSearchCV(
                estimator= model,
                param_distributions=self.lightgbm_params,
                **self.random_search_params
            )

            # Fit the model
            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed successfully")

            # Get the best parameters: 
            best_params = random_search.best_params_
            logger.info(f"Best parameters found: {best_params}")

            # Get best model:
            best_model = random_search.best_estimator_
            logger.info("Best model obtained from Randomized Search CV.")
            logger.info("Model training completed successfully.")

            return best_model

        except Exception as e:
            logger.error(f"Error occurred during model training: {e}")
            raise CustomException('Model training failed.', e)  

    def evaluate_model(self, model, x_test, y_test):

        try:
            logger.info("Starting model evaluation...")

            # Make predictions on the test set
            y_pred = model.predict(x_test)

            # Calculate evaluation metrics
            accuracy_score_value = accuracy_score(y_test, y_pred)
            f1_score_value = f1_score(y_test, y_pred, average='weighted')
            recall_score_value = recall_score(y_test, y_pred, average='weighted')
            precision_score_value = precision_score(y_test, y_pred, average='weighted')
            logger.info(f"Model evaluation metrics:\n"
                        f"Accuracy:     {accuracy_score_value}\n"
                        f"F1 Score:     {f1_score_value}\n"
                        f"Recall:       {recall_score_value}\n"
                        f"Precision:    {precision_score_value}\n")
            logger.info("Model evaluation completed successfully.")

            return {
                'accuracy': accuracy_score_value,
                'f1_score': f1_score_value,
                'recall': recall_score_value,
                'precision': precision_score_value
            }

        except Exception as e:
            logger.error(f"Error occurred during model evaluation: {e}")
            raise CustomException('Model evaluation failed.', e)
    
    def save_model(self, model):
        try:
            logger.info(f"Saving the trained model ...")
            # Save the trained model using joblib
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True) 

            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully to {self.model_output_path}.")

        except Exception as e:
            logger.error(f"Error occurred while saving the model: {e}")
            raise CustomException('Model saving failed.', e)
        

    def run_pipeline(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training process with MLFLOW experimentation...")

                logger.info('Logging Training and Testing dataset')

                mlflow.log_artifact(self.train_path , artifact_path= 'datasets')
                mlflow.log_artifact(self.test_path, artifact_path= 'datasets')

                # Load and split data
                X_train, y_train, X_test, y_test = self.load_and_split_data()

                # Train LightGBM model
                model = self.train_lgbm(X_train, y_train)

                # Evaluate the model
                evaluation_metrics = self.evaluate_model(model, X_test, y_test)

                # Save the trained model
                self.save_model(model )

                logger.info('Logging Model info to MLFLOW')

                mlflow.log_artifact(self.model_output_path, artifact_path= 'trained_models')
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(evaluation_metrics)

                logger.info("Model training process completed successfully.")


        except Exception as e:
            logger.error(f"Error occurred during model training process: {e}")
            raise CustomException('Model training process failed.', e)
        

if __name__ == "__main__":
    try:
        logger.info("Starting the main block of model training...")
        
        # Initialize ModelTraining instance
        Model_training  = ModelTraining(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_OUTPUT_PATH)
        
        # Start the model training process
        Model_training.run_pipeline()

    except Exception as e:
        logger.error(f"An error occurred in the main block: {e}")
        raise CustomException('Main block execution failed.', e)