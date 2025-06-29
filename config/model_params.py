## This file contains all the model parameters and configurations and randomnized search CV parameters for hyperparameter tuning.

from scipy.stats import randint, uniform

# Model parameters for LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators':314,             # Number of boosting iterations
    'max_depth': 23,                    # Maximum depth of the tree
    'learning_rate': 0.069,            # Learning rate
    'num_leaves': 94,                 # Number of leaves in one tree
    'boosting_type': ['gbdt'],      # Boosting type
}



RANDOM_SEARCH_PARAMS = {
    'n_iter': 5,                                    # Number of iterations for random search
    'cv': 5,                                        # Number of cross-validation folds
    'verbose': 1,                                   # Verbosity level
    'random_state': 42,                             # Random seed for reproducibility
    'scoring': 'f1',                                # Scoring metric for model evaluation
    'refit': True,                                  # Refit the model with the best parameters
    'n_jobs': -1,                                   # Use all available CPU cores
}
