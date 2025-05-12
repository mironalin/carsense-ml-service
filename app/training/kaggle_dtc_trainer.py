#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to train a supervised model to predict potential DTCs based on sensor data,
using heuristically generated labels from the anomaly detection step.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example classifier
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, accuracy_score, f1_score, precision_score, recall_score # ADDED metrics
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer # Import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier # Import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV # Import RandomizedSearchCV
from scipy.stats import randint, uniform # Import distributions for param grid
from joblib import dump, load # Import load as well for potentially loading later
import argparse
import os
import logging
import sys
from typing import Dict, Any, Tuple, List
import joblib # ADDED IMPORT

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    LGBMClassifier = None
    lightgbm_available = False
    logging.warning("LightGBM not installed. LGBMClassifier will not be available.")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define feature lists (mirroring anomaly_detection.py for consistency)
CORE_PIDS_FOR_SUPERVISED = [
    "ENGINE_RPM",
    "COOLANT_TEMPERATURE",
    "INTAKE_MANIFOLD_ABSOLUTE_PRESSURE",
    "INTAKE_AIR_TEMPERATURE",
    "MASS_AIR_FLOW",
    "THROTTLE_POSITION",
    "CONTROL_MODULE_VOLTAGE",
    "ENGINE_FUEL_RATE",
    "VEHICLE_SPEED",
    "AMBIENT_AIR_TEMPERATURE",
    "BAROMETRIC_PRESSURE",
    "CALCULATED_ENGINE_LOAD_VALUE",
]

DERIVED_FEATURES_FOR_SUPERVISED = [
    # Rolling Means (Window 10)
    'ENGINE_RPM_rol_10_mean',
    'VEHICLE_SPEED_rol_10_mean',
    'CALCULATED_ENGINE_LOAD_VALUE_rol_10_mean',
    # Rolling Std Devs (Window 10)
    'ENGINE_RPM_rol_10_std',
    'VEHICLE_SPEED_rol_10_std',
    'CALCULATED_ENGINE_LOAD_VALUE_rol_10_std',
    # Differences (Lag 1)
    'ENGINE_RPM_diff_1',
    'VEHICLE_SPEED_diff_1',
    'CALCULATED_ENGINE_LOAD_VALUE_diff_1'
]

# Potentially reuse feature lists from anomaly detection or define specific ones here
# from app.preprocessing.anomaly_detection import CORE_PIDS_FOR_ANOMALY, DERIVED_FEATURES_FOR_ANOMALY

# --- Configuration & Constants ---
DEFAULT_MODEL_DIR = "models/supervised"
TARGET_COL = 'parsed_dtcs' # Target column in the Kaggle input parquet
RANDOM_STATE = 42

# --- Helper Functions ---

def load_data(input_path: str) -> pd.DataFrame:
    """Loads features and targets from the final model input Parquet file."""
    logging.info(f"Loading data from: {input_path}")
    try:
        df = pd.read_parquet(input_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in {input_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        sys.exit(1)

def preprocess_target(y_raw: pd.Series) -> pd.Series:
    """Ensures target column contains lists, handling NaNs or other types (like numpy arrays)."""
    logging.info(f"Preprocessing target column '{TARGET_COL}'...")

    def convert_to_list(item):
        if isinstance(item, np.ndarray):
            return item.tolist() # Convert numpy array to list
        elif isinstance(item, list):
            return item # Already a list
        else:
            return [] # Treat NaNs or other types as empty lists

    y_processed = y_raw.apply(convert_to_list)

    # Log count of samples with no DTCs (empty list)
    no_dtc_count = (y_processed.apply(len) == 0).sum()
    logging.info(f"Found {no_dtc_count} samples with no DTCs (empty list) after conversion.")
    return y_processed

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects numeric features, excluding known non-feature columns."""
    logging.info("Preparing features...")
    # Define columns known NOT to be features
    cols_to_exclude = [
        TARGET_COL, 'absolute_timestamp', 'VEHICLE_ID', # Target, time, identifier
        'hour', 'dayofweek', 'is_weekend', # Base time features
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', # Cyclical time features
        # Add any other known metadata columns loaded from parquet if necessary
        'MARK', 'MODEL', 'YEAR', 'ENGINE_TYPE', 'FUEL', 'AUTOMATIC', 'VIN' # Example metadata
    ]
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Filter out excluded columns
    features_to_use = [col for col in numeric_cols if col not in cols_to_exclude]
    logging.info(f"Selected {len(features_to_use)} numeric features.")
    # Impute missing values in features (e.g., from scaling or original data)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df[features_to_use])
    X = pd.DataFrame(X_imputed, columns=features_to_use, index=df.index)
    logging.info("Feature imputation complete.")
    return X

def train_model(X_train: pd.DataFrame, y_train_bin: np.ndarray, model_type: str, use_tuning: bool, search_iterations: int, search_cv_folds: int) -> OneVsRestClassifier:
    """Trains a OneVsRestClassifier with the specified base estimator (RF or LGBM), optionally with hyperparameter tuning."""
    logging.info(f"Training OneVsRestClassifier with base model: {model_type}")

    # Define base estimators
    if model_type == 'rf':
        base_estimator = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        # Basic hyperparameter grid for Random Forest (adjust as needed)
        param_dist = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 20, 30, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'lgbm':
        base_estimator = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        # Basic hyperparameter grid for LightGBM (adjust as needed)
        param_dist = {
            'estimator__n_estimators': [100, 200, 400],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__num_leaves': [31, 50, 70],
            'estimator__max_depth': [-1, 10, 20],
             'estimator__reg_alpha': [0.0, 0.1, 0.5], # L1 regularization
             'estimator__reg_lambda': [0.0, 0.1, 0.5], # L2 regularization
        }
    else:
        raise ValueError("Unsupported model type. Choose 'rf' or 'lgbm'.")

    ovr_classifier = OneVsRestClassifier(base_estimator)

    if use_tuning:
        logging.info(f"Performing RandomizedSearchCV with {search_iterations} iterations and {search_cv_folds} CV folds...")
        random_search = RandomizedSearchCV(
            ovr_classifier,
            param_distributions=param_dist,
            n_iter=search_iterations,
            cv=search_cv_folds,
            scoring='f1_micro', # Micro-average F1 is often suitable for multi-label
            random_state=RANDOM_STATE,
            n_jobs=-1, # Use all available cores for search
            verbose=1 # Show progress
        )
        random_search.fit(X_train, y_train_bin)
        logging.info(f"Best parameters found: {random_search.best_params_}")
        logging.info(f"Best micro F1 score from search: {random_search.best_score_:.4f}")
        model = random_search.best_estimator_ # Use the best model found
    else:
        logging.info("Training model with default parameters...")
        model = ovr_classifier
        model.fit(X_train, y_train_bin)

    return model


def evaluate_model(model: OneVsRestClassifier, X_test: pd.DataFrame, y_test_bin: np.ndarray, mlb: MultiLabelBinarizer):
    """Evaluates the multi-label model and logs metrics."""
    logging.info("Evaluating model...")
    y_pred_bin = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) # Get probabilities for potential threshold tuning later

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_test_bin, y_pred_bin) # Exact match ratio
    hamming = hamming_loss(y_test_bin, y_pred_bin)   # Fraction of incorrect labels
    f1_micro = f1_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)
    f1_macro = f1_score(y_test_bin, y_pred_bin, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
    f1_samples = f1_score(y_test_bin, y_pred_bin, average='samples', zero_division=0) # Per-sample F1

    precision_micro = precision_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)
    recall_micro = recall_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)

    logging.info("--- Evaluation Metrics ---")
    logging.info(f"Subset Accuracy (Exact Match Ratio): {accuracy:.4f}")
    logging.info(f"Hamming Loss (Lower is better):     {hamming:.4f}")
    logging.info(f"F1 Score (Micro):                   {f1_micro:.4f}")
    logging.info(f"F1 Score (Macro):                   {f1_macro:.4f}")
    logging.info(f"F1 Score (Weighted):                {f1_weighted:.4f}")
    logging.info(f"F1 Score (Samples):                 {f1_samples:.4f}")
    logging.info(f"Precision (Micro):                  {precision_micro:.4f}")
    logging.info(f"Recall (Micro):                     {recall_micro:.4f}")
    logging.info("--------------------------")

    # Classification report per label
    report = classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0)
    logging.info(f"Classification Report (Per Label):\\n{report}")

    # Log feature importances if applicable (for the base estimator within OvR)
    try:
        if hasattr(model.estimators_[0], 'feature_importances_'):
            # Average importance across all OvR estimators
            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            feature_names = X_test.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            logging.info(f"Top 20 Feature Importances (Averaged across OvR estimators):\\n{feature_importance_df.head(20).to_string()}")
    except Exception as e:
        logging.warning(f"Could not calculate or log feature importances: {e}")


def save_model_and_binarizer(model: OneVsRestClassifier, mlb: MultiLabelBinarizer, model_path: str, binarizer_path: str):
    """Saves the trained model and the MultiLabelBinarizer."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Trained model saved to: {model_path}")
    except Exception as e:
        logging.error(f"Error saving model to {model_path}: {e}")

    try:
        os.makedirs(os.path.dirname(binarizer_path), exist_ok=True)
        joblib.dump(mlb, binarizer_path)
        logging.info(f"MultiLabelBinarizer saved to: {binarizer_path}")
    except Exception as e:
        logging.error(f"Error saving binarizer to {binarizer_path}: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train a multi-label classifier on Kaggle OBD data to predict DTCs.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the final model input Parquet file (e.g., data/model_input/exp1_..._model_input.parquet)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["rf", "lgbm"],
        help="Base model type for OneVsRestClassifier ('rf' or 'lgbm')."
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default=None, # Will be generated based on model type if None
        help="Path to save the trained multi-label model."
    )
    parser.add_argument(
        "--binarizer-save-path",
        type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, "kaggle_dtc_mlb.joblib"),
        help="Path to save the fitted MultiLabelBinarizer."
    )
    parser.add_argument(
        "--load-binarizer-path",
        type=str,
        default=None,
        help="Path to load a pre-fitted MultiLabelBinarizer (optional)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for the test split."
    )
    parser.add_argument(
        "--use-tuning",
        action="store_true",
        help="Perform hyperparameter tuning using RandomizedSearchCV."
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=10,
        help="Number of parameter settings sampled by RandomizedSearchCV."
    )
    parser.add_argument(
        "--search-cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds for RandomizedSearchCV."
    )

    args = parser.parse_args()

    # Generate default model save path if not provided
    if args.model_save_path is None:
        model_filename = f"kaggle_{args.model_type}_ovr_model.joblib"
        if args.use_tuning:
             model_filename = f"kaggle_{args.model_type}_ovr_tuned_model.joblib"
        args.model_save_path = os.path.join(DEFAULT_MODEL_DIR, model_filename)


    logging.info("--- Starting Kaggle DTC Training Pipeline ---")
    logging.info(f"Configuration: {vars(args)}")

    # 1. Load Data
    df = load_data(args.input_path)

    # 2. Prepare Features
    X = prepare_features(df)

    # 3. Prepare Target Variable
    y_raw = preprocess_target(df[TARGET_COL])

    # 4. Binarize Target Labels
    if args.load_binarizer_path and os.path.exists(args.load_binarizer_path):
        logging.info(f"Loading pre-fitted MultiLabelBinarizer from: {args.load_binarizer_path}")
        mlb = joblib.load(args.load_binarizer_path)
        y_bin = mlb.transform(y_raw)
    else:
        logging.info("Fitting new MultiLabelBinarizer...")
        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y_raw)
        logging.info(f"Target binarized. Found {len(mlb.classes_)} unique DTCs (classes): {mlb.classes_}")
        # Save the newly fitted binarizer if it wasn't loaded
        if not args.load_binarizer_path:
             save_model_and_binarizer(None, mlb, "", args.binarizer_save_path) # Pass dummy model path


    # 5. Split Data
    logging.info(f"Splitting data into train/test sets (Test size: {args.test_size})...")
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_bin, test_size=args.test_size, random_state=RANDOM_STATE, stratify=None # Stratify is tricky for multi-label
    )
    logging.info(f"Train set shape: X={X_train.shape}, y={y_train_bin.shape}")
    logging.info(f"Test set shape: X={X_test.shape}, y={y_test_bin.shape}")

    # 6. Train Model
    model = train_model(
        X_train, y_train_bin, args.model_type, args.use_tuning, args.search_iterations, args.search_cv_folds
    )

    # 7. Evaluate Model
    evaluate_model(model, X_test, y_test_bin, mlb)

    # 8. Save Model
    save_model_and_binarizer(model, None, args.model_save_path, "") # Pass dummy binarizer path

    logging.info("--- Kaggle DTC Training Pipeline Finished ---")

if __name__ == "__main__":
    main()