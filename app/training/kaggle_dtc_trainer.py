#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to train a supervised model to predict potential SYSTEM-LEVEL FAULTS
based on sensor data, using DTC occurrences from the Kaggle dataset.
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
from typing import Dict, Any, Tuple, List, Set
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
DTC_INPUT_COL = 'parsed_dtcs' # Original DTC column name from input parquet
SYSTEM_TARGET_COL = 'target_systems' # New target column with system labels
RANDOM_STATE = 42

# --- DTC to System Mapping ---
# Define a mapping from specific DTCs to broader vehicle system categories.
# This simplifies the prediction task and provides more actionable insights.
DTC_TO_SYSTEM_MAP = {
    # Fuel System (inc O2 Sensors)
    'P0171': 'FUEL_SYSTEM', 'P0174': 'FUEL_SYSTEM', # Lean
    'P0172': 'FUEL_SYSTEM', 'P0175': 'FUEL_SYSTEM', # Rich
    'P0133': 'FUEL_SYSTEM', # O2 Sensor Circuit Slow Response (Bank 1 Sensor 1) -> related to fuel mixture feedback
    # Air Intake System (MAF/MAP)
    'P0101': 'AIR_INTAKE_SYSTEM', 'P0102': 'AIR_INTAKE_SYSTEM', # MAF
    'P0106': 'AIR_INTAKE_SYSTEM', # MAP
    # Throttle System
    'P0121': 'THROTTLE_SYSTEM', 'P0122': 'THROTTLE_SYSTEM', 'P0123': 'THROTTLE_SYSTEM', # TPS
    # Coolant System
    'P0128': 'COOLANT_SYSTEM',
    # ABS/Wheel Speed Sensors
    'C0300': 'ABS_SYSTEM', 'C1004': 'ABS_SYSTEM',
    # Engine Control Circuits / Sensors (Grouping VVT, EGT etc broadly)
    'P0078': 'ENGINE_CONTROL_CIRCUITS', 'P0079': 'ENGINE_CONTROL_CIRCUITS', # VVT Solenoid Circuit
    'P2036': 'ENGINE_CONTROL_CIRCUITS', # EGT sensor circuit
    # Engine Mechanical/Misc (Grouping P1xxx, P3xxx, Misfires if added)
    'P1004': 'ENGINE_SYSTEM_OTHER', 'P3000': 'ENGINE_SYSTEM_OTHER',
    # Intake Manifold Components
    'P2004': 'INTAKE_MANIFOLD_SYSTEM',
    # Communication
    'U1004': 'COMMUNICATION_BUS',
    # Restraint System
    'B0004': 'RESTRAINT_SYSTEM',
    # --- Add more mappings as needed based on data exploration ---
}
UNKNOWN_SYSTEM_LABEL = 'UNKNOWN_SYSTEM'


# --- Helper Functions ---

def load_data(input_path: str) -> pd.DataFrame:
    """Loads features and targets from the final model input Parquet file."""
    logging.info(f"Loading data from: {input_path}")
    try:
        df = pd.read_parquet(input_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        if DTC_INPUT_COL not in df.columns:
            raise ValueError(f"Required DTC input column '{DTC_INPUT_COL}' not found in {input_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        sys.exit(1)

def map_dtcs_to_systems(dtc_list: List[str]) -> List[str]:
    """Maps a list of DTCs to a unique list of system names using DTC_TO_SYSTEM_MAP."""
    systems: Set[str] = set()
    unknown_dtcs_found = False
    for dtc in dtc_list:
        system = DTC_TO_SYSTEM_MAP.get(dtc)
        if system:
            systems.add(system)
        else:
            # Optionally log unknown DTCs the first time they're encountered
            # logging.debug(f"DTC '{dtc}' not found in system map.")
            unknown_dtcs_found = True
    # If any DTC didn't map and the list wasn't originally empty, add UNKNOWN
    if unknown_dtcs_found and dtc_list:
         systems.add(UNKNOWN_SYSTEM_LABEL)
    return sorted(list(systems))


def preprocess_target(y_raw: pd.Series) -> pd.Series:
    """
    Ensures target DTC column contains lists, then maps DTCs to system labels.
    Handles NaNs or other types (like numpy arrays).
    """
    logging.info(f"Preprocessing target column '{DTC_INPUT_COL}' to create system labels...")

    # Step 1: Ensure input is a list of strings
    def convert_dtc_list(item):
        dtc_list = []
        if isinstance(item, np.ndarray):
            dtc_list = item.tolist() # Convert numpy array to list
        elif isinstance(item, list):
            dtc_list = item # Already a list
        # else: Treat NaNs or other types as empty lists, already handled by default empty dtc_list

        # Ensure elements are strings
        return [str(d) for d in dtc_list if pd.notna(d)]

    y_dtc_lists = y_raw.apply(convert_dtc_list)

    # Step 2: Map DTC lists to system lists
    y_system_lists = y_dtc_lists.apply(map_dtcs_to_systems)

    # Log count of samples with no system issues (empty list)
    no_issue_count = (y_system_lists.apply(len) == 0).sum()
    logging.info(f"Found {no_issue_count} samples with no associated system issues (empty list) after mapping.")
    return y_system_lists


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects numeric features, excluding known non-feature columns."""
    logging.info("Preparing features...")
    # Define columns known NOT to be features
    cols_to_exclude = [
        DTC_INPUT_COL, SYSTEM_TARGET_COL, # Target columns
        'absolute_timestamp', 'VEHICLE_ID', # Time, identifier
        'hour', 'dayofweek', 'is_weekend', # Base time features
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', # Cyclical time features
        # Add any other known metadata columns loaded from parquet if necessary
        'MARK', 'MODEL', 'YEAR', 'ENGINE_TYPE', 'FUEL', 'AUTOMATIC', 'VIN', # Example metadata
        'FUEL_TYPE' # Often loaded/created during processing
    ]
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Filter out excluded columns - use set for efficiency
    excluded_set = set(cols_to_exclude)
    features_to_use = [col for col in numeric_cols if col not in excluded_set]

    # Check for constant columns (zero variance) - these provide no info
    non_constant_features = []
    for col in features_to_use:
        if df[col].nunique() > 1:
            non_constant_features.append(col)
        else:
            logging.warning(f"Dropping constant column: {col}")

    if len(non_constant_features) == 0:
         logging.error("No non-constant numeric features found after filtering. Exiting.")
         sys.exit(1)

    logging.info(f"Selected {len(non_constant_features)} non-constant numeric features.")

    # Impute missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df[non_constant_features])
    X = pd.DataFrame(X_imputed, columns=non_constant_features, index=df.index)
    logging.info("Feature imputation complete.")
    return X

def train_model(X_train: pd.DataFrame, y_train_bin: np.ndarray, model_type: str, use_tuning: bool, search_iterations: int, search_cv_folds: int) -> OneVsRestClassifier:
    """Trains a OneVsRestClassifier with the specified base estimator (RF or LGBM), optionally with hyperparameter tuning."""
    logging.info(f"Training OneVsRestClassifier for SYSTEM-LEVEL prediction with base model: {model_type}")

    # Define base estimators (Ensure class_weight='balanced' is suitable for system-level imbalance)
    if model_type == 'rf':
        base_estimator = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        # Hyperparameter grid for Random Forest
        param_dist = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 20, 30, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'lgbm':
        if not lightgbm_available:
             logging.error("LightGBM selected but not installed. Please install lightgbm.")
             sys.exit(1)
        base_estimator = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        # Hyperparameter grid for LightGBM
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
            scoring='f1_micro', # Micro-average F1 might still be okay, or consider 'f1_weighted' or 'f1_samples'
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
    """Evaluates the multi-label model (predicting systems) and logs metrics."""
    logging.info("Evaluating system-level prediction model...")
    y_pred_bin = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) # Get probabilities for potential threshold tuning later

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_test_bin, y_pred_bin) # Exact match ratio for system sets
    hamming = hamming_loss(y_test_bin, y_pred_bin)   # Fraction of incorrect system labels
    f1_micro = f1_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)
    f1_macro = f1_score(y_test_bin, y_pred_bin, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_bin, y_pred_bin, average='weighted', zero_division=0)
    f1_samples = f1_score(y_test_bin, y_pred_bin, average='samples', zero_division=0) # Per-sample F1

    precision_micro = precision_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)
    recall_micro = recall_score(y_test_bin, y_pred_bin, average='micro', zero_division=0)

    logging.info("--- Evaluation Metrics (System Level) ---")
    logging.info(f"Subset Accuracy (Exact Match Ratio): {accuracy:.4f}")
    logging.info(f"Hamming Loss (Lower is better):     {hamming:.4f}")
    logging.info(f"F1 Score (Micro):                   {f1_micro:.4f}")
    logging.info(f"F1 Score (Macro):                   {f1_macro:.4f}")
    logging.info(f"F1 Score (Weighted):                {f1_weighted:.4f}")
    logging.info(f"F1 Score (Samples):                 {f1_samples:.4f}")
    logging.info(f"Precision (Micro):                  {precision_micro:.4f}")
    logging.info(f"Recall (Micro):                     {recall_micro:.4f}")
    logging.info("---------------------------------------")

    # Classification report per system label
    # Check if mlb.classes_ is empty which can happen if y_raw was all empty lists
    if len(mlb.classes_) == 0:
        logging.warning("MultiLabelBinarizer has no classes_ (likely no non-empty targets found). Cannot generate classification report.")
    else:
        report = classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0)
        logging.info(f"Classification Report (Per System):\\n{report}")

    # Log feature importances (remains the same)
    try:
        if hasattr(model.estimators_[0], 'feature_importances_'):
            # Average importance across all OvR estimators
            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            feature_names = X_test.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            logging.info(f"Top 20 Feature Importances (Averaged across OvR estimators):\\n{feature_importance_df.head(20).to_string()}")
    except IndexError:
         logging.warning("Could not log feature importances: Model has no estimators (maybe training failed or produced no classes).")
    except Exception as e:
        logging.warning(f"Could not calculate or log feature importances: {e}")


def save_model_and_binarizer(model: OneVsRestClassifier | None, mlb: MultiLabelBinarizer | None, model_path: str, binarizer_path: str):
    """Saves the trained model and/or the MultiLabelBinarizer if they are provided."""
    if model is not None and model_path: # Check if model and path are provided
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            logging.info(f"Trained model saved to: {model_path}")
        except Exception as e:
            logging.error(f"Error saving model to {model_path}: {e}")

    if mlb is not None and binarizer_path: # Check if binarizer and path are provided
        try:
            os.makedirs(os.path.dirname(binarizer_path), exist_ok=True)
            joblib.dump(mlb, binarizer_path)
            logging.info(f"MultiLabelBinarizer saved to: {binarizer_path}")
        except Exception as e:
            logging.error(f"Error saving binarizer to {binarizer_path}: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train a multi-label classifier on Kaggle OBD data to predict potential VEHICLE SYSTEM issues.")
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
        help="Path to save the trained multi-label system prediction model."
    )
    parser.add_argument(
        "--binarizer-save-path",
        type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, "kaggle_system_mlb.joblib"),
        help="Path to save the fitted MultiLabelBinarizer (for systems)."
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
        tuning_suffix = "_tuned" if args.use_tuning else ""
        model_filename = f"kaggle_{args.model_type}_ovr{tuning_suffix}_system_model.joblib"
        args.model_save_path = os.path.join(DEFAULT_MODEL_DIR, model_filename)
    # Update binarizer path default based on model name structure (optional but consistent)
    if args.binarizer_save_path == os.path.join(DEFAULT_MODEL_DIR, "kaggle_dtc_mlb.joblib"): # If still default
         binarizer_filename = f"kaggle_{args.model_type}_ovr{tuning_suffix}_system_mlb.joblib"
         args.binarizer_save_path = os.path.join(DEFAULT_MODEL_DIR, binarizer_filename)


    logging.info("--- Starting Kaggle System-Level Training Pipeline ---")
    logging.info(f"Configuration: {vars(args)}")

    # 1. Load Data
    df = load_data(args.input_path)

    # 2. Prepare Features (Requires target col to be dropped)
    # Ensure the target system column is created before feature prep if needed
    # Note: prepare_features function already excludes SYSTEM_TARGET_COL if present

    # 3. Prepare Target Variable (Systems)
    y_system_labels = preprocess_target(df[DTC_INPUT_COL])
    df[SYSTEM_TARGET_COL] = y_system_labels # Add to df temporarily if needed for feature prep exclusion logic

    # 4. Prepare Features using the DataFrame potentially updated with target
    X = prepare_features(df.drop(columns=[SYSTEM_TARGET_COL], errors='ignore')) # Drop target before passing to model

    # 5. Binarize Target Labels (Systems)
    # We always fit a new binarizer based on the system map
    logging.info("Fitting new MultiLabelBinarizer for system labels...")
    mlb = MultiLabelBinarizer()
    # Handle case where all target lists might be empty after mapping
    if y_system_labels.apply(len).sum() == 0:
        logging.error("No non-empty target system lists found after mapping DTCs. Cannot train model.")
        # Optionally save an empty binarizer?
        # save_model_and_binarizer(None, mlb, "", args.binarizer_save_path)
        sys.exit(1)

    y_bin = mlb.fit_transform(y_system_labels)
    logging.info(f"Target systems binarized. Found {len(mlb.classes_)} unique systems (classes): {mlb.classes_}")
    # Save the newly fitted binarizer
    save_model_and_binarizer(None, mlb, "", args.binarizer_save_path) # Pass dummy model path


    # 6. Split Data
    logging.info(f"Splitting data into train/test sets (Test size: {args.test_size})...")
    # Ensure X and y_bin have consistent indices if df index was used
    X = X.loc[y_system_labels.index] # Align indices just in case

    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_bin, test_size=args.test_size, random_state=RANDOM_STATE, stratify=None # Stratify is tricky for multi-label
    )
    logging.info(f"Train set shape: X={X_train.shape}, y={y_train_bin.shape}")
    logging.info(f"Test set shape: X={X_test.shape}, y={y_test_bin.shape}")

    # 7. Train Model
    model = train_model(
        X_train, y_train_bin, args.model_type, args.use_tuning, args.search_iterations, args.search_cv_folds
    )

    # 8. Evaluate Model
    evaluate_model(model, X_test, y_test_bin, mlb)

    # 9. Save Model (Only model, binarizer already saved)
    save_model_and_binarizer(model, None, args.model_save_path, "") # Pass dummy binarizer path

    logging.info("--- Kaggle System-Level Training Pipeline Finished ---")

if __name__ == "__main__":
    main()