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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer # Import SimpleImputer
from joblib import dump, load # Import load as well for potentially loading later
import argparse
import os
import logging

try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except ImportError:
    LGBMClassifier = None
    lightgbm_available = False
    logging.warning("LightGBM not installed. LGBMClassifier will not be available.")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def load_and_prepare_data(features_path: str, anomalies_path: str) -> pd.DataFrame:
    """
    Loads the main features dataset and the anomaly analysis results,
    merges them, and prepares the target variable.
    """
    logging.info(f"Loading features from: {features_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    df_features = pd.read_parquet(features_path)
    logging.info(f"Feature data shape: {df_features.shape}")

    logging.info(f"Loading anomaly results from: {anomalies_path}")
    if not os.path.exists(anomalies_path):
        raise FileNotFoundError(f"Anomaly results file not found: {anomalies_path}")
    df_anomalies = pd.read_csv(anomalies_path, index_col=0) # Assuming index is timestamp/identifier
    # Parse the 'potential_dtcs' column back into lists (it gets saved as string)
    # Safely evaluate the string representation of the list
    try:
        df_anomalies['potential_dtcs'] = df_anomalies['potential_dtcs'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
    except Exception as e:
        logging.error(f"Error parsing 'potential_dtcs' column: {e}. Ensure it contains valid list strings.")
        # Handle error - maybe fill with empty lists or raise
        df_anomalies['potential_dtcs'] = [[] for _ in range(len(df_anomalies))]

    logging.info(f"Anomaly data shape: {df_anomalies.shape}")

    # Extract only the 'potential_dtcs' column to merge
    df_dtcs = df_anomalies[['potential_dtcs']]

    # Merge DTCs into the main feature DataFrame based on index
    # Use a left merge to keep all original feature data points
    df_merged = df_features.merge(df_dtcs, left_index=True, right_index=True, how='left')

    # Fill NaNs created by the merge (for data points not flagged as anomalies) with empty lists
    # Important: Access the column using .loc for reliable assignment
    dtc_col_mask = df_merged['potential_dtcs'].isna()
    df_merged.loc[dtc_col_mask, 'potential_dtcs'] = df_merged.loc[dtc_col_mask, 'potential_dtcs'].apply(lambda x: [])


    logging.info(f"Merged data shape: {df_merged.shape}")
    if df_merged.shape[0] != df_features.shape[0]:
        logging.warning("Row count changed after merging anomalies. Check index alignment.")

    # --- Define Target Variable (Example: Binary flag for *any* heuristic DTC) ---
    df_merged['target'] = df_merged['potential_dtcs'].apply(lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0)
    logging.info(f"Target variable created. Distribution:\n{df_merged['target'].value_counts(normalize=True)}")

    # Define specific features to use for training
    features_to_use = CORE_PIDS_FOR_SUPERVISED + DERIVED_FEATURES_FOR_SUPERVISED
    # features_to_use = [] # Placeholder

    # Ensure selected features exist in the dataframe
    available_features = [f for f in features_to_use if f in df_merged.columns]
    if len(available_features) != len(features_to_use):
        missing = set(features_to_use) - set(available_features)
        logging.warning(f"Some specified features are missing: {missing}")
    if not available_features:
         raise ValueError("No features selected or available for training.")
    logging.info(f"Using {len(available_features)} features for training: {available_features}")

    # Prepare X and y
    X = df_merged[available_features]
    y = df_merged['target']

    # Handle potential NaNs in feature columns (e.g., fill with mean/median or use imputation)
    # logging.info("Handling potential NaNs in feature columns (filling with 0 for now)...")
    # X = X.fillna(0) # Simple strategy, consider more robust imputation
    # Note: Imputation will be done *after* train/test split to avoid data leakage

    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_save_path: str, model_type: str = 'rf'):
    """Trains the specified classifier and saves it."""
    # Note: Imputer is now handled in main and saved separately
    logging.info(f"Training model ({model_type})...")

    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    elif model_type == 'lgbm' and lightgbm_available:
        # Basic LGBM parameters - might need tuning
        model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    elif model_type == 'lgbm' and not lightgbm_available:
         raise ValueError("LightGBM selected but not installed. Please install lightgbm.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Save the trained model
    if model_save_path:
        logging.info(f"Saving model to: {model_save_path}")
        output_dir = os.path.dirname(model_save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        dump(model, model_save_path)

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the model on the test set."""
    logging.info("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    logging.info("Confusion Matrix:")
    # Use np.array_str for potentially better formatting in logs
    cm_str = np.array_str(confusion_matrix(y_test, y_pred))
    logging.info("\n" + cm_str)


    logging.info("\nClassification Report:")
    # Get report as string
    report = classification_report(y_test, y_pred)
    logging.info("\n" + report)

    # --- Feature Importances ---
    if hasattr(model, 'feature_importances_') and hasattr(X_test, 'columns'):
        logging.info("\n--- Feature Importances ---")
        importances = model.feature_importances_
        feature_names = X_test.columns
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        # Log the feature importances
        logging.info("\n" + feature_importance_df.to_string(index=False))
    else:
        logging.warning("Could not extract feature importances (model type might not support it or X_test is not a DataFrame).")


def main(args):
    """Main execution function."""
    logging.info("Starting supervised DTC training process...")

    try:
        # Load data (before imputation)
        X, y = load_and_prepare_data(args.features_path, args.anomalies_path)

        # Split data
        logging.info(f"Splitting data (test_size={args.test_size}, random_state=42)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y # Stratify important for imbalanced data
        )
        logging.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

        # --- Imputation (Fit on Train, Transform Train & Test) ---
        logging.info(f"Applying SimpleImputer (strategy='{args.imputer_strategy}')...")
        imputer = SimpleImputer(strategy=args.imputer_strategy)

        # Get feature names before imputation (imputer returns numpy array)
        feature_names = X_train.columns.tolist()

        # Fit on the training data ONLY
        X_train_imputed = imputer.fit_transform(X_train)
        # Transform the test data using the *fitted* imputer
        X_test_imputed = imputer.transform(X_test)

        # Convert back to DataFrame to keep feature names (optional but good practice)
        X_train = pd.DataFrame(X_train_imputed, columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=feature_names, index=X_test.index)
        logging.info("Imputation complete.")

        # Save the fitted imputer
        if args.imputer_output_path:
            logging.info(f"Saving imputer to: {args.imputer_output_path}")
            output_dir = os.path.dirname(args.imputer_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            dump(imputer, args.imputer_output_path)
        # --- End Imputation ---

        # Train model using imputed data
        model = train_model(
            X_train, y_train,
            args.model_output_path,
            model_type=args.model_type # Pass model type
        )

        # Evaluate model using imputed data
        evaluate_model(model, X_test, y_test)

        logging.info("Supervised DTC training script finished successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
    except ValueError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a supervised model to predict potential DTCs using anomaly heuristics as labels.")
    parser.add_argument("--features-path", required=True, help="Path to the main Parquet file with features.")
    parser.add_argument("--anomalies-path", required=True, help="Path to the CSV file containing anomaly detection results (including 'potential_dtcs').")
    parser.add_argument("--model-output-path", required=True, help="Path to save the trained model (e.g., .joblib file).")
    parser.add_argument("--imputer-output-path", required=True, help="Path to save the fitted imputer (e.g., .joblib file).")
    parser.add_argument("--imputer-strategy", type=str, default='mean', choices=['mean', 'median', 'most_frequent'], help="Strategy for SimpleImputer.")
    parser.add_argument("--model-type", type=str, default='rf', choices=['rf', 'lgbm'], help="Type of model to train ('rf' for RandomForest, 'lgbm' for LightGBM).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for the test set.")
    # Add arguments for feature selection, model choice, hyperparameters etc. later

    args = parser.parse_args()
    main(args)