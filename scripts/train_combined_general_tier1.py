import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # For median imputation
from joblib import dump
import os
import sys
import logging
import argparse
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Constants ---
# Define the core PID names as they appear in the *input* combined parquet file
# These should match TARGET_CORE_PIDS from the combination script.
CORE_PIDS_FOR_TRAINING = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "INTAKE_AIR_TEMP",
    "THROTTLE_POS",
    "VEHICLE_SPEED",
    "ENGINE_LOAD",
]

# Default model parameters (can be overridden via args)
DEFAULT_IFOREST_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 'auto', # Or a float like 0.01 (1% expected anomalies)
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1, # Use all available cores
    'random_state': 42
}

def train_model(input_parquet_path: str,
                output_model_path: str,
                output_scaler_path: str,
                output_imputer_path: str, # Path to save the imputer
                iforest_params: dict):
    """
    Trains a StandardScaler, SimpleImputer (median), and an Isolation Forest model
    on the provided combined dataset.
    """
    logging.info(f"Starting Tier 1 combined general model training...")
    logging.info(f"Input data: {input_parquet_path}")
    logging.info(f"Output model: {output_model_path}")
    logging.info(f"Output scaler: {output_scaler_path}")
    logging.info(f"Output imputer: {output_imputer_path}")
    logging.info(f"Isolation Forest parameters: {iforest_params}")

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_scaler_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_imputer_path), exist_ok=True)

    try:
        df = pd.read_parquet(input_parquet_path)
        logging.info(f"Successfully loaded data. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Failed to load input Parquet file: {input_parquet_path}. Error: {e}", exc_info=True)
        sys.exit(1)

    # Select only the PIDs required for training
    df_pids = df[CORE_PIDS_FOR_TRAINING].copy()
    logging.info(f"Selected {len(CORE_PIDS_FOR_TRAINING)} PIDs for training. Shape: {df_pids.shape}")

    # --- 1. Imputation ---
    logging.info("Applying Median Imputation for missing values...")
    imputer = SimpleImputer(strategy='median')
    df_imputed_values = imputer.fit_transform(df_pids)
    df_imputed = pd.DataFrame(df_imputed_values, columns=df_pids.columns, index=df_pids.index)

    # Log how many NaNs were imputed for each column
    for col in CORE_PIDS_FOR_TRAINING:
        original_nans = df_pids[col].isnull().sum()
        if original_nans > 0:
            logging.info(f"Imputed {original_nans} NaN values in column '{col}'.")

    logging.info("Imputation complete.")
    dump(imputer, output_imputer_path)
    logging.info(f"Imputer saved to {output_imputer_path}")

    # --- 2. Scaling ---
    logging.info("Applying StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_imputed) # Use imputed data for scaling
    logging.info("Scaling complete.")
    dump(scaler, output_scaler_path)
    logging.info(f"Scaler saved to {output_scaler_path}")

    # --- 3. Model Training ---
    logging.info("Training Isolation Forest model...")
    model = IsolationForest(**iforest_params)
    model.fit(scaled_features)
    logging.info("Isolation Forest training complete.")
    dump(model, output_model_path)
    logging.info(f"Model saved to {output_model_path}")

    logging.info("Tier 1 combined general model training finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a combined general Tier 1 anomaly detection model.")

    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(project_root, "data/model_input/combined_raw_pids_for_tier1_training.parquet"),
        help="Path to the combined input Parquet file for training."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(project_root, "models/anomaly"),
        help="Directory to save the trained model, scaler, and imputer."
    )
    parser.add_argument(
        "--model_name_prefix",
        type=str,
        default="tier1_combined_general",
        help="Prefix for the output model, scaler, and imputer files."
    )
    parser.add_argument(
        "--iforest_params",
        type=str,
        default=None,
        help=(
            "JSON string of Isolation Forest parameters to override defaults. "
            "Example: '{\"n_estimators\": 150, \"contamination\": 0.02}'"
        )
    )

    args = parser.parse_args()

    # Construct full output paths
    output_model_file = os.path.join(args.model_dir, f"{args.model_name_prefix}_isolation_forest.joblib")
    output_scaler_file = os.path.join(args.model_dir, f"{args.model_name_prefix}_scaler.joblib")
    output_imputer_file = os.path.join(args.model_dir, f"{args.model_name_prefix}_imputer.joblib")

    # Parse iforest_params if provided
    current_iforest_params = DEFAULT_IFOREST_PARAMS.copy()
    if args.iforest_params:
        try:
            override_params = json.loads(args.iforest_params)
            current_iforest_params.update(override_params)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON for iforest_params: {args.iforest_params}. Error: {e}. Using defaults.")

    train_model(
        args.input_path,
        output_model_file,
        output_scaler_file,
        output_imputer_file,
        current_iforest_params
    )