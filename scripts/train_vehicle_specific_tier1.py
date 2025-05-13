import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import sys
import logging
import argparse
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define the core PID names as they appear in the *input* parquet file
# Adjust these based on the specific dataset being trained (e.g., Volvo raw pids)
RAW_CORE_PIDS = [
    "ENGINE_RPM",
    "COOLANT_TEMPERATURE",
    "INTAKE_AIR_TEMPERATURE",
    "THROTTLE_POSITION",
    "VEHICLE_SPEED",
    "CALCULATED_ENGINE_LOAD_VALUE",
]

# Default model parameters (can be overridden via args)
DEFAULT_IFOREST_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 'auto', # Let IF estimate contamination
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1,
    'random_state': 42
}
# --- End Constants ---


def train_tier1_model(input_path: str, model_dir: str, model_name_prefix: str, iforest_params: dict):
    """
    Trains a StandardScaler and IsolationForest model on the specified raw PID columns
    from the input parquet file and saves them.
    """
    logging.info(f"--- Starting Tier 1 Model Training for: {model_name_prefix} ---")
    logging.info(f"Input data path: {input_path}")
    logging.info(f"Output model directory: {model_dir}")
    logging.info(f"Using features: {RAW_CORE_PIDS}")
    logging.info(f"Isolation Forest parameters: {iforest_params}")

    # --- 1. Load Data ---
    if not os.path.exists(input_path):
        logging.error(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    try:
        logging.info(f"Loading data and selecting features: {RAW_CORE_PIDS}")
        df = pd.read_parquet(input_path, columns=RAW_CORE_PIDS)
        logging.info(f"Successfully loaded data. Shape before NaN handling: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading Parquet file or selecting columns: {e}")
        # Check if columns exist
        try:
            all_cols = pd.read_parquet(input_path, columns=[]).columns.tolist()
            missing = [c for c in RAW_CORE_PIDS if c not in all_cols]
            if missing:
                logging.error(f"Input file missing required columns: {missing}")
        except Exception:
             pass # Ignore error during error reporting
        sys.exit(1)

    # --- 2. Handle NaNs in Core PIDs ---
    logging.info("Handling NaNs in core PIDs using median imputation...")
    nan_counts = df.isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        logging.warning(f"Found NaNs in columns: \n{cols_with_nans}")
        for col in cols_with_nans.index:
            median_val = df[col].median()
            if pd.isna(median_val): # Handle column with all NaNs or issue computing median
                logging.warning(f"Median for column '{col}' is NaN. Filling NaNs with 0.")
                median_val = 0.0
            df[col].fillna(median_val, inplace=True)
            logging.info(f"  NaNs in '{col}' filled with median ({median_val:.2f})")
    else:
        logging.info("No NaNs found in the selected core PID columns.")

    if df.isnull().any().any():
         logging.error("Error: NaNs still present after imputation. Check data.")
         sys.exit(1)

    # --- 3. Train Scaler ---
    logging.info("Training StandardScaler on raw (NaN-filled) PIDs...")
    scaler = StandardScaler()
    try:
        scaler.fit(df)
        logging.info("StandardScaler trained successfully.")
        # Log means/stds for reference
        scaler_means = dict(zip(df.columns, scaler.mean_))
        scaler_stds = dict(zip(df.columns, np.sqrt(scaler.var_)))
        logging.info(f"Scaler Means: {json.dumps(scaler_means)}")
        logging.info(f"Scaler Stds: {json.dumps(scaler_stds)}")
    except Exception as e:
        logging.error(f"Error training StandardScaler: {e}")
        sys.exit(1)

    # --- 4. Apply Scaler ---
    logging.info("Applying trained scaler to the data...")
    try:
        df_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
        logging.info("Data scaled successfully.")
    except Exception as e:
        logging.error(f"Error applying scaler: {e}")
        sys.exit(1)

    # --- 5. Train Isolation Forest ---
    logging.info("Training Isolation Forest model...")
    model = IsolationForest(**iforest_params)
    try:
        model.fit(df_scaled)
        logging.info("Isolation Forest model trained successfully.")
    except Exception as e:
        logging.error(f"Error training Isolation Forest: {e}")
        sys.exit(1)

    # --- 6. Save Artifacts ---
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, f"{model_name_prefix}_scaler.joblib")
    model_path = os.path.join(model_dir, f"{model_name_prefix}_model.joblib")
    features_path = os.path.join(model_dir, f"{model_name_prefix}_features.json")

    try:
        logging.info(f"Saving scaler to: {scaler_path}")
        dump(scaler, scaler_path)

        logging.info(f"Saving model to: {model_path}")
        dump(model, model_path)

        # Save feature names used for training
        features_used = df.columns.tolist()
        logging.info(f"Saving feature list to: {features_path}")
        with open(features_path, 'w') as f:
            json.dump(features_used, f, indent=2)

        logging.info("Model artifacts saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        sys.exit(1)

    logging.info(f"--- Tier 1 Model Training for {model_name_prefix} Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Train a vehicle-specific Tier 1 Anomaly Detection model (Scaler + Isolation Forest).")
    parser.add_argument("--input_path", required=True, help="Path to the input Parquet file (e.g., *_raw_pids_final.parquet).")
    parser.add_argument("--model_dir", default="models/anomaly", help="Directory to save the trained model and scaler.")
    parser.add_argument("--model_name_prefix", required=True, help="Prefix for the saved model files (e.g., 'volvo_v40_raw_tier1').")
    parser.add_argument("--iforest_params", type=json.loads, default=json.dumps(DEFAULT_IFOREST_PARAMS), help='JSON string of Isolation Forest parameters (e.g., \'{"contamination": 0.01, "n_estimators": 150}\').')

    args = parser.parse_args()

    # Use default params and update with any user-provided ones
    iforest_config = DEFAULT_IFOREST_PARAMS.copy()
    iforest_config.update(args.iforest_params) # User args override defaults

    train_tier1_model(
        input_path=args.input_path,
        model_dir=args.model_dir,
        model_name_prefix=args.model_name_prefix,
        iforest_params=iforest_config
    )

if __name__ == "__main__":
    main()