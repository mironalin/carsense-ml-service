import pandas as pd
import numpy as np
from joblib import load
import os
import sys
import logging
import json
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler # For type hinting and scaler loading

# Assuming anomaly_detection.py is in app/preprocessing relative to project root
# Adjust sys.path if necessary, or structure as a package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Import necessary functions from the training script
try:
    from app.preprocessing.anomaly_detection import analyze_anomalies, TIER1_CORE_PIDS, load_dtc_data
except ImportError as e:
    logging.error(f"Could not import from app.preprocessing.anomaly_detection. Make sure the path is correct and the file exists. Error: {e}")
    sys.exit(1)

# --- Configuration ---
# Paths relative to project root
DATA_PATH = os.path.join(project_root, "data/model_input/romanian_renamed_raw_pids_for_generic_tier1.parquet")
MODEL_PATH = os.path.join(project_root, "models/anomaly/tier1_combined_general_isolation_forest.joblib")
SCALER_PATH = os.path.join(project_root, "models/anomaly/tier1_combined_general_scaler.joblib")
IMPUTER_PATH = os.path.join(project_root, "models/anomaly/tier1_combined_general_imputer.joblib")
DTC_JSON_PATH = os.path.join(project_root, "dtc.json")
OUTPUT_ANOMALY_ANALYSIS_LOG = os.path.join(project_root, "logs/combined_general_tier1_test_on_romanian_raw_pids_analysis.log")
OUTPUT_PREDICTIONS_PATH = os.path.join(project_root, "data/processed/combined_general_tier1_test_on_romanian_raw_pids_predictions.parquet")
# Flag to indicate if input data is already scaled
INPUT_DATA_PRESCALED = False # Crucial: new model expects raw (or rather, data scaled by *its* scaler)

# Context columns expected in the input data (adjust if needed based on DATA_PATH)
# These are kept alongside predictions but not used by the model itself.
CONTEXT_COLUMNS = [
    "TIME_SEC",
    "absolute_timestamp",
    "hour",
    "dayofweek",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "source_file",
    # "event_type", # Temporarily removed as it's not in the current Volvo test data
    "make",
    "model",
    "fuel_type"
]

# Setup logging
# Ensure logs directory exists
os.makedirs(os.path.dirname(OUTPUT_ANOMALY_ANALYSIS_LOG), exist_ok=True)
# Configure logging to file and console
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# File handler
file_handler = logging.FileHandler(OUTPUT_ANOMALY_ANALYSIS_LOG, mode='w') # Overwrite log each run
file_handler.setFormatter(log_formatter)
# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
# Root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# --- End Configuration ---

def load_model_scaler_imputer(model_path: str, scaler_path: str, imputer_path: str):
    """Loads the pre-trained model, scaler, and imputer."""
    logger.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    model = load(model_path)

    logger.info(f"Loading scaler from: {scaler_path}")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        sys.exit(1)
    scaler = load(scaler_path)

    logger.info(f"Loading imputer from: {imputer_path}")
    if not os.path.exists(imputer_path):
        logger.error(f"Imputer file not found: {imputer_path}")
        sys.exit(1)
    imputer = load(imputer_path)

    # Basic check: ensure scaler has expected attributes
    if not hasattr(scaler, 'transform') or not hasattr(scaler, 'scale_'):
         logger.error("Loaded scaler object does not appear to be a valid StandardScaler.")
         sys.exit(1)
    if not hasattr(imputer, 'transform'):
         logger.error("Loaded imputer object does not appear to be a valid SimpleImputer.")
         sys.exit(1)

    logger.info("Model, scaler, and imputer loaded successfully.")
    return model, scaler, imputer

def load_testing_data(data_path: str, required_features: List[str], context_features: List[str]) -> pd.DataFrame:
    """Loads the dataset for testing."""
    logger.info(f"Loading testing data from: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Testing data file not found: {data_path}")
        sys.exit(1)
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")

        # Verify required feature and context columns are present
        all_required_cols = required_features + context_features
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Input data is missing required columns: {missing_cols}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)

        return df
    except Exception as e:
        logger.error(f"Error loading Parquet file: {e}")
        sys.exit(1)

def preprocess_for_prediction(df: pd.DataFrame, features_to_scale: List[str], imputer: Any, scaler: StandardScaler, is_prescaled: bool = False) -> pd.DataFrame:
    """Handles NaNs using the pre-loaded imputer and optionally scales the specified features using the pre-loaded scaler."""
    logger.info(f"Preprocessing data for prediction (is_prescaled={is_prescaled})...")
    df_features = df[features_to_scale].copy() # Work on a copy

    # 1. Impute missing values using the loaded imputer
    logger.info(f"Applying imputer to features: {features_to_scale}")
    try:
        df_imputed_values = imputer.transform(df_features)
        df_imputed = pd.DataFrame(df_imputed_values, columns=df_features.columns, index=df_features.index)

        # Log if any NaNs were actually imputed by checking original vs imputed (optional)
        for col in features_to_scale:
            original_nans = df_features[col].isnull().sum()
            if original_nans > 0:
                # After imputation, there should be 0 NaNs in df_imputed[col]
                # This logging confirms imputation happened.
                logger.info(f"Imputer handled {original_nans} NaN values in column '{col}'.")
        logger.info("Imputation complete.")
        df_features_to_scale = df_imputed # Use imputed data for potential scaling
    except Exception as e:
        logger.error(f"Error during imputation: {e}")
        logger.error(f"Imputer features seen during fit: {getattr(imputer, 'feature_names_in_', 'N/A')}")
        logger.error(f"Data features provided for imputation: {features_to_scale}")
        sys.exit(1)

    if is_prescaled:
        logger.info("Skipping scaling as input data is marked as pre-scaled.")
        return df_features_to_scale # Return the imputed (but not re-scaled) features
    else:
        # Scale data using the loaded scaler
        try:
            # Ensure columns are in the same order as the scaler expects, if possible
            expected_cols = []
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = list(scaler.feature_names_in_)
                if set(df_features_to_scale.columns.tolist()) != set(expected_cols):
                     logger.warning(f"Feature columns for scaling {df_features_to_scale.columns.tolist()} do not exactly match scaler's expected features {expected_cols}. Attempting to reorder.")
                     # Attempt to reorder if all expected columns are present
                     if set(expected_cols).issubset(set(df_features_to_scale.columns.tolist())):
                         logger.info(f"Reordering columns to match scaler: {expected_cols}")
                         df_features_to_scale = df_features_to_scale[expected_cols]
                     else:
                         logger.error("Cannot proceed: Data columns for scaling mismatch scaler's expected features significantly.")
                         sys.exit(1)
                else:
                     # Ensure order matches
                     df_features_to_scale = df_features_to_scale[expected_cols]
            else:
                logger.warning("Scaler does not have 'feature_names_in_' attribute. Scaling features as is.")

            logger.info(f"Scaling features: {df_features_to_scale.columns.tolist()}")
            df_scaled = scaler.transform(df_features_to_scale)
            df_scaled_pd = pd.DataFrame(df_scaled, columns=df_features_to_scale.columns, index=df_features_to_scale.index)
            logger.info("Scaling complete.")
            return df_scaled_pd
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            logger.error(f"Scaler expected features: {getattr(scaler, 'feature_names_in_', 'N/A')}")
            logger.error(f"Data features provided: {df_features_to_scale.columns.tolist()}")
            sys.exit(1)

def predict_anomalies(model, df_features: pd.DataFrame) -> np.ndarray:
    """Uses the loaded model to predict anomalies on the provided feature data."""
    # Expects df_features to be appropriately scaled (or pre-scaled)
    logger.info("Predicting anomalies using the loaded model...")
    predictions = model.predict(df_features) # Returns -1 for anomalies, 1 for inliers
    num_anomalies = np.sum(predictions == -1)
    total_preds = len(predictions)
    anomaly_ratio = num_anomalies / total_preds if total_preds > 0 else 0
    logger.info(f"Prediction complete. Found {num_anomalies} anomalies out of {total_preds} samples ({anomaly_ratio:.2%}).")
    return predictions

def main():
    """Main execution function."""
    logger.info("--- Starting Tier 1 Anomaly Model Testing (with Combined General Model) ---")

    # 1. Load Model, Scaler, and Imputer
    model, scaler, imputer = load_model_scaler_imputer(MODEL_PATH, SCALER_PATH, IMPUTER_PATH)

    # Check scaler features if available
    scaler_features = []
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        logger.info(f"Scaler was trained on features: {scaler_features}")
        # Verify TIER1_CORE_PIDS match the scaler's features
        if set(TIER1_CORE_PIDS) != set(scaler_features):
            logger.warning(f"TIER1_CORE_PIDS defined in script {TIER1_CORE_PIDS} do not match scaler features {scaler_features}. Using scaler features.")
            features_to_use = scaler_features
        else:
            features_to_use = TIER1_CORE_PIDS # Order might matter, handled in preprocess
    else:
        logger.warning("Scaler does not have 'feature_names_in_' attribute. Assuming TIER1_CORE_PIDS are correct.")
        features_to_use = TIER1_CORE_PIDS

    # 2. Load Testing Data
    df_test = load_testing_data(DATA_PATH, features_to_use, CONTEXT_COLUMNS)
    original_index = df_test.index # Preserve index if needed

    # 3. Preprocess Data (Select features, Impute, Scale if necessary)
    df_processed_features = preprocess_for_prediction(
        df_test,
        features_to_use,
        imputer, # Pass the loaded imputer
        scaler,
        is_prescaled=INPUT_DATA_PRESCALED
    )

    # 4. Predict Anomalies
    # Pass the processed features (scaled or pre-scaled) to the model
    anomaly_predictions = predict_anomalies(model, df_processed_features)

    # 5. Add predictions to the original DataFrame
    # Ensure index alignment (should be okay if preprocess didn't change it)
    df_test['anomaly'] = anomaly_predictions
    # Restore original index if it was changed (it shouldn't be here)
    if not df_test.index.equals(original_index):
         logger.info("Index mismatch detected after prediction, restoring original index.")
         df_test.set_index(original_index, inplace=True)

    # 6. Filter Anomalous Data for Analysis
    df_anomalous = df_test[df_test['anomaly'] == -1].copy()
    logger.info(f"Extracted {len(df_anomalous)} anomalous rows for analysis.")

    # 7. Analyze Anomalies if any were found
    if not df_anomalous.empty:
        # Load DTC data
        dtc_data = load_dtc_data(DTC_JSON_PATH)
        if not dtc_data:
             logger.warning("Could not load DTC data. Analysis will proceed without descriptions.")
             dtc_data = {} # Ensure it's a dict

        logger.info("Running heuristic analysis on anomalous data points...")
        # Pass the original anomalous data (unscaled features but correct names)
        # The scaler is passed in case analysis needs it (e.g., for threshold interpretation)
        # feature_names should be the ones used for scaling/prediction
        # Pass the df_anomalous which contains the *original* (pre-scaled) values + context
        # The analyze_anomalies function needs the *original* (or pre-scaled) values
        # along with the scaler object for context / threshold interpretation.
        analysis_results = analyze_anomalies(
            anomalous_data=df_anomalous, # Contains original values + context
            scaler=scaler, # Pass scaler for context even if data was pre-scaled
            feature_names=features_to_use,
            dtc_data=dtc_data
        )

        # Log the analysis results (nicely formatted)
        logger.info("--- Anomaly Analysis Results ---")
        logger.info(json.dumps(analysis_results, indent=2))
        logger.info("--- End Anomaly Analysis Results ---")
    else:
        logger.info("No anomalies detected. Skipping analysis.")

    # 8. Optional: Save predictions
    if OUTPUT_PREDICTIONS_PATH:
         try:
             output_pred_dir = os.path.dirname(OUTPUT_PREDICTIONS_PATH)
             if output_pred_dir and not os.path.exists(output_pred_dir):
                  os.makedirs(output_pred_dir, exist_ok=True)
             # Save the full dataframe with the 'anomaly' column
             df_test.to_parquet(OUTPUT_PREDICTIONS_PATH, index=False)
             logger.info(f"Saved DataFrame with anomaly predictions to: {OUTPUT_PREDICTIONS_PATH}")
         except Exception as e:
              logger.error(f"Error saving predictions Parquet file to {OUTPUT_PREDICTIONS_PATH}: {e}")


    logger.info("--- Tier 1 Anomaly Model Testing Finished ---")

if __name__ == "__main__":
    main()