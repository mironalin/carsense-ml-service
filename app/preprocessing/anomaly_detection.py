import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import argparse # Import argparse
import sys # Import sys for exit
import logging # Import logging
from typing import List, Dict, Any, Optional

# Import the dtc lookup utility (we might use it later)
from app.preprocessing.dtc_lookup import load_dtc_data, get_dtc_description

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Definitions ---

# Tier 1: Minimal set of core PIDs expected to be widely available
# Used for the generic, initial anomaly detection model.
TIER1_CORE_PIDS = [
    "ENGINE_RPM",
    "COOLANT_TEMPERATURE",
    "INTAKE_AIR_TEMPERATURE",
    "THROTTLE_POSITION",
    "VEHICLE_SPEED",
    "CALCULATED_ENGINE_LOAD_VALUE",
]

# Tier 2/Comprehensive: More extensive list used for detailed analysis
# or models trained on specific make/models where available.
CORE_PIDS_FOR_ANOMALY = [
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

# We might also include some key derived features if available and deemed important
# Example: Rolling averages or differences of key PIDs
DERIVED_FEATURES_FOR_ANOMALY = [
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


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a Parquet file."""
    logging.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found at: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading Parquet file: {e}")
        raise


def select_features(df: pd.DataFrame, core_pids: list, derived_features: list) -> pd.DataFrame:
    """Selects relevant features for anomaly detection."""
    logging.info("Selecting features...")
    available_core = [pid for pid in core_pids if pid in df.columns]
    available_derived = [feat for feat in derived_features if feat in df.columns]

    missing_core = set(core_pids) - set(available_core)
    if missing_core:
        logging.warning(f"Core PIDs not found in DataFrame: {missing_core}")

    missing_derived = set(derived_features) - set(available_derived)
    if missing_derived:
        logging.warning(f"Derived features not found in DataFrame: {missing_derived}")

    features_to_use = available_core + available_derived
    if not features_to_use:
        raise ValueError("No features selected for anomaly detection. Check PIDs and derived feature names.")

    logging.info(f"Selected features ({len(features_to_use)}): {features_to_use}")
    return df[features_to_use]


def preprocess_data(df: pd.DataFrame, scaler_path: str = None) -> (pd.DataFrame, StandardScaler):
    """Handles NaNs and scales the data. Saves or loads the scaler."""
    logging.info("Preprocessing data (handling NaNs, scaling)...")
    # Simple NaN handling: fill with the mean of the column
    # More sophisticated strategies might be needed (e.g., interpolation)
    for col in df.columns:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            # Check if mean_val is NaN (happens if all values are NaN)
            if pd.isna(mean_val):
                logging.warning(f"Column '{col}' consists entirely of NaNs. Filling with 0.")
                mean_val = 0 # Or another default value
            df[col].fillna(mean_val, inplace=True)
            # Use round() for potentially large numbers, ensure it's float before formatting
            log_val = round(float(mean_val), 2) if not pd.isna(mean_val) else 0
            logging.info(f"  NaNs in '{col}' filled with mean ({log_val}) or 0 if all NaNs")


    if scaler_path and os.path.exists(scaler_path):
        logging.info(f"Loading existing scaler from: {scaler_path}")
        scaler = load(scaler_path)
        # Ensure columns match before transforming
        if hasattr(scaler, 'feature_names_in_'):
             expected_cols = list(scaler.feature_names_in_)
             if list(df.columns) != expected_cols:
                  logging.warning(f"Scaler columns {expected_cols} differ from DataFrame columns {list(df.columns)}. Attempting transform anyway.")
        try:
            df_scaled = scaler.transform(df.copy())
        except ValueError as e:
             logging.error(f"Error transforming data with loaded scaler: {e}. Columns might mismatch.")
             logging.error(f"Scaler expected: {getattr(scaler, 'feature_names_in_', 'N/A')}")
             logging.error(f"Data has: {list(df.columns)}")
             raise
    else:
        logging.info("Fitting new StandardScaler...")
        scaler = StandardScaler()
        # Fit on a copy to avoid changing the original df if it's a slice
        df_scaled = scaler.fit_transform(df.copy())
        if scaler_path:
            # Ensure directory exists before saving
            scaler_dir = os.path.dirname(scaler_path)
            if scaler_dir and not os.path.exists(scaler_dir):
                 os.makedirs(scaler_dir, exist_ok=True)
                 logging.info(f"Created directory for scaler: {scaler_dir}")

            dump(scaler, scaler_path)
            logging.info(f"Scaler saved to {scaler_path}")

    return pd.DataFrame(df_scaled, columns=df.columns, index=df.index), scaler


def train_and_predict_anomalies(
    data_path: str,
    output_dir: str = "models/anomaly",
    contamination: float = 0.01, # Expected proportion of anomalies
    vehicle_segment_tag: Optional[str] = None # e.g., "gasoline", "diesel"
) -> pd.DataFrame:
    """
    Loads data, preprocesses, trains Isolation Forest for a specific vehicle segment,
    predicts anomalies, and saves the model and scaler.
    Assumes data_path contains data for the specified segment.

    Args:
        data_path (str): Path to the input Parquet file (pre-filtered for segment).
        output_dir (str): Directory to save the model and scaler.
        contamination (float): The expected proportion of outliers in the data set.
        vehicle_segment_tag (Optional[str]): Tag for the vehicle segment (e.g., "gasoline", "diesel").
                                             If None, uses a generic name.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'anomaly' column (-1 for anomalies, 1 for normal).
    """
    segment_prefix = f"tier1_{vehicle_segment_tag.lower()}_" if vehicle_segment_tag else "tier1_generic_"
    model_filename = f"{segment_prefix}isolation_forest.joblib"
    scaler_filename = f"{segment_prefix}scaler.joblib"

    logging.info(f"--- Running Tier 1 Anomaly Detection for: {data_path} (Segment: {vehicle_segment_tag or 'Generic'}) ---")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Model file: {model_filename}, Scaler file: {scaler_filename}")
    logging.info(f"Contamination rate: {contamination}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_filename)
    scaler_path = os.path.join(output_dir, scaler_filename)

    # 1. Load Data
    df_full = load_data(data_path)
    original_index = df_full.index

    # 2. Select Features (Strictly TIER1_CORE_PIDS for Tier 1 models)
    logging.info(f"Selecting features for Tier 1 model (Segment: {vehicle_segment_tag or 'Generic'}). Expecting: {TIER1_CORE_PIDS}")
    available_tier1_pids = [pid for pid in TIER1_CORE_PIDS if pid in df_full.columns]
    missing_tier1_pids = set(TIER1_CORE_PIDS) - set(available_tier1_pids)

    if missing_tier1_pids:
        # For Tier 1, we are stricter. If essential PIDs are missing from the pre-segmented data, it's an issue.
        logging.error(f"Critical TIER1_CORE_PIDS missing from input data '{data_path}' for segment '{vehicle_segment_tag or 'Generic'}': {missing_tier1_pids}")
        raise ValueError(f"Missing essential Tier 1 PIDs: {missing_tier1_pids}. Cannot train model for this segment.")
    
    if len(available_tier1_pids) < len(TIER1_CORE_PIDS):
        # This case should ideally be caught by the check above, but as a safeguard:
        logging.warning(f"Some TIER1_CORE_PIDS were found but not all. Using: {available_tier1_pids}")
        # Depending on policy, one might choose to raise an error here too.

    df_features = df_full[available_tier1_pids]
    actual_features_used = df_features.columns.tolist()
    logging.info(f"Using features for Tier 1 model: {actual_features_used}")

    # 3. Preprocess Data (Scaling, NaN handling)
    # For segmented models, ensure scaler is specific to this segment.
    # Removing existing scaler to ensure refit for THIS segment's data characteristics.
    if os.path.exists(scaler_path):
         logging.info(f"Removing existing scaler to ensure refit for segment '{vehicle_segment_tag or 'Generic'}': {scaler_path}")
         try:
             os.remove(scaler_path)
         except OSError as e:
             logging.error(f"Error removing existing scaler for segment: {e}")
             # Potentially raise error or exit

    df_scaled, scaler = preprocess_data(df_features.copy(), scaler_path)

    # 4. Train Isolation Forest Model
    logging.info("Training Isolation Forest model...")
    # Consider making n_estimators, max_samples, etc., configurable
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(df_scaled)
    logging.info(f"Saving model to: {model_path}")
    dump(model, model_path)
    logging.info("Model training complete.")

    # 5. Predict Anomalies
    logging.info("Predicting anomalies...")
    # predict returns -1 for outliers and 1 for inliers.
    anomaly_predictions = model.predict(df_scaled)
    logging.info(f"Prediction complete. Found {np.sum(anomaly_predictions == -1)} potential anomalies.")

    # Add predictions back to the original DataFrame
    # Ensure index alignment if preprocessing changed it (it shouldn't here)
    df_full['anomaly'] = anomaly_predictions
    # Restore index *before* potentially adding analysis columns
    if not df_full.index.equals(original_index):
         logging.info("Restoring original index...")
         df_full.set_index(original_index, inplace=True)

    # Add the list of features used to the dataframe attributes for later use
    df_full.attrs['features_used'] = actual_features_used

    # Optional: Add anomaly scores as well
    # anomaly_scores = model.decision_function(df_scaled)
    # df_full['anomaly_score'] = anomaly_scores

    return df_full


# Helper function to lookup multiple DTCs
def lookup_dtcs(dtc_list: List[str], dtc_data: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Looks up descriptions for a list of DTCs."""
    results = {}
    for code in dtc_list:
        results[code] = get_dtc_description(code, dtc_data)
    return results


def analyze_anomalies(
    anomalous_data: pd.DataFrame,
    scaler: StandardScaler, # Scaler might be needed for inverse transform later
    feature_names: List[str], # Features used by the model
    dtc_data: Dict[str, Dict[str, str]], # Loaded DTC descriptions
    # Removed model_name and dataset_name as they weren't used
) -> Dict[str, Any]:
    """
    Analyzes the detected anomalies to provide more context.
    Adds heuristics to identify potential root causes like low voltage or coolant issues.
    Returns a dictionary summarizing the analysis.
    """
    analysis_results = {"total_anomalies": len(anomalous_data)}
    logging.info(f"Starting anomaly analysis for {len(anomalous_data)} anomalies...")

    if anomalous_data.empty:
        logging.info("No anomalies detected to analyze.")
        return analysis_results

    # --- Heuristic 1: Low Control Module Voltage ---
    voltage_col = "CONTROL_MODULE_VOLTAGE"
    if voltage_col in anomalous_data.columns:
        try:
            # Calculate threshold based on the 25th percentile of anomalous voltage values
            voltage_threshold = anomalous_data[voltage_col].quantile(0.25)
            low_voltage_anomalies = anomalous_data[anomalous_data[voltage_col] < voltage_threshold]
            count_low_voltage = len(low_voltage_anomalies)
            analysis_results["low_voltage_analysis"] = {
                "threshold_scaled": voltage_threshold,
                "count": count_low_voltage,
                "insight": "Detected anomalies exhibiting patterns consistent with low system voltage."
            }
            logging.info(f"Low Voltage Heuristic: Found {count_low_voltage} anomalies below threshold {voltage_threshold:.2f} (scaled). Insight added.")
            # TODO: Add logic to check if these anomalies cluster in specific trips/times
        except Exception as e:
            logging.error(f"Error during low voltage analysis: {e}")
            analysis_results["low_voltage_analysis"] = {"error": str(e)}
    else:
        logging.warning(f"Skipping Low Voltage analysis: Column '{voltage_col}' not found.")
        analysis_results["low_voltage_analysis"] = {"skipped": f"Column '{voltage_col}' not found."}


    # --- Heuristic 2: Low Coolant Temperature (after sufficient runtime) ---
    coolant_col = "COOLANT_TEMPERATURE"
    time_col = "TIME_SEC" # Assuming 'TIME_SEC' holds elapsed seconds in trip
    runtime_threshold_sec = 120 # E.g., ignore first 2 minutes

    # Check if BOTH required columns are present
    if coolant_col in anomalous_data.columns and time_col in anomalous_data.columns:
        try:
            # Filter anomalies that occur after the initial runtime threshold
            anomalies_after_warmup = anomalous_data[anomalous_data[time_col] > runtime_threshold_sec]

            if not anomalies_after_warmup.empty:
                # Calculate threshold based on 25th percentile of *filtered* anomalous coolant temps
                coolant_threshold = anomalies_after_warmup[coolant_col].quantile(0.25)
                low_coolant_anomalies = anomalies_after_warmup[
                    anomalies_after_warmup[coolant_col] < coolant_threshold
                ]
                count_low_coolant = len(low_coolant_anomalies)

                analysis_results["low_coolant_analysis"] = {
                    "threshold_scaled": coolant_threshold,
                    "count_after_warmup": count_low_coolant,
                    "runtime_filter_sec": runtime_threshold_sec,
                    "insight": "Detected anomalies exhibiting patterns consistent with low coolant temperature after expected warmup period. Possible thermostat or sensor issue."
                }
                logging.info(
                    f"Low Coolant Heuristic (Runtime > {runtime_threshold_sec}s): "
                    f"Found {count_low_coolant} anomalies below threshold {coolant_threshold:.2f} (scaled). Insight added."
                )
            else:
                 logging.info(f"Low Coolant Heuristic: No anomalies found after runtime threshold ({runtime_threshold_sec}s).")
                 analysis_results["low_coolant_analysis"] = {
                     "count_after_warmup": 0,
                     "runtime_filter_sec": runtime_threshold_sec,
                     "message": "No anomalies occurred after the specified runtime threshold."
                 }

        except Exception as e:
            logging.error(f"Error during low coolant temperature analysis: {e}")
            analysis_results["low_coolant_analysis"] = {"error": str(e)}
    else:
        # Log which specific columns are missing
        missing_cols = []
        if coolant_col not in anomalous_data.columns:
            missing_cols.append(coolant_col)
        if time_col not in anomalous_data.columns:
            missing_cols.append(time_col)
        logging.warning(f"Skipping Low Coolant Temperature analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["low_coolant_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Add more heuristics here ---
    # Example: High MAF readings, Erratic TPS, etc.


    # --- Summary/Further Analysis ---
    # Potentially analyze distribution of anomalies over time, features involved etc.
    # Example: Feature contribution (requires model-specific methods like SHAP if using complex models,
    # or can look at feature distributions for simpler models like IF)

    # Get descriptive stats for anomalous data (on original scale if possible, needs inverse transform)
    # Note: Inverse transform might be tricky if scaler wasn't fit on the exact same columns
    # present in anomalous_data. For now, show stats on scaled data.
    try:
        # Ensure feature_names only includes columns actually present in anomalous_data
        available_features = [f for f in feature_names if f in anomalous_data.columns]
        if available_features:
             analysis_results["anomaly_stats_scaled"] = anomalous_data[available_features].describe().to_dict()
        else:
             logging.warning("Could not generate anomaly stats: No features used by the model were present in the anomalous data.")
             analysis_results["anomaly_stats_scaled"] = {"error": "No relevant features found in anomalous data."}
    except KeyError as e:
        logging.warning(f"Could not generate anomaly stats: Feature mismatch? Error: {e}")
        analysis_results["anomaly_stats_scaled"] = {"error": f"Feature mismatch: {e}"}
    except Exception as e:
        logging.error(f"Error generating anomaly stats: {e}")
        analysis_results["anomaly_stats_scaled"] = {"error": str(e)}


    logging.info("Anomaly analysis finished.")
    return analysis_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest for anomaly detection on OBD data and map potential DTCs.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True, 
        help="Path to the input Parquet file (assumed to be pre-filtered for the specified fuel_type_segment)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True, 
        help="Directory to save the model and scaler."
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Expected proportion of anomalies in the data (contamination factor). Default: 0.02"
    )
    parser.add_argument(
        "--output-csv", 
        type=str,
        default=None,
        help="Optional path to save the DataFrame with detected anomalies and potential DTCs as a CSV file."
    )
    parser.add_argument(
        "--fuel-type-segment", # New argument for segment tag
        type=str,
        default=None, # Make it optional; if None, a generic model is trained
        choices=["gasoline", "diesel"], # Example choices, can be expanded
        help="Fuel type segment to train the model for (e.g., 'gasoline', 'diesel'). Affects model/scaler naming."
    )

    args = parser.parse_args()

    logging.info(f"Running Anomaly Detection script with args: {args}")

    # Check if the data file exists (already done in load_data, but good practice here too)
    if not os.path.exists(args.data_path):
        logging.error(f"Error: Data file not found at '{args.data_path}'.")
        sys.exit(1) # Exit if data file not found

    try:
        # Run the main training and prediction process
        df_full_with_anomalies = train_and_predict_anomalies(
            data_path=args.data_path,
            output_dir=args.output_dir,
            contamination=args.contamination,
            vehicle_segment_tag=args.fuel_type_segment # Pass the segment tag
        )

        # Get the features used from the DataFrame attribute
        actual_features_used = df_full_with_anomalies.attrs.get('features_used', [])
        if not actual_features_used:
             logging.warning("Could not retrieve list of features used from DataFrame attributes.")
             # Fallback (less reliable if columns were missing)
             actual_features_used = TIER1_CORE_PIDS


        # Display info about anomalies found
        anomaly_count = df_full_with_anomalies[df_full_with_anomalies['anomaly'] == -1].shape[0]
        logging.info("\nAnomaly Detection Summary:")
        logging.info(f"Processed data shape: {df_full_with_anomalies.shape}")
        logging.info(f"Total potential anomalies detected: {anomaly_count}")

        # Analyze the anomalies and get the DataFrame with potential DTCs
        analyzed_anomalies_df = analyze_anomalies(df_full_with_anomalies, actual_features_used)

        # --- Display anomalies with mapped DTCs ---
        # Ensure the potential_dtcs column exists and is of list type before filtering
        if 'potential_dtcs' in analyzed_anomalies_df.columns:
             anomalies_with_dtcs = analyzed_anomalies_df[analyzed_anomalies_df['potential_dtcs'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
             if not anomalies_with_dtcs.empty:
                 logging.info("\n\n--- Anomalies with Potential DTCs Mapped ---")
                 logging.info(f"Found {anomalies_with_dtcs.shape[0]} anomalies associated with potential DTCs based on heuristics.")
                 logging.info("Sample (First 10 rows with mapped DTCs):")
                 # Select a few key columns for concise logging
                 log_cols = ['TIME_SEC', 'CONTROL_MODULE_VOLTAGE', 'COOLANT_TEMPERATURE', 'potential_dtcs']
                 log_cols = [c for c in log_cols if c in anomalies_with_dtcs.columns] # Ensure columns exist
                 try:
                      logging.info("\n" + anomalies_with_dtcs[log_cols].head(10).to_string())
                 except Exception as e:
                      logging.error(f"Could not display sample anomalies with DTCs: {e}")
             else:
                 logging.info("\n\nNo anomalies were associated with potential DTCs based on the current heuristics.")
        else:
             logging.warning("'potential_dtcs' column not found in analyzed anomalies DataFrame.")


        # --- Save results if output path is provided ---
        if args.output_csv:
            output_csv_path = args.output_csv
            logging.info(f"\nSaving analyzed anomalies with potential DTCs to: {output_csv_path}")
            try:
                # Ensure directory exists
                output_csv_dir = os.path.dirname(output_csv_path)
                if output_csv_dir and not os.path.exists(output_csv_dir):
                    os.makedirs(output_csv_dir, exist_ok=True)
                    logging.info(f"Created directory for output CSV: {output_csv_dir}")

                # Convert list column to string for CSV compatibility if needed
                # df_to_save = analyzed_anomalies_df.copy()
                # if 'potential_dtcs' in df_to_save.columns:
                #     df_to_save['potential_dtcs'] = df_to_save['potential_dtcs'].astype(str)

                # Save only the anomalous rows
                analyzed_anomalies_df.to_csv(output_csv_path, index=True) # Keep index
                logging.info("Successfully saved anomalies to CSV.")
            except Exception as e:
                logging.error(f"Error saving anomalies DataFrame to CSV at '{output_csv_path}': {e}")

        logging.info("\nAnomaly detection script finished successfully.")

    except FileNotFoundError as fnf_error:
        logging.error(f"File Error: {fnf_error}")
        sys.exit(1)
    except ValueError as val_error:
        logging.error(f"Value Error: {val_error}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
        sys.exit(1)