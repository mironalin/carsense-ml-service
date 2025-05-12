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
import json

# Import the dtc lookup utility (we might use it later)
from app.preprocessing.dtc_lookup import load_dtc_data, get_dtc_description

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Definitions ---

# Tier 1: Minimal set of core PIDs expected to be widely available
# Used for the generic, initial anomaly detection model.
# Using actual column names found in exp1_14drivers... dataset
TIER1_CORE_PIDS = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",  # Was COOLANT_TEMPERATURE
    "AIR_INTAKE_TEMP",      # Was INTAKE_AIR_TEMPERATURE
    "THROTTLE_POS",         # Was THROTTLE_POSITION
    "SPEED",                # Was VEHICLE_SPEED
    "ENGINE_LOAD",          # Was CALCULATED_ENGINE_LOAD_VALUE
    # CONTROL_MODULE_VOLTAGE is excluded from Tier 1 training as requested
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
    coolant_col = "ENGINE_COOLANT_TEMP"
    time_col = "TIME_SEC" # Assuming 'TIME_SEC' holds elapsed seconds in trip
    runtime_threshold_sec = 120 # E.g., ignore first 2 minutes
    low_coolant_scaled_threshold = -1.5 # Define fixed scaled threshold (e.g., -1.5 std deviations)

    # Check if BOTH required columns are present
    if coolant_col in anomalous_data.columns and time_col in anomalous_data.columns:
        try:
            # Filter anomalies that occur after the initial runtime threshold
            # Also ensure TIME_SEC is not NaN for the comparison
            anomalies_after_warmup = anomalous_data[
                (anomalous_data[time_col].notna()) & (anomalous_data[time_col] > runtime_threshold_sec)
            ]

            if not anomalies_after_warmup.empty:
                # Apply the fixed scaled threshold
                low_coolant_anomalies = anomalies_after_warmup[
                    anomalies_after_warmup[coolant_col] < low_coolant_scaled_threshold
                ]
                count_low_coolant = len(low_coolant_anomalies)

                analysis_results["low_coolant_analysis"] = {
                    "threshold_scaled": low_coolant_scaled_threshold, # Report the fixed threshold used
                    "count_after_warmup": count_low_coolant,
                    "runtime_filter_sec": runtime_threshold_sec,
                    "insight": f"Detected anomalies exhibiting patterns consistent with low {coolant_col} (scaled < {low_coolant_scaled_threshold}) after expected warmup period. Possible thermostat or sensor issue."
                }
                logging.info(
                    f"Low {coolant_col} Heuristic (Runtime > {runtime_threshold_sec}s): "
                    f"Found {count_low_coolant} anomalies below scaled threshold {low_coolant_scaled_threshold:.2f}. Insight added."
                )
            else:
                 logging.info(f"Low {coolant_col} Heuristic: No anomalies found after runtime threshold ({runtime_threshold_sec}s).")
                 analysis_results["low_coolant_analysis"] = {
                     "count_after_warmup": 0,
                     "runtime_filter_sec": runtime_threshold_sec,
                     "message": "No anomalies occurred after the specified runtime threshold."
                 }

        except Exception as e:
            logging.error(f"Error during low {coolant_col} temperature analysis: {e}")
            analysis_results["low_coolant_analysis"] = {"error": str(e)}
    else:
        # Log which specific columns are missing
        missing_cols = []
        if coolant_col not in anomalous_data.columns:
            missing_cols.append(coolant_col)
        if time_col not in anomalous_data.columns:
            missing_cols.append(time_col)
        logging.warning(f"Skipping Low {coolant_col} Temperature analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["low_coolant_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Heuristic 3: High Engine RPM at Low Speed ---
    rpm_col = "ENGINE_RPM"
    speed_col = "SPEED"
    high_rpm_scaled_threshold = 1.5 # e.g., > 1.5 std deviations
    low_speed_scaled_threshold = -0.5 # e.g., < -0.5 std deviations (close to zero for scaled speed)

    if rpm_col in anomalous_data.columns and speed_col in anomalous_data.columns:
        try:
            high_rpm_low_speed_anomalies = anomalous_data[
                (anomalous_data[rpm_col] > high_rpm_scaled_threshold) &
                (anomalous_data[speed_col] < low_speed_scaled_threshold)
            ]
            count_high_rpm_low_speed = len(high_rpm_low_speed_anomalies)

            analysis_results["high_rpm_low_speed_analysis"] = {
                "rpm_threshold_scaled": high_rpm_scaled_threshold,
                "speed_threshold_scaled": low_speed_scaled_threshold,
                "count": count_high_rpm_low_speed,
                "insight": f"Detected anomalies exhibiting patterns consistent with high {rpm_col} (scaled > {high_rpm_scaled_threshold}) at very low {speed_col} (scaled < {low_speed_scaled_threshold}). Possible clutch/transmission issue or sensor error."
            }
            logging.info(
                f"High RPM / Low Speed Heuristic: "
                f"Found {count_high_rpm_low_speed} anomalies matching criteria (RPM > {high_rpm_scaled_threshold:.2f}, Speed < {low_speed_scaled_threshold:.2f}, scaled). Insight added."
            )
        except Exception as e:
            logging.error(f"Error during High RPM / Low Speed analysis: {e}")
            analysis_results["high_rpm_low_speed_analysis"] = {"error": str(e)}
    else:
        missing_cols = []
        if rpm_col not in anomalous_data.columns:
            missing_cols.append(rpm_col)
        if speed_col not in anomalous_data.columns:
            missing_cols.append(speed_col)
        logging.warning(f"Skipping High RPM / Low Speed analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["high_rpm_low_speed_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Heuristic 4: High Engine Load at Low RPM ---
    load_col = "ENGINE_LOAD"
    # rpm_col already defined as "ENGINE_RPM"
    high_load_scaled_threshold = 0.75 # Changed from 1.5
    low_rpm_scaled_threshold = -0.5

    if load_col in anomalous_data.columns and rpm_col in anomalous_data.columns:
        try:
            high_load_low_rpm_anomalies = anomalous_data[
                (anomalous_data[load_col] > high_load_scaled_threshold) &
                (anomalous_data[rpm_col] < low_rpm_scaled_threshold)
            ]
            count_high_load_low_rpm = len(high_load_low_rpm_anomalies)

            analysis_results["high_load_low_rpm_analysis"] = {
                "load_threshold_scaled": high_load_scaled_threshold,
                "rpm_threshold_scaled": low_rpm_scaled_threshold,
                "count": count_high_load_low_rpm,
                "insight": f"Detected anomalies consistent with high {load_col} (scaled > {high_load_scaled_threshold}) at low {rpm_col} (scaled < {low_rpm_scaled_threshold}). Possible engine lugging or sensor issue."
            }
            logging.info(
                f"High Load / Low RPM Heuristic: "
                f"Found {count_high_load_low_rpm} anomalies matching criteria (Load > {high_load_scaled_threshold:.2f}, RPM < {low_rpm_scaled_threshold:.2f}, scaled). Insight added."
            )
        except Exception as e:
            logging.error(f"Error during High Load / Low RPM analysis: {e}")
            analysis_results["high_load_low_rpm_analysis"] = {"error": str(e)}
    else:
        missing_cols = []
        if load_col not in anomalous_data.columns:
            missing_cols.append(load_col)
        if rpm_col not in anomalous_data.columns:
            missing_cols.append(rpm_col)
        logging.warning(f"Skipping High Load / Low RPM analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["high_load_low_rpm_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Heuristic 5: High Throttle Position with Low Engine Load ---
    tps_col = "THROTTLE_POS"
    # load_col already defined as "ENGINE_LOAD"
    high_tps_scaled_threshold = 0.75 # Changed from 1.5
    low_load_scaled_threshold = 0.0 # Below average load

    if tps_col in anomalous_data.columns and load_col in anomalous_data.columns:
        try:
            high_tps_low_load_anomalies = anomalous_data[
                (anomalous_data[tps_col] > high_tps_scaled_threshold) &
                (anomalous_data[load_col] < low_load_scaled_threshold)
            ]
            count_high_tps_low_load = len(high_tps_low_load_anomalies)

            analysis_results["high_tps_low_load_analysis"] = {
                "tps_threshold_scaled": high_tps_scaled_threshold,
                "load_threshold_scaled": low_load_scaled_threshold,
                "count": count_high_tps_low_load,
                "insight": f"Detected anomalies consistent with high {tps_col} (scaled > {high_tps_scaled_threshold}) but low {load_col} (scaled < {low_load_scaled_threshold}). Possible throttle/load sensor mismatch or other performance issue."
            }
            logging.info(
                f"High TPS / Low Load Heuristic: "
                f"Found {count_high_tps_low_load} anomalies matching criteria (TPS > {high_tps_scaled_threshold:.2f}, Load < {low_load_scaled_threshold:.2f}, scaled). Insight added."
            )
        except Exception as e:
            logging.error(f"Error during High TPS / Low Load analysis: {e}")
            analysis_results["high_tps_low_load_analysis"] = {"error": str(e)}
    else:
        missing_cols = []
        if tps_col not in anomalous_data.columns:
            missing_cols.append(tps_col)
        if load_col not in anomalous_data.columns:
            missing_cols.append(load_col)
        logging.warning(f"Skipping High TPS / Low Load analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["high_tps_low_load_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Heuristic 6: High Intake Air Temperature ---
    iat_col = "AIR_INTAKE_TEMP"
    high_iat_scaled_threshold = 2.0 # e.g., > 2 std deviations

    if iat_col in anomalous_data.columns:
        try:
            high_iat_anomalies = anomalous_data[anomalous_data[iat_col] > high_iat_scaled_threshold]
            count_high_iat = len(high_iat_anomalies)

            analysis_results["high_iat_analysis"] = {
                "iat_threshold_scaled": high_iat_scaled_threshold,
                "count": count_high_iat,
                "insight": f"Detected anomalies consistent with unusually high {iat_col} (scaled > {high_iat_scaled_threshold}). Possible sensor issue or heat soak condition."
            }
            logging.info(
                f"High IAT Heuristic: "
                f"Found {count_high_iat} anomalies matching criteria (IAT > {high_iat_scaled_threshold:.2f}, scaled). Insight added."
            )
        except Exception as e:
            logging.error(f"Error during High IAT analysis: {e}")
            analysis_results["high_iat_analysis"] = {"error": str(e)}
    else:
        logging.warning(f"Skipping High IAT analysis: Required column missing: {iat_col}.")
        analysis_results["high_iat_analysis"] = {"skipped": f"Required column missing: {iat_col}"}


    # --- Heuristic 7: Coolant Temp vs. Intake Air Temp (After Warmup) ---
    # coolant_col, iat_col, time_col defined earlier
    # runtime_threshold_sec defined earlier
    coolant_iat_diff_threshold = 0.5 # Scaled coolant temp should be at least this much higher than scaled IAT after warmup

    if coolant_col in anomalous_data.columns and iat_col in anomalous_data.columns and time_col in anomalous_data.columns:
        try:
            anomalies_after_warmup_h7 = anomalous_data[
                (anomalous_data[time_col].notna()) & (anomalous_data[time_col] > runtime_threshold_sec)
            ]
            if not anomalies_after_warmup_h7.empty:
                # Check where scaled coolant temp is not significantly higher than scaled IAT
                low_coolant_vs_iat_anomalies = anomalies_after_warmup_h7[
                    anomalies_after_warmup_h7[coolant_col] < (anomalies_after_warmup_h7[iat_col] + coolant_iat_diff_threshold)
                ]
                count_low_coolant_vs_iat = len(low_coolant_vs_iat_anomalies)

                analysis_results["coolant_vs_iat_analysis"] = {
                    "diff_threshold_scaled": coolant_iat_diff_threshold,
                    "count_after_warmup": count_low_coolant_vs_iat,
                    "runtime_filter_sec": runtime_threshold_sec,
                    "insight": f"Detected anomalies where scaled {coolant_col} (< {iat_col} + {coolant_iat_diff_threshold}) is not significantly higher than scaled {iat_col} after expected warmup. Possible coolant sensor or thermostat issue."
                }
                logging.info(
                    f"Coolant vs IAT Heuristic (Runtime > {runtime_threshold_sec}s): "
                    f"Found {count_low_coolant_vs_iat} anomalies where scaled Coolant Temp was not significantly higher than scaled IAT. Insight added."
                )
            else:
                 # Re-use message from Heuristic 2 if no anomalies after warmup
                 logging.info(f"Coolant vs IAT Heuristic: No anomalies found after runtime threshold ({runtime_threshold_sec}s).")
                 analysis_results["coolant_vs_iat_analysis"] = {
                     "count_after_warmup": 0,
                     "runtime_filter_sec": runtime_threshold_sec,
                     "message": "No anomalies occurred after the specified runtime threshold."
                 }
        except Exception as e:
            logging.error(f"Error during Coolant vs IAT analysis: {e}")
            analysis_results["coolant_vs_iat_analysis"] = {"error": str(e)}
    else:
        missing_cols = []
        if coolant_col not in anomalous_data.columns:
            missing_cols.append(coolant_col)
        if iat_col not in anomalous_data.columns:
            missing_cols.append(iat_col)
        if time_col not in anomalous_data.columns:
            missing_cols.append(time_col)
        logging.warning(f"Skipping Coolant vs IAT analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["coolant_vs_iat_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Heuristic 8: Low Engine Load at High Speed ---
    # load_col, speed_col defined earlier
    high_speed_scaled_threshold = 1.5
    very_low_load_scaled_threshold = 0.05 # Very close to minimum scaled load

    if load_col in anomalous_data.columns and speed_col in anomalous_data.columns:
        try:
            low_load_high_speed_anomalies = anomalous_data[
                (anomalous_data[speed_col] > high_speed_scaled_threshold) &
                (anomalous_data[load_col] < very_low_load_scaled_threshold)
            ]
            count_low_load_high_speed = len(low_load_high_speed_anomalies)

            analysis_results["low_load_high_speed_analysis"] = {
                "speed_threshold_scaled": high_speed_scaled_threshold,
                "load_threshold_scaled": very_low_load_scaled_threshold,
                "count": count_low_load_high_speed,
                "insight": f"Detected anomalies consistent with very low {load_col} (scaled < {very_low_load_scaled_threshold}) at high {speed_col} (scaled > {high_speed_scaled_threshold}). Possible sensor issue or coasting condition."
            }
            logging.info(
                f"Low Load / High Speed Heuristic: "
                f"Found {count_low_load_high_speed} anomalies matching criteria (Load < {very_low_load_scaled_threshold:.2f}, Speed > {high_speed_scaled_threshold:.2f}, scaled). Insight added."
            )
        except Exception as e:
            logging.error(f"Error during Low Load / High Speed analysis: {e}")
            analysis_results["low_load_high_speed_analysis"] = {"error": str(e)}
    else:
        missing_cols = []
        if load_col not in anomalous_data.columns:
            missing_cols.append(load_col)
        if speed_col not in anomalous_data.columns:
            missing_cols.append(speed_col)
        logging.warning(f"Skipping Low Load / High Speed analysis: Required columns missing: {', '.join(missing_cols)}.")
        analysis_results["low_load_high_speed_analysis"] = {"skipped": f"Required columns missing: {', '.join(missing_cols)}"}


    # --- Add more heuristics here ---


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

        # Load the DTC descriptions json file (needed for analysis context)
        # Assuming dtc.json is in the root directory - Use os.path.join correctly
        dtc_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dtc.json') # Correct usage
        try:
            dtc_data = load_dtc_data(dtc_data_path)
            logging.info(f"Successfully loaded DTC descriptions from {dtc_data_path}")
        except FileNotFoundError:
            logging.warning(f"DTC description file not found at {dtc_data_path}. Heuristic insights may be less specific.")
            dtc_data = {} # Proceed without DTC data
        except Exception as e:
            logging.error(f"Error loading DTC descriptions: {e}")
            dtc_data = {}

        # Load the scaler that was saved during training
        # Check if fuel_type_segment is provided before accessing attributes
        segment_tag = None
        if args.fuel_type_segment:
            segment_tag = args.fuel_type_segment.lower()
        segment_prefix = f"tier1_{segment_tag}_" if segment_tag else "tier1_generic_"
        scaler_filename = f"{segment_prefix}scaler.joblib"
        scaler_path = os.path.join(args.output_dir, scaler_filename)
        try:
            scaler = load(scaler_path)
            logging.info(f"Successfully loaded scaler from {scaler_path}")
        except FileNotFoundError:
            logging.error(f"Scaler file not found at {scaler_path}. Cannot perform analysis requiring the scaler.")
            # Exit or handle appropriately - for now, we'll log and potentially skip analysis
            scaler = None # Set scaler to None
        except Exception as e:
            logging.error(f"Error loading scaler: {e}")
            scaler = None

        # Only proceed with analysis if scaler loaded successfully
        if scaler:
            # Filter the DataFrame to get only the anomalous rows
            anomalous_df = df_full_with_anomalies[df_full_with_anomalies['anomaly'] == -1].copy()

            # Analyze the anomalies
            analysis_results = analyze_anomalies(
                anomalous_data=anomalous_df, # Pass only anomalous rows
                scaler=scaler,               # Pass the loaded scaler
                feature_names=actual_features_used, # Pass the list of features
                dtc_data=dtc_data            # Pass the loaded DTC data
            )

            logging.info("\nAnomaly Analysis Results:")
            # Log the results dictionary in a readable format
            logging.info(json.dumps(analysis_results, indent=4, default=str)) # Use default=str for non-serializable types like numpy numbers

            # --- Save anomaly data and analysis results if output path is provided ---
            if args.output_csv:
                # --- Save Analysis Results Dictionary (JSON) ---
                output_json_path = args.output_csv.replace('.csv', '_analysis.json') # Change extension
                logging.info(f"\nSaving analysis results dictionary to: {output_json_path}")
                try:
                    output_json_dir = os.path.dirname(output_json_path)
                    if output_json_dir and not os.path.exists(output_json_dir):
                        os.makedirs(output_json_dir, exist_ok=True)
                        logging.info(f"Created directory for output JSON: {output_json_dir}")
                    with open(output_json_path, 'w') as f:
                        json.dump(analysis_results, f, indent=4, default=str)
                    logging.info("Successfully saved analysis results to JSON.")
                except Exception as e:
                    logging.error(f"Error saving analysis results dictionary to JSON at '{output_json_path}': {e}")

                # --- Save Anomalous Data DataFrame (Parquet) ---
                # Reuse the base path from output_csv, change extension to .parquet
                output_parquet_path = args.output_csv.replace('.csv', '_anomalous_data.parquet')
                logging.info(f"\nSaving anomalous data DataFrame (with scaled features) to: {output_parquet_path}")
                try:
                    output_parquet_dir = os.path.dirname(output_parquet_path)
                    if output_parquet_dir and not os.path.exists(output_parquet_dir):
                        os.makedirs(output_parquet_dir, exist_ok=True)
                        logging.info(f"Created directory for output Parquet: {output_parquet_dir}")
                    # anomalous_df already contains the scaled features and other columns for anomalies
                    anomalous_df.to_parquet(output_parquet_path, index=True)
                    logging.info("Successfully saved anomalous data to Parquet.")
                except Exception as e:
                    logging.error(f"Error saving anomalous data DataFrame to Parquet at '{output_parquet_path}': {e}")

        else:
            logging.error("Skipping anomaly analysis and saving due to failure in loading the scaler.")

        logging.info("\nAnomaly detection script finished.") # Removed 'successfully' as analysis might be skipped

    except FileNotFoundError as fnf_error:
        logging.error(f"File Error: {fnf_error}")
        sys.exit(1)
    except ValueError as val_error:
        logging.error(f"Value Error: {val_error}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
        sys.exit(1)