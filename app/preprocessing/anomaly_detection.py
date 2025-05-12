import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import argparse # Import argparse
import sys # Import sys for exit
import logging # Import logging

# Import the dtc lookup utility (we might use it later)
from app.preprocessing.dtc_lookup import load_dtc_data, get_dtc_description

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get RELEVANT_PIDS from feature_engineering.py (or define a subset here)
# For now, let's define a core subset relevant for anomaly detection.
# This avoids direct dependency but should be kept in sync or imported.
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
    model_filename: str = "isolation_forest.joblib",
    scaler_filename: str = "scaler.joblib",
    contamination: float = 0.01 # Expected proportion of anomalies
) -> pd.DataFrame:
    """
    Loads data, preprocesses, trains Isolation Forest, predicts anomalies,
    and saves the model and scaler.

    Args:
        data_path (str): Path to the input Parquet file.
        output_dir (str): Directory to save the model and scaler.
        model_filename (str): Filename for the saved Isolation Forest model.
        scaler_filename (str): Filename for the saved StandardScaler.
        contamination (float): The expected proportion of outliers in the data set.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'anomaly' column (-1 for anomalies, 1 for normal).
    """
    logging.info(f"--- Running Anomaly Detection for: {data_path} ---")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Contamination rate: {contamination}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_filename)
    scaler_path = os.path.join(output_dir, scaler_filename)

    # 1. Load Data
    df_full = load_data(data_path)
    # Preserve original index if needed (e.g., timestamp)
    original_index = df_full.index

    # 2. Select Features
    df_features = select_features(df_full, CORE_PIDS_FOR_ANOMALY, DERIVED_FEATURES_FOR_ANOMALY)
    actual_features_used = df_features.columns.tolist() # Store the actual features used

    # 3. Preprocess Data (Scaling, NaN handling)
    # Important: Force refit if scaler exists but is for different features?
    # For now, just remove if exists to simplify workflow for single dataset runs
    if os.path.exists(scaler_path):
         logging.info(f"Removing existing scaler to ensure refit for this dataset: {scaler_path}")
         try:
             os.remove(scaler_path)
         except OSError as e:
             logging.error(f"Error removing existing scaler: {e}")
             # Decide if we should proceed or exit
             # sys.exit(1) # Option: exit if scaler cannot be removed

    df_scaled, scaler = preprocess_data(df_features.copy(), scaler_path) # Use copy to avoid SettingWithCopyWarning

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


def analyze_anomalies(df_with_anomalies: pd.DataFrame, features_used: list):
    """
    Analyzes the data points flagged as anomalies and maps potential DTCs.
    Adds a 'potential_dtcs' column (list of strings) to the anomalous subset.

    Args:
        df_with_anomalies (pd.DataFrame): DataFrame including the 'anomaly' column.
        features_used (list): List of feature names used by the anomaly model.

    Returns:
        pd.DataFrame: A DataFrame containing only the anomalous rows,
                      with the added 'potential_dtcs' column.
    """
    if not isinstance(df_with_anomalies, pd.DataFrame):
        logging.error("analyze_anomalies expects a Pandas DataFrame.")
        return pd.DataFrame() # Return empty DataFrame on invalid input

    anomalous_df = df_with_anomalies[df_with_anomalies['anomaly'] == -1].copy()

    if anomalous_df.empty:
        logging.info("No anomalies detected for analysis.")
        if 'potential_dtcs' not in anomalous_df.columns:
             anomalous_df['potential_dtcs'] = None
             anomalous_df['potential_dtcs'] = anomalous_df['potential_dtcs'].astype('object')
        return anomalous_df


    logging.info(f"Number of anomalous data points: {anomalous_df.shape[0]}")

    # --- Initialize column for potential DTCs (once, if not already present from prior steps) ---
    if 'potential_dtcs' not in anomalous_df.columns:
        anomalous_df['potential_dtcs'] = pd.Series([[] for _ in range(len(anomalous_df))], index=anomalous_df.index).astype('object')
    else: # Ensure existing column can hold lists properly
        anomalous_df['potential_dtcs'] = anomalous_df['potential_dtcs'].apply(lambda x: x if isinstance(x, list) else [])


    # Dictionary to collect all DTC updates. Key: index, Value: list of DTCs
    dtc_updates_map = {}

    # Helper function to add DTCs to the map for a given index
    def add_dtc_to_map(idx, dtc_to_add):
        # Get current list for this index from the map, or from the DataFrame if not yet in map
        current_dtcs_for_idx = dtc_updates_map.get(idx, anomalous_df.loc[idx, 'potential_dtcs'])
        if not isinstance(current_dtcs_for_idx, list): # Ensure it's a list
            current_dtcs_for_idx = []
        
        new_list = list(current_dtcs_for_idx) # Work with a copy
        if isinstance(dtc_to_add, list): # If adding multiple DTCs
            for d in dtc_to_add:
                if d not in new_list:
                    new_list.append(d)
        else: # If adding a single DTC
            if dtc_to_add not in new_list:
                new_list.append(dtc_to_add)
        
        if new_list != current_dtcs_for_idx: # Only update map if there was a change
             dtc_updates_map[idx] = new_list


    # --- Load DTC descriptions (once) ---
    dtc_data = load_dtc_data() # Use default path "dtc.json"

    # --- Start: Heuristic DTC Mapping Example (Voltage Issues) ---
    logging.info("\n\n--- Anomaly Pattern Analysis: Control Module Voltage Issues ---")
    voltage_col = 'CONTROL_MODULE_VOLTAGE' # Define column name
    time_col_present_for_voltage = 'TIME_SEC' in anomalous_df.columns # Check if time column is available
    min_time_for_voltage_check = 60.0 # Ignore first 60 seconds for voltage checks

    if dtc_data and voltage_col in anomalous_df.columns: # Check column exists
        # Define fixed thresholds
        low_voltage_threshold = 12.0
        high_voltage_threshold = 15.0
        sustained_low_voltage_duration = 3 # Require 3 consecutive seconds below threshold

        voltage_dtcs_low = ["P0560", "P0561", "P0562"] # System Voltage Malfunction/Unstable/Low
        voltage_dtcs_high = ["P0563"] # System Voltage High

        # --- Sustained Low Voltage Check ---
        # This requires identifying consecutive anomalies meeting the low voltage criteria
        logging.info(f"Identifying anomalies with SUSTAINED ({sustained_low_voltage_duration}s) {voltage_col} < {low_voltage_threshold:.1f} V (and TIME_SEC > {min_time_for_voltage_check}s if available)")

        # 1. Find all individual anomalies meeting the base condition
        voltage_base_condition = pd.Series(True, index=anomalous_df.index)
        if time_col_present_for_voltage:
            logging.info(f"  (Initial filter: TIME_SEC <= {min_time_for_voltage_check}s ignored)")
            voltage_base_condition &= (anomalous_df['TIME_SEC'] > min_time_for_voltage_check)
        else:
            logging.warning("  TIME_SEC column not found, cannot filter initial voltage anomalies.")

        individual_low_voltage_mask = voltage_base_condition & (anomalous_df[voltage_col] < low_voltage_threshold)
        individual_low_indices = anomalous_df[individual_low_voltage_mask].index
        logging.info(f"  Found {len(individual_low_indices)} individual anomalies initially meeting low voltage criteria.")

        # 2. Find consecutive sequences among these individual anomalies
        sustained_low_indices = pd.Index([]) # Initialize as empty Index
        if not individual_low_indices.empty and len(individual_low_indices) >= sustained_low_voltage_duration:
            try:
                # Calculate differences between consecutive indices; assumes index is sortable (numeric or datetime)
                # Using .to_series() to handle potential MultiIndex safely, ensure index is sorted first
                sorted_indices = individual_low_indices.sort_values()
                index_diff = sorted_indices.to_series().diff()

                # Heuristic to define a "break" - adjust if needed based on typical time gaps
                # If index is numeric (like default RangeIndex), diff > 1 is a break.
                # If index is datetime, a larger threshold (e.g., > pd.Timedelta('2s')) might be needed.
                # Assuming default-like index for now.
                if pd.api.types.is_numeric_dtype(sorted_indices.dtype):
                     break_threshold = 1 # For default integer index
                elif pd.api.types.is_datetime64_any_dtype(sorted_indices.dtype):
                     # Adjust this timedelta based on expected data frequency
                     break_threshold = pd.Timedelta('2s')
                else:
                     logging.warning("Sustained voltage check using index diff might be unreliable for non-numeric/non-datetime index.")
                     break_threshold = 1 # Default fallback

                # Mark group starts: first element or where diff > break_threshold
                group_starts = (index_diff > break_threshold) | (index_diff.isna())
                group_ids = group_starts.cumsum()

                # Count occurrences in each group
                group_counts = anomalous_df.loc[sorted_indices].index.to_series().groupby(group_ids).size()

                # Identify groups meeting the duration criteria (count >= duration)
                valid_group_ids = group_counts[group_counts >= sustained_low_voltage_duration].index

                # Filter the original indices to keep only those belonging to valid groups
                sustained_low_indices = sorted_indices[group_ids.isin(valid_group_ids)]
            except Exception as e:
                logging.error(f"Error during sustained low voltage check using index diff: {e}. Skipping this check.")
                sustained_low_indices = pd.Index([]) # Reset on error

        if sustained_low_indices.empty:
            logging.info(f"  No anomalies found part of sustained low voltage periods (>= {sustained_low_voltage_duration} occurrences based on index diff)." )
        else:
            logging.info(f"  Found {len(sustained_low_indices)} anomalies potentially part of sustained low voltage periods (>= {sustained_low_voltage_duration} occurrences based on index diff)." )

            # Filter the anomalous_df using the identified sustained indices
            sustained_low_voltage_df_subset = anomalous_df.loc[sustained_low_indices]

            # Iterate safely using iterrows over this subset
            # Build a dictionary of updates first to avoid repeated .loc assignments in loop
            for idx, _ in sustained_low_voltage_df_subset.iterrows():
                add_dtc_to_map(idx, voltage_dtcs_low) # Use helper

            # After the loop, apply the updates using pd.Series and df.update()
            # This will be done once at the end of the function
            # if dtcs_to_update:
            #     update_series = pd.Series(dtcs_to_update, name=\'potential_dtcs\')
            #     anomalous_df.update(update_series)
            #     logging.info(f\"  Applied low voltage DTCs to {len(dtcs_to_update)} indices.\")


            # --- High Voltage Check ---
            logging.info(f"\\nIdentifying anomalies with {voltage_col} > {high_voltage_threshold:.1f} V (and TIME_SEC > {min_time_for_voltage_check}s if available)")
            high_voltage_condition = voltage_base_condition & (anomalous_df[voltage_col] > high_voltage_threshold)
            high_voltage_indices = anomalous_df[high_voltage_condition].index

            if not high_voltage_indices.empty:
                logging.info(f"Found {len(high_voltage_indices)} anomalies with high voltage (after initial period).")
                for idx in high_voltage_indices:
                    add_dtc_to_map(idx, voltage_dtcs_high) # Use helper
            else:
                logging.info("No anomalies found above the high voltage threshold (after initial period).")

            # Print potentially relevant DTC descriptions
            all_voltage_dtcs = sorted(list(set(voltage_dtcs_low + voltage_dtcs_high)))
            logging.info("\nPotentially Relevant DTCs (Voltage Issues):")
            for code in all_voltage_dtcs:
                desc = get_dtc_description(code, dtc_data)
                logging.info(f"  - {code}: {desc}")

    else:
        if not dtc_data:
            logging.warning("Skipping voltage analysis because DTC data failed to load.")
        else:
            # Log the specific column name that is missing
            logging.warning(f"Skipping voltage analysis because {voltage_col} column is missing.")
    # --- End: Voltage Heuristic ---

    # --- Start: Low Coolant Temp Heuristic ---
    logging.info("\n\n--- Anomaly Pattern Analysis: Low Coolant Temperature After Warmup ---")
    coolant_col = 'COOLANT_TEMPERATURE'
    time_col = 'TIME_SEC' # Assuming TIME_SEC represents runtime or similar progression
    ambient_temp_col = 'AMBIENT_AIR_TEMPERATURE' # For conditional warmup time
    coolant_diff_col = 'COOLANT_TEMPERATURE_diff_1' # Assuming derived feature exists
    load_diff_col = 'CALCULATED_ENGINE_LOAD_VALUE_diff_1' # Assuming derived feature exists


    if dtc_data and coolant_col in anomalous_df.columns and time_col in anomalous_df.columns:
        # Define fixed thresholds
        low_temp_threshold = 75.0 # Degrees C
        # Define base runtime and adjustments based on ambient temp
        base_min_runtime_sec = 300 # 5 minutes (default)
        ambient_temp_present = ambient_temp_col in anomalous_df.columns

        coolant_dtcs_thermostat = ["P0128"] # Thermostat Rationality
        coolant_dtcs_sensor = ["P0117", "P0118", "P0119"] # Sensor Low/High/Intermittent

        # --- Determine Runtime Threshold ---
        # Create a Series for the runtime threshold, matching the anomalous_df index
        min_runtime_thresholds = pd.Series(base_min_runtime_sec, index=anomalous_df.index)
        if ambient_temp_present:
            logging.info(f"Adjusting minimum runtime based on {ambient_temp_col}:")
            cold_condition = anomalous_df[ambient_temp_col] < 5.0
            cool_condition = (anomalous_df[ambient_temp_col] >= 5.0) & (anomalous_df[ambient_temp_col] < 15.0)

            min_runtime_thresholds[cold_condition] = 600.0 # 10 mins if < 5C
            min_runtime_thresholds[cool_condition] = 450.0 # 7.5 mins if 5-15C

            n_cold = cold_condition.sum()
            n_cool = cool_condition.sum()
            n_warm = (~cold_condition & ~cool_condition).sum()
            logging.info(f"  Runtime > 600s required for {n_cold} anomalies (Ambient < 5C)")
            logging.info(f"  Runtime > 450s required for {n_cool} anomalies (5C <= Ambient < 15C)")
            logging.info(f"  Runtime > 300s required for {n_warm} anomalies (Ambient >= 15C)")
        else:
            logging.warning(f"{ambient_temp_col} not found. Using default minimum runtime of {base_min_runtime_sec}s for all anomalies.")

        logging.info(f"Identifying anomalies with {coolant_col} < {low_temp_threshold:.1f} C AND {time_col} > adjusted minimum runtime")

        # Apply conditions using the calculated runtime thresholds
        stuck_open_condition = (
            (anomalous_df[coolant_col] < low_temp_threshold) &
            (anomalous_df[time_col] > min_runtime_thresholds)
        )
        stuck_open_indices = anomalous_df[stuck_open_condition].index


        if not stuck_open_indices.empty:
            logging.info(f"Found {len(stuck_open_indices)} anomalies with low coolant temperature after adjusted runtime (Potential P0128).")
            # (Optional: Print sample rows)

            # Append P0128 (Thermostat) for these anomalies
            for idx in stuck_open_indices:
                add_dtc_to_map(idx, "P0128") # Use helper

            # Further analysis: Check for sudden drops within this subset (Potential Sensor Issue)
            if coolant_diff_col in anomalous_df.columns:
                # Define a threshold for a significant drop (Using absolute diff for clarity now)
                # Need original non-scaled data for this ideally, or adjust scaled threshold carefully
                # Let's assume scaled diff is available and use a threshold based on that distribution
                # A large negative scaled value implies a significant drop relative to recent history
                significant_drop_threshold_scaled = -0.5 # Example: drop more than 0.5 std dev in one step

                # Condition 1: Temperature is dropping significantly (using scaled diff)
                # Apply condition only on the 'stuck_open_indices' subset for efficiency
                dropping_condition = anomalous_df.loc[stuck_open_indices, coolant_diff_col] < significant_drop_threshold_scaled

                # Condition 2 (Optional): Engine load is NOT decreasing significantly (more suspicious)
                load_not_decreasing_condition = True
                if load_diff_col in anomalous_df.columns:
                    # Load decrease less than 0.01 std dev (i.e., stable or increasing)
                    load_not_decreasing_condition = anomalous_df.loc[stuck_open_indices, load_diff_col] >= -0.01
                else:
                    logging.warning(f"\n  {load_diff_col} not available for load condition.")

                # Combine conditions for potential sensor issue
                # Get indices from the boolean series derived from the subset
                valid_indices = stuck_open_indices[dropping_condition & load_not_decreasing_condition]


                if not valid_indices.empty:
                    logging.info(f"\n  Out of these, {len(valid_indices)} also showed a significant coolant temperature drop (scaled_diff < {significant_drop_threshold_scaled}) while load was stable/increasing (Potential Sensor DTCs).")
                    # (Optional: Print sample rows)

                    # Append Sensor-related DTCs for these specific cases
                    for idx in valid_indices:
                        add_dtc_to_map(idx, coolant_dtcs_sensor) # Use helper
                else:
                    logging.info(f"\n  None of the low coolant temperature anomalies showed a significant drop (scaled_diff < {significant_drop_threshold_scaled}) while engine load was stable or increasing.")
            else:
                logging.warning(f"\n  {coolant_diff_col} column not available for drop analysis.")

            # Print potentially relevant DTC descriptions
            all_coolant_related_dtcs = sorted(list(set(coolant_dtcs_thermostat + coolant_dtcs_sensor)))
            logging.info("\nPotentially Relevant DTCs (Coolant Issues):")
            for dtc in all_coolant_related_dtcs:
                 desc = get_dtc_description(dtc, dtc_data)
                 logging.info(f"  - {dtc}: {desc}")

        else:
            logging.info("No low coolant temperature anomalies found after adjusted runtime.")
    else:
        if not dtc_data:
            logging.warning("Skipping low coolant temp analysis because DTC data failed to load.")
        elif coolant_col not in anomalous_df.columns:
             logging.warning(f"Skipping low coolant temp analysis because {coolant_col} column is missing.")
        elif time_col not in anomalous_df.columns:
             # Add check for time column missing
             logging.warning(f"Skipping low coolant temp analysis because {time_col} column is missing.")
    # --- End: Low Coolant Temp Heuristic ---


    # --- Start: High RPM / Low Speed Heuristic ---
    logging.info("\n\n--- Anomaly Pattern Analysis: High Engine RPM at Low Speed ---")
    rpm_col = 'ENGINE_RPM'
    speed_col = 'VEHICLE_SPEED'

    if dtc_data and rpm_col in anomalous_df.columns and speed_col in anomalous_df.columns:
        # Define fixed thresholds based on typical vehicle behavior
        high_rpm_threshold = 1500.0 # RPM - well above normal idle
        low_speed_threshold = 10.0 # km/h - essentially stationary or creeping
        idle_dtcs = ["P0506", "P0507"] # RPM lower/higher than expected

        logging.info(f"Identifying anomalies with {rpm_col} > {high_rpm_threshold:.0f} RPM AND {speed_col} < {low_speed_threshold:.0f} km/h")

        # Find indices meeting the condition
        high_rpm_low_speed_indices = anomalous_df[
            (anomalous_df[rpm_col] > high_rpm_threshold) &
            (anomalous_df[speed_col] < low_speed_threshold)
        ].index

        if not high_rpm_low_speed_indices.empty:
            logging.info(f"Found {len(high_rpm_low_speed_indices)} anomalies with high RPM at low speed.")
            # (Optional: Print sample rows)

            # Append DTCs
            for idx in high_rpm_low_speed_indices:
                add_dtc_to_map(idx, idle_dtcs) # Use helper

            # Print potentially relevant DTC descriptions
            logging.info("\nPotentially Relevant DTCs (Idle Control):")
            for code in idle_dtcs:
                desc = get_dtc_description(code, dtc_data)
                logging.info(f"  - {code}: {desc}")
        else:
            logging.info("No anomalies found matching the high RPM / low speed condition.")
    else:
        if not dtc_data:
            logging.warning("Skipping high RPM/low speed analysis because DTC data failed to load.")
        elif rpm_col not in anomalous_df.columns:
            logging.warning(f"Skipping high RPM/low speed analysis because {rpm_col} column is missing.")
        elif speed_col not in anomalous_df.columns:
            # Add check for speed column missing
            logging.warning(f"Skipping high RPM/low speed analysis because {speed_col} column is missing.")
    # --- End: High RPM / Low Speed Heuristic ---


    # --- Start: Low MAF / High Load Heuristic ---
    logging.info("\n\n--- Anomaly Pattern Analysis: Low MAF at High Engine Load ---")
    maf_col = 'MASS_AIR_FLOW'
    load_col = 'CALCULATED_ENGINE_LOAD_VALUE'

    # Check column existence in anomalous_df
    if dtc_data and maf_col in anomalous_df.columns and load_col in anomalous_df.columns:
        # Define fixed thresholds
        high_load_threshold = 80.0  # percent
        low_maf_threshold = 20.0   # g/s

        potential_maf_issues = anomalous_df[
            (anomalous_df[load_col] > high_load_threshold) &
            (anomalous_df[maf_col] < low_maf_threshold)
        ]

        count = len(potential_maf_issues)
        logging.info(f"Found {count} anomalies with Load > {high_load_threshold}% and MAF < {low_maf_threshold} g/s.")
        if count > 0:
            sample_issues = potential_maf_issues.head(5)
            logging.info(f"Sample 'Low MAF / High Load' anomalies:\n{sample_issues[[load_col, maf_col]].to_string()}")

            # Correct approach: Iterate or use apply
            maf_dtcs = ["P0101", "P0102"]
            for idx in potential_maf_issues.index:
                 add_dtc_to_map(idx, maf_dtcs) # Use helper


            logging.info("\nPotentially Relevant DTCs (MAF Sensor Issues):")
            for dtc in sorted(list(set(maf_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 logging.info(f"  - {dtc}: {desc}")
        else:
            logging.info("No anomalies found matching the Low MAF / High Load criteria.")
    else:
        # Refine missing column logging
        missing_cols = []
        if not dtc_data: missing_cols.append("DTC data")
        if maf_col not in anomalous_df.columns: missing_cols.append(maf_col)
        if load_col not in anomalous_df.columns: missing_cols.append(load_col)
        if missing_cols:
             logging.warning(f"Skipping low MAF / high load analysis due to missing: {', '.join(missing_cols)}.")

    # --- End: Low MAF / High Load Heuristic ---


    # --- Start: Low MAP / High Load Heuristic ---
    logging.info("\n\n--- Anomaly Pattern Analysis: Low MAP Sensor Reading at High Engine Load ---")
    map_col = 'INTAKE_MANIFOLD_ABSOLUTE_PRESSURE'
    # load_col already defined

    if dtc_data and map_col in anomalous_df.columns and load_col in anomalous_df.columns:
        # Define fixed thresholds
        high_load_threshold = 80.0  # percent
        low_map_threshold = 110.0 # kPa (slightly above atmospheric)

        logging.info(f"Identifying anomalies with {map_col} < {low_map_threshold:.1f} kPa AND {load_col} > {high_load_threshold:.1f}%")

        low_map_high_load_indices = anomalous_df[
            (anomalous_df[map_col] < low_map_threshold) &
            (anomalous_df[load_col] > high_load_threshold)
        ].index

        if not low_map_high_load_indices.empty:
            logging.info(f"Found {len(low_map_high_load_indices)} anomalies with low MAP at high engine load.")
            logging.info("Sample Anomalies (First 5):")
            try:
                pd.set_option('display.max_columns', 50)
                pd.set_option('display.width', 1000)
                # Include TIME_SEC if available
                display_cols = ['TIME_SEC'] if 'TIME_SEC' in anomalous_df.columns else []
                display_cols.extend([map_col, load_col, 'ENGINE_RPM', 'potential_dtcs'])
                # Filter display_cols to only those present in anomalous_df
                display_cols = [col for col in display_cols if col in anomalous_df.columns]
                # Log the head(), converting to string first
                logging.info("\n" + anomalous_df.loc[low_map_high_load_indices, display_cols].head().to_string())

                pd.reset_option('display.max_columns')
                pd.reset_option('display.width')
            except Exception as e:
                logging.error(f"Could not display sample Low MAP/High Load anomalies: {e}")

            map_dtcs = ["P0106", "P0107", "P0299"] # MAP/Baro Circuit Range/Perf, MAP Circuit Low, Turbo Underboost
            for idx in low_map_high_load_indices:
                add_dtc_to_map(idx, map_dtcs) # Use helper

            logging.info("\nPotentially Relevant DTCs (MAP Sensor/Boost Issues):")
            for dtc in sorted(list(set(map_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 logging.info(f"  - {dtc}: {desc}")
        else:
            logging.info("No anomalies found matching the low MAP / high load condition.")
    else:
        # Refine missing column logging
        missing_cols = []
        if not dtc_data: missing_cols.append("DTC data")
        if map_col not in anomalous_df.columns: missing_cols.append(map_col)
        if load_col not in anomalous_df.columns: missing_cols.append(load_col)
        if missing_cols:
             logging.warning(f"Skipping low MAP / high load analysis due to missing: {', '.join(missing_cols)}.")

    # --- End: Low MAP / High Load Heuristic ---


    # --- Start: High Throttle / Low Load Heuristic ---
    logging.info("\n\n--- Anomaly Pattern Analysis: High Throttle Position at Low Engine Load ---")
    throttle_col = 'THROTTLE_POSITION'
    # load_col already defined

    # Check if columns exist using the original anomalous_df
    if dtc_data and throttle_col in anomalous_df.columns and load_col in anomalous_df.columns:
        # Define fixed thresholds
        high_throttle_threshold = 60.0  # percent
        low_load_threshold = 30.0     # percent

        logging.info(f"Identifying anomalies with {throttle_col} > {high_throttle_threshold:.1f}% AND {load_col} < {low_load_threshold:.1f}%")

        high_throttle_low_load_indices = anomalous_df[
            (anomalous_df[throttle_col] > high_throttle_threshold) &
            (anomalous_df[load_col] < low_load_threshold)
        ].index

        if not high_throttle_low_load_indices.empty:
            logging.info(f"Found {len(high_throttle_low_load_indices)} anomalies with high throttle at low engine load.")
            logging.info("Sample Anomalies (First 5):")
            try:
                pd.set_option('display.max_columns', 50)
                pd.set_option('display.width', 1000)
                # Include TIME_SEC if available
                display_cols = ['TIME_SEC'] if 'TIME_SEC' in anomalous_df.columns else []
                display_cols.extend([throttle_col, load_col, 'ENGINE_RPM', 'potential_dtcs'])
                # Filter display_cols to only those present in anomalous_df
                display_cols = [col for col in display_cols if col in anomalous_df.columns]
                # Log the head(), converting to string first
                logging.info("\n" + anomalous_df.loc[high_throttle_low_load_indices, display_cols].head().to_string())
                pd.reset_option('display.max_columns')
                pd.reset_option('display.width')
            except Exception as e:
                logging.error(f"Could not display sample High Throttle/Low Load anomalies: {e}")

            tps_dtcs = ["P0121", "P0122", "P2135"] # TPS Range/Performance, TPS Circuit Low, TPS Correlation (A/B)
            for idx in high_throttle_low_load_indices:
                add_dtc_to_map(idx, tps_dtcs) # Use helper

            logging.info("\nPotentially Relevant DTCs (Throttle Position Sensor Issues):")
            for dtc in sorted(list(set(tps_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 logging.info(f"  - {dtc}: {desc}")
        else:
            logging.info("No anomalies found matching the high throttle / low load condition.")
    else:
        # Refine missing column logging
        missing_cols = []
        if not dtc_data: missing_cols.append("DTC data")
        if throttle_col not in anomalous_df.columns: missing_cols.append(throttle_col)
        if load_col not in anomalous_df.columns: missing_cols.append(load_col)
        if missing_cols:
             logging.warning(f"Skipping high throttle / low load analysis due to missing: {', '.join(missing_cols)}.")

    # --- End: High Throttle / Low Load Heuristic ---

    # --- New Heuristic ---

    # 7. EGR Error (Potential P0401, P0402, P0404, P0405, P0406 - EGR Flow/Range/Sensor Issues)
    egr_error_col = 'EGR_ERROR'
    rpm_col = 'ENGINE_RPM' # Already defined, but good for clarity
    load_col = 'CALCULATED_ENGINE_LOAD_VALUE' # Already defined

    if dtc_data and egr_error_col in anomalous_df.columns and rpm_col in anomalous_df.columns and load_col in anomalous_df.columns:
        egr_dtcs_flow_range = ["P0401", "P0402", "P0404"] # Insufficient, Excessive, Range/Performance
        egr_dtcs_sensor_stuck = ["P0405", "P0406"]       # Sensor A Circuit, Sensor A Range/Performance (often if stuck)

        # Conditions where EGR is typically closed or its error is more indicative of a problem
        # Low RPM (idle), very high RPM (WOT), or high load often mean EGR should be minimal or closed.
        # Cold engine also a factor, but we don't have a simple "engine_warmed_up" flag here yet.
        # For simplicity, focus on RPM and Load for now.
        # Consider TIME_SEC > some_warmup_period if adding later.
        
        # Check for large errors regardless of operating condition
        large_error_condition = anomalous_df[egr_error_col].abs() > 25.0 # Error > 25%

        # Check for smaller but still significant errors when EGR should ideally be closed or very low
        # Example: At idle (RPM < 850) or high load (Load > 75%), EGR error should be minimal.
        idle_condition = anomalous_df[rpm_col] < 850
        high_load_condition = anomalous_df[load_col] > 75
        
        # EGR error when it should be closed/low (e.g. > 5-10% error)
        error_when_closed_condition = (idle_condition | high_load_condition) & (anomalous_df[egr_error_col].abs() > 10.0)

        # Combine conditions
        egr_problem_indices = anomalous_df[large_error_condition | error_when_closed_condition].index
        
        if not egr_problem_indices.empty:
            logging.info(f"\nFound {len(egr_problem_indices)} anomalies with potential EGR issues based on '{egr_error_col}'.")
            # Apply specific DTCs based on the type of condition met (simplified for now)
            for idx in egr_problem_indices:
                add_dtc_to_map(idx, egr_dtcs_flow_range) # Use helper
                add_dtc_to_map(idx, egr_dtcs_sensor_stuck) # Use helper
                
            logging.info("\nPotentially Relevant DTCs (EGR Issues):")
            all_egr_dtcs = sorted(list(set(egr_dtcs_flow_range + egr_dtcs_sensor_stuck)))
            for dtc_code in all_egr_dtcs:
                desc = get_dtc_description(dtc_code, dtc_data)
                logging.info(f"  - {dtc_code}: {desc}")
        else:
            logging.info("\nNo anomalies found matching EGR error conditions.")
            
    else:
        missing_info = []
        if not dtc_data: missing_info.append("DTC data")
        if egr_error_col not in anomalous_df.columns: missing_info.append(f"'{egr_error_col}' column")
        if rpm_col not in anomalous_df.columns: missing_info.append(f"'{rpm_col}' column")
        if load_col not in anomalous_df.columns: missing_info.append(f"'{load_col}' column")
        if missing_info:
            logging.warning(f"Skipping EGR error analysis due to missing: {', '.join(missing_info)}.")


    # --- Apply all collected DTC updates from the map to the DataFrame ---
    if dtc_updates_map:
        logging.info(f"\nApplying {len(dtc_updates_map)} collected DTC updates to the DataFrame.")
        # Create a Series from the map to use with df.update()
        # Ensure the Series has the same index type as anomalous_df for proper alignment
        update_series = pd.Series(dtc_updates_map, name='potential_dtcs', index=anomalous_df.index.intersection(dtc_updates_map.keys()))
        
        # Fill NaNs in the update_series with empty lists where necessary
        # This is to ensure that if an index was in dtc_updates_map but then an empty list was assigned,
        # it doesn't cause issues with .update() if it tries to update with NaN.
        # However, our helper `add_dtc_to_map` should always put a list.
        # More robustly, iterate and assign if `update` causes issues with list types.
        for_update_df = pd.DataFrame(update_series)
        anomalous_df.update(for_update_df)


    # --- Final Logging of DTC counts ---
    # Convert potential_dtcs set to a sorted list for consistent output (already lists now)
    # anomalous_df['potential_dtcs'] = potential_dtcs_list # This was from older logic

    logging.info(f"Analysis complete. {len(anomalous_df)} anomalies analyzed.")
    # Ensure 'potential_dtcs' is treated as string for value_counts if it contains lists
    logging.info(f"Value counts for potential DTCs (before filtering empty):\\n{anomalous_df['potential_dtcs'].astype(str).value_counts(dropna=False)}")

    # Filter out rows where no potential DTCs were assigned (optional, but keeps output clean)
    original_anomaly_count = len(anomalous_df)
    # Ensure we are checking for empty lists correctly
    anomalous_df_filtered = anomalous_df[anomalous_df['potential_dtcs'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    logging.info(f"Filtered out {original_anomaly_count - len(anomalous_df_filtered)} anomalies with no assigned heuristic DTCs.")
    logging.info(f"Final count of anomalies with potential DTCs: {len(anomalous_df_filtered)}")
    if not anomalous_df_filtered.empty:
        logging.info(f"Value counts for potential DTCs (final):\\n{anomalous_df_filtered['potential_dtcs'].astype(str).value_counts(dropna=False)}")
    else:
        logging.info("No anomalies with assigned heuristic DTCs after filtering.")

    return anomalous_df_filtered # Return the DataFrame with the new column, filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest for anomaly detection on OBD data and map potential DTCs.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True, # Make data-path mandatory
        help="Path to the input Parquet file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True, # Make output-dir mandatory
        help="Directory to save the model and scaler."
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Expected proportion of anomalies in the data (contamination factor). Default: 0.02"
    )
    parser.add_argument(
        "--output-csv", # New argument
        type=str,
        default=None,
        help="Optional path to save the DataFrame with detected anomalies and potential DTCs as a CSV file."
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
            contamination=args.contamination
        )

        # Get the features used from the DataFrame attribute
        actual_features_used = df_full_with_anomalies.attrs.get('features_used', [])
        if not actual_features_used:
             logging.warning("Could not retrieve list of features used from DataFrame attributes.")
             # Fallback (less reliable if columns were missing)
             actual_features_used = CORE_PIDS_FOR_ANOMALY + DERIVED_FEATURES_FOR_ANOMALY


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