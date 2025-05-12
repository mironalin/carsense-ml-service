import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import argparse # Import argparse

# Import the dtc lookup utility (we might use it later)
from app.preprocessing.dtc_lookup import load_dtc_data, get_dtc_description

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
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found at: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        raise


def select_features(df: pd.DataFrame, core_pids: list, derived_features: list) -> pd.DataFrame:
    """Selects relevant features for anomaly detection."""
    print("Selecting features...")
    available_core = [pid for pid in core_pids if pid in df.columns]
    available_derived = [feat for feat in derived_features if feat in df.columns]

    missing_core = set(core_pids) - set(available_core)
    if missing_core:
        print(f"Warning: Core PIDs not found in DataFrame: {missing_core}")

    missing_derived = set(derived_features) - set(available_derived)
    if missing_derived:
        print(f"Warning: Derived features not found in DataFrame: {missing_derived}")

    features_to_use = available_core + available_derived
    if not features_to_use:
        raise ValueError("No features selected for anomaly detection. Check PIDs and derived feature names.")

    print(f"Selected features ({len(features_to_use)}): {features_to_use}")
    return df[features_to_use]


def preprocess_data(df: pd.DataFrame, scaler_path: str = None) -> (pd.DataFrame, StandardScaler):
    """Handles NaNs and scales the data. Saves or loads the scaler."""
    print("Preprocessing data (handling NaNs, scaling)...")
    # Simple NaN handling: fill with the mean of the column
    # More sophisticated strategies might be needed (e.g., interpolation)
    for col in df.columns:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"  NaNs in '{col}' filled with mean ({mean_val:.2f})")

    if scaler_path and os.path.exists(scaler_path):
        print(f"Loading existing scaler from: {scaler_path}")
        scaler = load(scaler_path)
    else:
        print("Fitting new StandardScaler...")
        scaler = StandardScaler()
        scaler.fit(df)
        if scaler_path:
            print(f"Saving scaler to: {scaler_path}")
            dump(scaler, scaler_path)

    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    print("Data scaled.")
    return scaled_df, scaler


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
    print(f"--- Running Anomaly Detection for: {data_path} ---")
    print(f"Output directory: {output_dir}")
    print(f"Contamination rate: {contamination}")

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
    # Important: Always use a fresh scaler for a new dataset/output dir
    # Delete old scaler if it exists to force refit for this dataset
    if os.path.exists(scaler_path):
         print(f"Removing existing scaler to ensure refit for this dataset: {scaler_path}")
         os.remove(scaler_path)
    df_scaled, scaler = preprocess_data(df_features.copy(), scaler_path) # Use copy to avoid SettingWithCopyWarning

    # 4. Train Isolation Forest Model
    print("Training Isolation Forest model...")
    # Consider making n_estimators, max_samples, etc., configurable
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(df_scaled)
    print(f"Saving model to: {model_path}")
    dump(model, model_path)
    print("Model training complete.")

    # 5. Predict Anomalies
    print("Predicting anomalies...")
    # predict returns -1 for outliers and 1 for inliers.
    anomaly_predictions = model.predict(df_scaled)
    print(f"Prediction complete. Found {np.sum(anomaly_predictions == -1)} potential anomalies.")

    # Add predictions back to the original DataFrame
    # Ensure index alignment if preprocessing changed it (it shouldn't here)
    df_full['anomaly'] = anomaly_predictions
    # Restore index *before* potentially adding analysis columns
    if not df_full.index.equals(original_index):
         print("Restoring original index...")
         df_full.set_index(original_index, inplace=True)

    # Add the list of features used to the dataframe attributes for later use
    df_full.attrs['features_used'] = actual_features_used

    # Optional: Add anomaly scores as well
    # anomaly_scores = model.decision_function(df_scaled)
    # df_full['anomaly_score'] = anomaly_scores

    return df_full


def analyze_anomalies(df_with_anomalies: pd.DataFrame, features_used: list):
    """
    Analyzes the data points flagged as anomalies.

    Args:
        df_with_anomalies (pd.DataFrame): DataFrame including the 'anomaly' column.
        features_used (list): List of feature names used by the anomaly model.
    """
    print("\nAnalyzing detected anomalies...")
    anomalous_df = df_with_anomalies[df_with_anomalies['anomaly'] == -1].copy()

    if anomalous_df.empty:
        print("No anomalies detected for analysis.")
        return anomalous_df # Return the empty dataframe

    print(f"Number of anomalous data points: {anomalous_df.shape[0]}")

    # --- Initialize column for potential DTCs ---
    # Use lists to allow multiple potential DTCs per anomaly
    anomalous_df['potential_dtcs'] = [[] for _ in range(anomalous_df.shape[0])]


    # Ensure only the features used by the model (+ maybe timestamp) are described
    analysis_cols = [col for col in features_used if col in anomalous_df.columns]
    # Optionally add timestamp or other context columns if available and helpful
    if 'absolute_timestamp' in df_with_anomalies.columns:
        analysis_cols.append('absolute_timestamp')
    if 'TIME_SEC' in df_with_anomalies.columns:
         analysis_cols.append('TIME_SEC')

    print("\nDescriptive Statistics for Anomalous Data Points (Features Used by Model):")
    # Select only the relevant columns for describe()
    anomalous_subset = anomalous_df[analysis_cols]
    desc_stats = anomalous_subset.describe()
    print(desc_stats.to_string())

    # Optional: Print a few full anomalous rows for detailed inspection
    print("\n\nSample Anomalous Rows (Overall - First 5):")
    # Display more columns for context if needed
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    print(anomalous_df.head().to_string())
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

    # --- Load DTC descriptions (if available) ---
    dtc_data = load_dtc_data() # Use default path "dtc.json"

    # --- Start: Heuristic DTC Mapping Example (Low Voltage) ---
    print("\n\n--- Anomaly Pattern Analysis: Low Control Module Voltage ---")

    if dtc_data and 'CONTROL_MODULE_VOLTAGE' in desc_stats.columns:
        # Define a threshold, e.g., below the 25th percentile of anomalous voltage
        low_voltage_threshold = desc_stats.loc['25%', 'CONTROL_MODULE_VOLTAGE']
        print(f"Identifying anomalies with CONTROL_MODULE_VOLTAGE < {low_voltage_threshold:.2f} (25th percentile of anomalies)")

        low_voltage_indices = anomalous_df[
            anomalous_df['CONTROL_MODULE_VOLTAGE'] < low_voltage_threshold
        ].index

        # Append DTCs to the corresponding rows
        voltage_dtcs = ["P0560", "P0561", "P0562", "P0563"] # Example potential DTCs
        for idx in low_voltage_indices:
            # Ensure we don't add duplicates if other heuristics flag the same DTC
            for dtc in voltage_dtcs:
                if dtc not in anomalous_df.loc[idx, 'potential_dtcs']:
                    anomalous_df.loc[idx, 'potential_dtcs'].append(dtc)

        if not low_voltage_indices.empty:
            print(f"Found {len(low_voltage_indices)} anomalies with low voltage.")
            print("\nSample Low Voltage Anomalous Rows (First 5):")
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 1000)
            print(anomalous_df.loc[low_voltage_indices].head().to_string())
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')

            # Look up potentially relevant DTCs
            print("\nPotentially Relevant DTCs (Voltage Issues):")
            for code in voltage_dtcs:
                desc = get_dtc_description(code, dtc_data)
                if desc:
                    fault = desc.get('fault', 'N/A')
                    desc_text = desc.get('description', 'N/A')
                    print(f"  - {code}: {fault} | {desc_text}")
                else:
                    print(f"  - {code}: Not found in dtc.json")
        else:
            print("No anomalies found below the low voltage threshold.")
    else:
        if not dtc_data:
            print("Skipping low voltage analysis because DTC data failed to load.")
        else:
            print("Skipping low voltage analysis because CONTROL_MODULE_VOLTAGE stats are missing.")
    # --- End: Heuristic DTC Mapping Example ---

    # --- Start: Heuristic DTC Mapping Example (Low Coolant Temp) ---
    print("\n\n--- Anomaly Pattern Analysis: Low Coolant Temperature ---")

    if dtc_data and 'COOLANT_TEMPERATURE' in desc_stats.columns and 'TIME_SEC' in anomalous_df.columns:
        # Define thresholds
        low_temp_threshold = desc_stats.loc['25%', 'COOLANT_TEMPERATURE']
        min_runtime_sec = 120 # Only flag if engine has been running for > 2 minutes
        coolant_dtcs = ["P0128", "P0117", "P0118", "P0119"] # Example potential DTCs

        print(f"Identifying anomalies with COOLANT_TEMPERATURE < {low_temp_threshold:.2f} (25th percentile of anomalies) AND TIME_SEC > {min_runtime_sec}")

        # Apply both conditions
        low_temp_anomalies_indices = anomalous_df[
            (anomalous_df['COOLANT_TEMPERATURE'] < low_temp_threshold) &
            (anomalous_df['TIME_SEC'] > min_runtime_sec)
        ].index

        if not low_temp_anomalies_indices.empty:
            print(f"Found {len(low_temp_anomalies_indices)} anomalies with low coolant temperature after {min_runtime_sec}s runtime.")
            print("\nSample Low Coolant Temp Anomalous Rows (First 5 after runtime filter):")
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 1000)
            print(anomalous_df.loc[low_temp_anomalies_indices].head().to_string())
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')

            # Append P0128 (Thermostat) for all low-temp-after-runtime anomalies
            for idx in low_temp_anomalies_indices:
                if "P0128" not in anomalous_df.loc[idx, 'potential_dtcs']:
                    anomalous_df.loc[idx, 'potential_dtcs'].append("P0128")

            # Further analysis: Check for sudden drops in these low-temp anomalies
            if 'COOLANT_TEMPERATURE_diff_1' in anomalous_df.columns:
                # Define a threshold for a significant drop (e.g., more than 0.5 scaled units)
                significant_drop_threshold = -0.5 # Scaled value

                # Condition 1: Temperature is dropping significantly
                dropping_condition = anomalous_df.loc[low_temp_anomalies_indices, 'COOLANT_TEMPERATURE_diff_1'] < significant_drop_threshold

                # Condition 2 (Optional refinement): Engine load is NOT decreasing (more suspicious)
                load_not_decreasing_condition = True # Default to True if load diff is not available
                if 'CALCULATED_ENGINE_LOAD_VALUE_diff_1' in anomalous_df.columns:
                    # Check if load diff is greater than or equal to 0 (or a small negative tolerance)
                    load_not_decreasing_condition = anomalous_df.loc[low_temp_anomalies_indices, 'CALCULATED_ENGINE_LOAD_VALUE_diff_1'] >= -0.01 # Allow small negative tolerance
                else:
                    print("\n  CALCULATED_ENGINE_LOAD_VALUE_diff_1 not available for load condition.")

                # Combine conditions
                dropping_unexpectedly_indices = anomalous_df.loc[low_temp_anomalies_indices][
                    dropping_condition & load_not_decreasing_condition
                ].index

                if not dropping_unexpectedly_indices.empty:
                    print(f"\n  Out of the low temp anomalies, {len(dropping_unexpectedly_indices)} also showed a significant coolant temperature drop (diff < {significant_drop_threshold}) while engine load was NOT decreasing.")
                    print("  Sample of low & dropping (unexpectedly) coolant temp anomalies (First 5):")
                    print(anomalous_df.loc[dropping_unexpectedly_indices].head().to_string())

                    # Append Sensor-related DTCs (P0117/P0118/P0119) for these specific cases
                    sensor_dtcs = ["P0117", "P0118", "P0119"]
                    for idx in dropping_unexpectedly_indices:
                        for dtc in sensor_dtcs:
                             if dtc not in anomalous_df.loc[idx, 'potential_dtcs']:
                                 anomalous_df.loc[idx, 'potential_dtcs'].append(dtc)
                else:
                    print(f"\n  None of the low coolant temperature anomalies showed a significant drop (diff < {significant_drop_threshold}) while engine load was stable or increasing.")
            else:
                print("\n  COOLANT_TEMPERATURE_diff_1 column not available for drop analysis.")

            # Print DTC descriptions (moved outside the drop check, covers both P0128 and sensor DTCs)
            all_coolant_related_dtcs = set(coolant_dtcs) # Combine potential DTCs
            print("\nPotentially Relevant DTCs (Coolant Issues):")
            for dtc in sorted(list(all_coolant_related_dtcs)):
                 desc = get_dtc_description(dtc, dtc_data)
                 print(f"  - {dtc}: {desc}")

        else:
            print(f"No low coolant temperature anomalies found after {min_runtime_sec}s runtime.")
    else:
        # Add check for missing TIME_SEC as well
        if not dtc_data:
            print("Skipping low coolant temp analysis because DTC data failed to load.")
        elif 'COOLANT_TEMPERATURE' not in desc_stats.columns:
             print("Skipping low coolant temp analysis because COOLANT_TEMPERATURE stats are missing.")
        elif 'TIME_SEC' not in anomalous_df.columns:
             print("Skipping low coolant temp analysis because TIME_SEC column is missing.")
    # --- End: Heuristic DTC Mapping Example ---

    # --- Start: Heuristic DTC Mapping Example (High RPM / Low Speed) ---
    print("\n\n--- Anomaly Pattern Analysis: High Engine RPM at Low Speed ---")

    if dtc_data and 'ENGINE_RPM' in desc_stats.columns and 'VEHICLE_SPEED' in desc_stats.columns:
        # Define thresholds (e.g., using percentiles of the anomalous data)
        high_rpm_threshold = desc_stats.loc['75%', 'ENGINE_RPM']
        low_speed_threshold = desc_stats.loc['25%', 'VEHICLE_SPEED']

        print(f"Identifying anomalies with ENGINE_RPM > {high_rpm_threshold:.2f} (75th percentile) AND VEHICLE_SPEED < {low_speed_threshold:.2f} (25th percentile)")

        # Find indices meeting the condition
        high_rpm_low_speed_indices = anomalous_df[
            (anomalous_df['ENGINE_RPM'] > high_rpm_threshold) &
            (anomalous_df['VEHICLE_SPEED'] < low_speed_threshold)
        ].index

        if not high_rpm_low_speed_indices.empty:
            print(f"Found {len(high_rpm_low_speed_indices)} anomalies with high RPM at low speed.")
            print("Sample Anomalies (First 5):")
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 1000)
            print(anomalous_df.loc[high_rpm_low_speed_indices, ['TIME_SEC', 'ENGINE_RPM', 'VEHICLE_SPEED', 'potential_dtcs']].head().to_string())
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')

            # Assign potential DTCs
            rpm_dtcs = ["P0507", "P0506"] # P0507: Idle RPM High, P0506: Idle RPM Low (might add later based on low RPM analysis)
            for idx in high_rpm_low_speed_indices:
                for dtc in rpm_dtcs:
                    if dtc not in anomalous_df.loc[idx, 'potential_dtcs']:
                        anomalous_df.loc[idx, 'potential_dtcs'].append(dtc)

            # Print DTC descriptions
            print("\nPotentially Relevant DTCs (Idle Control):")
            for dtc in sorted(list(set(rpm_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 print(f"  - {dtc}: {desc}")

        else:
            print("No anomalies found matching the high RPM / low speed condition.")
    else:
        print("Skipping high RPM / low speed analysis due to missing columns (ENGINE_RPM or VEHICLE_SPEED).")
    # --- End: Heuristic DTC Mapping Example (High RPM / Low Speed) ---

    # --- Start: Heuristic DTC Mapping Example (Low MAF / High Load) ---
    print("\n\n--- Anomaly Pattern Analysis: Low MAF at High Engine Load ---")

    if dtc_data and 'MASS_AIR_FLOW' in desc_stats.columns and 'CALCULATED_ENGINE_LOAD_VALUE' in desc_stats.columns:
        low_maf_threshold = desc_stats.loc['25%', 'MASS_AIR_FLOW']
        high_load_threshold = desc_stats.loc['50%', 'CALCULATED_ENGINE_LOAD_VALUE'] # Using 50th percentile for load

        print(f"Identifying anomalies with MASS_AIR_FLOW < {low_maf_threshold:.2f} (25th percentile MAF) AND CALCULATED_ENGINE_LOAD_VALUE > {high_load_threshold:.2f} (50th percentile Load)")

        low_maf_high_load_indices = anomalous_df[
            (anomalous_df['MASS_AIR_FLOW'] < low_maf_threshold) &
            (anomalous_df['CALCULATED_ENGINE_LOAD_VALUE'] > high_load_threshold)
        ].index

        if not low_maf_high_load_indices.empty:
            print(f"Found {len(low_maf_high_load_indices)} anomalies with low MAF at high engine load.")
            print("Sample Anomalies (First 5):")
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 1000)
            print(anomalous_df.loc[low_maf_high_load_indices, ['TIME_SEC', 'MASS_AIR_FLOW', 'CALCULATED_ENGINE_LOAD_VALUE', 'ENGINE_RPM', 'potential_dtcs']].head().to_string())
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')

            maf_dtcs = ["P0101", "P0102"] # MAF Range/Performance, MAF Circuit Low
            for idx in low_maf_high_load_indices:
                for dtc in maf_dtcs:
                    if dtc not in anomalous_df.loc[idx, 'potential_dtcs']:
                        anomalous_df.loc[idx, 'potential_dtcs'].append(dtc)

            print("\nPotentially Relevant DTCs (MAF Sensor Issues):")
            for dtc in sorted(list(set(maf_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 print(f"  - {dtc}: {desc}")
        else:
            print("No anomalies found matching the low MAF / high load condition.")
    else:
        print("Skipping low MAF / high load analysis due to missing columns (MASS_AIR_FLOW or CALCULATED_ENGINE_LOAD_VALUE).")
    # --- End: Heuristic DTC Mapping Example (Low MAF / High Load) ---

    # --- Start: Heuristic DTC Mapping Example (Low MAP / High Load) ---
    print("\n\n--- Anomaly Pattern Analysis: Low MAP Sensor Reading at High Engine Load ---")

    if dtc_data and 'INTAKE_MANIFOLD_ABSOLUTE_PRESSURE' in desc_stats.columns and 'CALCULATED_ENGINE_LOAD_VALUE' in desc_stats.columns:
        # Thresholds based on anomalous data percentiles
        low_map_threshold = desc_stats.loc['25%', 'INTAKE_MANIFOLD_ABSOLUTE_PRESSURE']
        high_load_threshold = desc_stats.loc['75%', 'CALCULATED_ENGINE_LOAD_VALUE'] # Using 75th percentile for higher confidence in load

        print(f"Identifying anomalies with INTAKE_MANIFOLD_ABSOLUTE_PRESSURE < {low_map_threshold:.2f} (25th percentile MAP) AND CALCULATED_ENGINE_LOAD_VALUE > {high_load_threshold:.2f} (75th percentile Load)")

        low_map_high_load_indices = anomalous_df[
            (anomalous_df['INTAKE_MANIFOLD_ABSOLUTE_PRESSURE'] < low_map_threshold) &
            (anomalous_df['CALCULATED_ENGINE_LOAD_VALUE'] > high_load_threshold)
        ].index

        if not low_map_high_load_indices.empty:
            print(f"Found {len(low_map_high_load_indices)} anomalies with low MAP at high engine load.")
            print("Sample Anomalies (First 5):")
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 1000)
            print(anomalous_df.loc[low_map_high_load_indices, ['TIME_SEC', 'INTAKE_MANIFOLD_ABSOLUTE_PRESSURE', 'CALCULATED_ENGINE_LOAD_VALUE', 'ENGINE_RPM', 'potential_dtcs']].head().to_string())
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')

            map_dtcs = ["P0106", "P0107"] # MAP/Baro Circuit Range/Perf, MAP Circuit Low
            for idx in low_map_high_load_indices:
                for dtc in map_dtcs:
                    if dtc not in anomalous_df.loc[idx, 'potential_dtcs']:
                        anomalous_df.loc[idx, 'potential_dtcs'].append(dtc)

            print("\nPotentially Relevant DTCs (MAP Sensor Issues):")
            for dtc in sorted(list(set(map_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 print(f"  - {dtc}: {desc}")
        else:
            print("No anomalies found matching the low MAP / high load condition.")
    else:
        print("Skipping low MAP / high load analysis due to missing columns (INTAKE_MANIFOLD_ABSOLUTE_PRESSURE or CALCULATED_ENGINE_LOAD_VALUE).")
    # --- End: Heuristic DTC Mapping Example (Low MAP / High Load) ---

    # --- Start: Heuristic DTC Mapping Example (High Throttle / Low Load) ---
    print("\n\n--- Anomaly Pattern Analysis: High Throttle Position at Low Engine Load ---")

    if dtc_data and 'THROTTLE_POSITION' in desc_stats.columns and 'CALCULATED_ENGINE_LOAD_VALUE' in desc_stats.columns:
        # Thresholds based on anomalous data percentiles
        high_throttle_threshold = desc_stats.loc['75%', 'THROTTLE_POSITION']
        low_load_threshold = desc_stats.loc['25%', 'CALCULATED_ENGINE_LOAD_VALUE']

        print(f"Identifying anomalies with THROTTLE_POSITION > {high_throttle_threshold:.2f} (75th percentile Throttle) AND CALCULATED_ENGINE_LOAD_VALUE < {low_load_threshold:.2f} (25th percentile Load)")

        high_throttle_low_load_indices = anomalous_df[
            (anomalous_df['THROTTLE_POSITION'] > high_throttle_threshold) &
            (anomalous_df['CALCULATED_ENGINE_LOAD_VALUE'] < low_load_threshold)
        ].index

        if not high_throttle_low_load_indices.empty:
            print(f"Found {len(high_throttle_low_load_indices)} anomalies with high throttle at low engine load.")
            print("Sample Anomalies (First 5):")
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 1000)
            print(anomalous_df.loc[high_throttle_low_load_indices, ['TIME_SEC', 'THROTTLE_POSITION', 'CALCULATED_ENGINE_LOAD_VALUE', 'ENGINE_RPM', 'potential_dtcs']].head().to_string())
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')

            tps_dtcs = ["P0121", "P0122"] # TPS Range/Performance, TPS Circuit Low
            for idx in high_throttle_low_load_indices:
                for dtc in tps_dtcs:
                    if dtc not in anomalous_df.loc[idx, 'potential_dtcs']:
                        anomalous_df.loc[idx, 'potential_dtcs'].append(dtc)

            print("\nPotentially Relevant DTCs (Throttle Position Sensor Issues):")
            for dtc in sorted(list(set(tps_dtcs))):
                 desc = get_dtc_description(dtc, dtc_data)
                 print(f"  - {dtc}: {desc}")
        else:
            print("No anomalies found matching the high throttle / low load condition.")
    else:
        print("Skipping high throttle / low load analysis due to missing columns (THROTTLE_POSITION or CALCULATED_ENGINE_LOAD_VALUE).")
    # --- End: Heuristic DTC Mapping Example (High Throttle / Low Load) ---

    # --- Display anomalies with mapped DTCs ---
    anomalies_with_dtcs = anomalous_df[anomalous_df['potential_dtcs'].apply(lambda x: len(x) > 0)]
    if not anomalies_with_dtcs.empty:
        print("\n\n--- Anomalies with Potential DTCs Mapped ---")
        print(f"Found {anomalies_with_dtcs.shape[0]} anomalies associated with potential DTCs based on heuristics.")
        print("Sample (First 10 rows with mapped DTCs):")
        print(anomalies_with_dtcs[['TIME_SEC', 'CONTROL_MODULE_VOLTAGE', 'COOLANT_TEMPERATURE', 'potential_dtcs']].head(10).to_string())
    else:
        print("\n\nNo anomalies were associated with potential DTCs based on the current heuristics.")

    return anomalous_df # Return the DataFrame with the new column


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest for anomaly detection on OBD data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/model_input/romanian_driving_ds_final.parquet",
        help="Path to the input Parquet file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/anomaly/romanian",
        help="Directory to save the model and scaler."
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Expected proportion of anomalies in the data (contamination factor)."
    )

    args = parser.parse_args()

    print(f"Running Anomaly Detection script with args: {args}")

    # Check if the data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at '{args.data_path}'.")
        # Consider adding sys.exit(1) here
    else:
        try:
            # Run the main process using arguments
            df_full_with_anomalies = train_and_predict_anomalies(
                data_path=args.data_path,
                output_dir=args.output_dir,
                contamination=args.contamination
            )

            # Get the features used from the DataFrame attribute
            actual_features_used = df_full_with_anomalies.attrs.get('features_used', [])
            if not actual_features_used:
                 print("Warning: Could not retrieve list of features used from DataFrame attributes.")
                 # Fallback (less reliable if columns were missing)
                 actual_features_used = CORE_PIDS_FOR_ANOMALY + DERIVED_FEATURES_FOR_ANOMALY


            # Display info about anomalies found
            print("\nAnomaly Detection Summary:")
            print(f"Processed data shape: {df_full_with_anomalies.shape}")
            anomaly_count = df_full_with_anomalies[df_full_with_anomalies['anomaly'] == -1].shape[0]
            print(f"Total potential anomalies detected: {anomaly_count}")
            if anomaly_count > 0:
                print("\nSample of detected anomalies (first 5):")
                print(df_full_with_anomalies[df_full_with_anomalies['anomaly'] == -1].head().to_string())

            # Analyze the anomalies
            analyzed_anomalies_df = analyze_anomalies(df_full_with_anomalies, actual_features_used)

        except FileNotFoundError as fnf_error:
            print(f"File Error: {fnf_error}")
        except ValueError as val_error:
            print(f"Value Error: {val_error}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Consider more detailed error logging or re-raising