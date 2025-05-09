import pandas as pd
from typing import Optional, Dict
import os

# Attempt to import from the preprocessing module. This might need adjustment based on PYTHONPATH.
try:
    from app.preprocessing.feature_engineering import PID_COLUMN_MAPPING, RELEVANT_PIDS, PID_COLUMN_MAPPING_VOLVO_V40
except ImportError:
    # Fallback for direct execution or if path issues occur initially.
    # This is not ideal for a production system but helps in interactive development.
    print("Warning: Could not import from app.preprocessing.feature_engineering. Using local placeholders if defined, or will fail if not.")
    # Define local placeholders if you want to run this file standalone for testing without full app structure
    # RELEVANT_PIDS = [...]
    # PID_COLUMN_MAPPING = { ... }
    # For now, we'll let it potentially fail if the import doesn't work, to highlight the dependency.
    PID_COLUMN_MAPPING = {} # Define as empty if not imported
    RELEVANT_PIDS = []      # Define as empty if not imported
    PID_COLUMN_MAPPING_VOLVO_V40 = {} # For Volvo dataset

# Fallback definitions (if feature_engineering not available)
if 'PID_COLUMN_MAPPING' not in globals():
    PID_COLUMN_MAPPING = {}
if 'RELEVANT_PIDS' not in globals():
    RELEVANT_PIDS = []

def load_romanian_dataset_csv(file_path: str, pid_mapping: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file from the Romanian driving dataset, selects relevant PID columns based on the mapping,
    renames them to standardized PID names, and ensures all RELEVANT_PIDS are present.

    Args:
        file_path: Path to the CSV file.
        pid_mapping: A dictionary mapping standardized PID names (keys) to CSV column names (values).
                        If None, defaults to PID_COLUMN_MAPPING from feature_engineering.

    Returns:
        A pandas DataFrame with standardized PID columns, or None if loading fails.
    """
    current_pid_mapping = pid_mapping if pid_mapping is not None else PID_COLUMN_MAPPING

    try:
        df_raw = pd.read_csv(file_path)
        print(f"Successfully loaded raw CSV: {file_path}")

        # Normalize column headers: strip leading/trailing whitespace
        df_raw.columns = df_raw.columns.str.strip()
        print(f"Normalized columns in raw CSV: {df_raw.columns.tolist()}\n")

        # --- START: Time Step Analysis (moved here) ---
        time_col_original_name = 'Time (sec)' # Standard name from dataset
        if time_col_original_name in df_raw.columns:
            time_diffs = df_raw[time_col_original_name].diff()
            print(f"\n--- Original Time Step Analysis for: {os.path.basename(file_path)} ---")
            print(f"  Min time diff: {time_diffs.min():.4f} sec")
            print(f"  Max time diff: {time_diffs.max():.4f} sec")
            print(f"  Mean time diff: {time_diffs.mean():.4f} sec")
            print(f"  Median time diff: {time_diffs.median():.4f} sec")
            print(f"  Std dev of time diff: {time_diffs.std():.4f} sec")
            unique_diffs = time_diffs.dropna()
            print(f"  Number of unique time diffs (excluding first NaN): {unique_diffs.nunique()}")
            if not unique_diffs.empty:
                print("  Top 5 most frequent time diffs (excluding NaN):")
                print(unique_diffs.value_counts().nlargest(5).to_string().replace('\n', '\n  ')) # Indent print
            else:
                print("  No time differences to report (column might be empty or have one value).")
            print("--- END: Original Time Step Analysis ---")

            # --- Resampling to 1S frequency ---
            print(f"\nResampling data to 1-second frequency for {os.path.basename(file_path)}...")
            # Convert 'Time (sec)' to TimedeltaIndex, assuming it starts from 0
            df_raw[time_col_original_name] = pd.to_timedelta(df_raw[time_col_original_name], unit='s')
            df_raw = df_raw.set_index(time_col_original_name)

            # Resample to 1 second frequency and forward fill
            # This handles non-unique index by taking the mean of duplicates before resampling, then ffilling
            # It's possible some raw data might have multiple readings for the exact same microsecond timestamp after conversion.
            # Group by index and take mean if duplicates exist, then resample.
            if df_raw.index.has_duplicates:
                print(f"Warning: Duplicate timestamps found in {os.path.basename(file_path)}. Averaging values at duplicate timestamps before resampling.")
                df_raw = df_raw.groupby(df_raw.index).mean()

            df_resampled = df_raw.resample('1s').ffill()

            # Reset index to get 'Time (sec)' back as a column, now regularized
            df_resampled = df_resampled.reset_index()
            # Convert the TimedeltaIndex back to seconds (float)
            df_resampled[time_col_original_name] = df_resampled[time_col_original_name].dt.total_seconds()
            print(f"Resampling complete. New shape: {df_resampled.shape}")
            df_raw = df_resampled # Replace df_raw with the resampled version for further processing
        else:
            print(f"Warning: Column '{time_col_original_name}' not found. Skipping resampling.")
        # --- END: Time Step Analysis ---

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data loading or resampling for {file_path}: {e}")
        return None

    processed_df = pd.DataFrame()

    # Columns to select and rename
    # Invert the mapping for easier lookup: csv_col_name -> standard_pid_name
    csv_to_standard_map = {v: k for k, v in current_pid_mapping.items() if v is not None}

    for csv_col_name, standard_pid_name in csv_to_standard_map.items():
        if csv_col_name in df_raw.columns:
            if standard_pid_name in RELEVANT_PIDS: # Only process if it's in our master list of relevant PIDs
                # Convert to numeric, coercing errors. This will turn non-convertible strings into NaN.
                processed_df[standard_pid_name] = pd.to_numeric(df_raw[csv_col_name], errors='coerce')
            else:
                # If a mapped CSV column is somehow not in this specific file, add it as NaNs
                processed_df[standard_pid_name] = pd.NA
        else:
            # If a mapped CSV column is somehow not in this specific file, add it as NaNs
            processed_df[standard_pid_name] = pd.NA

    # Ensure all RELEVANT_PIDS are columns, adding NaN columns if they weren't in the mapping or data
    for standard_pid_name in RELEVANT_PIDS:
        if standard_pid_name not in processed_df.columns:
            processed_df[standard_pid_name] = pd.NA

    # Reorder columns to match RELEVANT_PIDS for consistency, if RELEVANT_PIDS is comprehensive
    # Or, just ensure the ones we processed are there. For now, let's ensure order based on all_relevant_pids
    # Filter all_relevant_pids to only those that could have been processed or added
    final_columns = [pid for pid in RELEVANT_PIDS if pid in processed_df.columns]
    # Add any columns that were in pid_mapping but not in RELEVANT_PIDS (if any)
    for pid in current_pid_mapping.keys():
        if pid not in final_columns and pid in processed_df.columns:
            final_columns.append(pid)

    processed_df = processed_df[final_columns]

    if processed_df.empty and RELEVANT_PIDS:
        # If no relevant PIDs were found or processed, return an empty DataFrame with RELEVANT_PIDS as columns
        # This can happen if PID_COLUMN_MAPPING is incorrect or file is very different
        print(f"Warning: Processed DataFrame is empty for {file_path} but RELEVANT_PIDS are specified. Returning DataFrame with NaN columns.")
        return pd.DataFrame(columns=RELEVANT_PIDS, dtype='float64')
    elif processed_df.empty:
        print(f"Warning: Processed DataFrame is empty for {file_path} and no RELEVANT_PIDS to structure. Returning empty DataFrame.")
        return pd.DataFrame()

    return processed_df

def load_volvo_v40_csv(file_path: str, pid_mapping: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
    """
    Loads and preprocesses a single CSV file from the Volvo V40 (CarScanner) dataset.
    Pivots, resamples, selects relevant PIDs, and converts to numeric types.
    """
    current_pid_mapping = pid_mapping if pid_mapping is not None else PID_COLUMN_MAPPING_VOLVO_V40

    try:
        df_long = pd.read_csv(file_path, delimiter=';')
        print(f"Successfully loaded long format CSV: {file_path} with shape {df_long.shape}")

        expected_cols = ["SECONDS", "PID", "VALUE"]
        if not all(col in df_long.columns for col in expected_cols):
            print(f"Error: CSV {file_path} does not contain expected columns. Found: {df_long.columns.tolist()}")
            return None

        try:
            df_wide = df_long.pivot(index="SECONDS", columns="PID", values="VALUE")
        except ValueError as ve:
            if "Index contains duplicate entries, cannot reshape" in str(ve):
                print(f"Warning: Duplicate (SECONDS, PID) pairs found. Dropping duplicates, keeping last.")
                df_long.drop_duplicates(subset=["SECONDS", "PID"], keep='last', inplace=True)
                df_wide = df_long.pivot(index="SECONDS", columns="PID", values="VALUE")
            else:
                raise

        print(f"Pivoted to wide format. Shape: {df_wide.shape}")
        df_wide = df_wide.reset_index()
        df_wide.rename(columns={"SECONDS": "Time (sec)"}, inplace=True)
        print(f"'SECONDS' column reset and renamed to 'Time (sec)'.")

        # Convert all columns (except Time (sec)) to numeric, coercing errors
        pid_cols_to_convert = [col for col in df_wide.columns if col != "Time (sec)"]
        for col in pid_cols_to_convert:
            df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
        print(f"Converted {len(pid_cols_to_convert)} PID columns to numeric.")

        # --- Resampling Logic ---
        if "Time (sec)" not in df_wide.columns:
            print("Error: 'Time (sec)' column not found after pivot for resampling.")
            return None

        print(f"Resampling data to 1-second frequency for {os.path.basename(file_path)}...")
        df_wide["Time (sec)"] = pd.to_numeric(df_wide["Time (sec)"], errors='coerce')
        df_wide.dropna(subset=["Time (sec)"], inplace=True)
        if df_wide.empty:
            print("DataFrame is empty after dropping NaN Time (sec) values. Cannot resample.")
            return None

        df_wide["Timestamp"] = pd.to_timedelta(df_wide["Time (sec)"], unit='s')
        df_wide.set_index("Timestamp", inplace=True)

        if df_wide.index.has_duplicates:
            print(f"Warning: Duplicate timestamps found. Averaging values at duplicate timestamps before resampling.")
            df_wide = df_wide.groupby(df_wide.index).mean()

        # df_resampled = df_wide.resample('1s').ffill().bfill() # Old: ffill then bfill
        df_resampled = df_wide.resample('1s').mean() # New: Use mean for aggregation within 1s bins

        df_resampled.reset_index(inplace=True)
        df_resampled["Time (sec)"] = df_resampled["Timestamp"].dt.total_seconds()
        df_resampled.drop(columns=["Timestamp"], inplace=True)
        print(f"Resampling complete. New shape: {df_resampled.shape}")

        # --- Select and Rename PIDs ---
        processed_df = pd.DataFrame()
        if "Time (sec)" in df_resampled.columns:
            processed_df["TIME_SEC"] = df_resampled["Time (sec)"]
        carscanner_to_standard_map = {v: k for k, v in current_pid_mapping.items() if v is not None}
        for carscanner_pid_name, standard_pid_name in carscanner_to_standard_map.items():
            if standard_pid_name in RELEVANT_PIDS:
                if carscanner_pid_name in df_resampled.columns:
                    processed_df[standard_pid_name] = df_resampled[carscanner_pid_name]
                else:
                    processed_df[standard_pid_name] = pd.Series([pd.NA] * len(df_resampled))
        if RELEVANT_PIDS:
            for std_pid in RELEVANT_PIDS:
                if std_pid not in processed_df.columns:
                    processed_df[std_pid] = pd.Series([pd.NA] * len(df_resampled))
        else:
            print("Warning: RELEVANT_PIDS list is empty or not available.")
        if RELEVANT_PIDS and "TIME_SEC" in RELEVANT_PIDS:
            ordered_cols = [pid for pid in RELEVANT_PIDS if pid in processed_df.columns]
            missing_cols = [pid for pid in RELEVANT_PIDS if pid not in processed_df.columns]
            final_cols = ordered_cols + missing_cols
            processed_df = processed_df.reindex(columns=final_cols)
        elif RELEVANT_PIDS:
            ordered_cols = [pid for pid in RELEVANT_PIDS if pid in processed_df.columns]
            processed_df = processed_df.reindex(columns=ordered_cols)
        print(f"PID selection and renaming complete. Final shape: {processed_df.shape}")
        return processed_df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing Volvo V40 CSV {file_path}: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error during development
        return None

# if __name__ == "__main__":
#     # ... (all the old test code will be commented out)
#     print("Testing app.data_loader.py...")
#
#     # Define example RELEVANT_PIDS and PID_COLUMN_MAPPING if not imported (e.g., for standalone test)
#     if not RELEVANT_PIDS:
#         RELEVANT_PIDS = [
#             "ENGINE_RPM", "COOLANT_TEMPERATURE", "VEHICLE_SPEED",
#             "ENGINE_LOAD", "INTAKE_AIR_TEMPERATURE", "THROTTLE_POSITION"
#         ]
#         print("Using fallback RELEVANT_PIDS for standalone test.")
#
#     if not PID_COLUMN_MAPPING:
#         PID_COLUMN_MAPPING = {
#             "ENGINE_RPM": "Engine RPM (RPM)",
#             "COOLANT_TEMPERATURE": "Engine coolant temperature (°C)",
#             "VEHICLE_SPEED": "Vehicle speed (MPH)", # Example, original might differ
#             "ENGINE_LOAD": "Calculated load value (%)",
#             "INTAKE_AIR_TEMPERATURE": "Intake air temperature (°C)",
#             "THROTTLE_POSITION": "Absolute throttle position B (%)" # Example
#         }
#         print("Using fallback PID_COLUMN_MAPPING for standalone test.")
#
#     example_file_path = "data/datasets/romanian_driving_ds/dataset/r1_20220811-normal-behavior.csv"
#     print(f"Attempting to load: {example_file_path}")
#
#     try:
#         print(f"PID_COLUMN_MAPPING first item: {next(iter(PID_COLUMN_MAPPING.items())) if PID_COLUMN_MAPPING else 'Empty'}")
#     except NameError:
#         print("PID_COLUMN_MAPPING not loaded, example will likely fail or use limited columns.")
#         PID_COLUMN_MAPPING = {}
#         RELEVANT_PIDS = []
#
#     df = load_romanian_dataset_csv(example_file_path)
#
#     if df is not None:
#         print("Successfully loaded and processed DataFrame:")
#         print(f"Shape: {df.shape}")
#         print("Columns:", df.columns.tolist())
#         print("First 5 rows:")
#         print(df.head().to_string())
#         print("\nInfo:")
#         df.info()
#     else:
#         print("Failed to load DataFrame.")
#
#     print("\n--- Testing Volvo V40 Loader ---")
#     volvo_example_file = "data/datasets/data_volvo_v40/2019-03-05 19-30-27.csv"
#     print(f"Attempting to load Volvo V40 file: {volvo_example_file}")
#     df_volvo = load_volvo_v40_csv(volvo_example_file)
#
#     if df_volvo is not None:
#         print("\nSuccessfully loaded and pivoted Volvo V40 DataFrame:")
#         print(f"Shape: {df_volvo.shape}")
#         print(f"All Columns after pivot: {df_volvo.columns.tolist()}")
#         print("Head:")
#         print(df_volvo.head().to_string())
#     else:
#         print(f"Failed to load or process {volvo_example_file}")
#
#     print("\nVolvo V40 loader test finished.")