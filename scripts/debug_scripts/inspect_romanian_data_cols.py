import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
INPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/romanian_driving_ds_final.parquet")
COLUMNS_TO_CHECK = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "AIR_INTAKE_TEMP",
    "THROTTLE_POS",
    "SPEED",
    "ENGINE_LOAD",
    "TIME_SEC",        # Check for this first
    "ENGINE_RUNTIME"   # Check for this as fallback
]
NUM_UNIQUE_TO_SHOW = 20

def inspect_columns(file_path: str, column_list: list[str]):
    logging.info(f"Inspecting columns in file: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    try:
        # Get all column names from the file
        # Read the first row to get column names more robustly
        df_sample = pd.read_parquet(file_path, engine='pyarrow') # Explicitly use pyarrow
        if df_sample.empty:
            logging.error("Parquet file appears to be empty.")
            return
        all_columns = df_sample.columns.tolist()
        logging.info(f"Total columns found in file: {len(all_columns)}")
        logging.info(f"Column names: {all_columns}") # Log all column names

        # Check presence of required columns
        logging.info("\nChecking for required columns...")
        found_core_pids = []
        missing_core_pids = []
        found_time_col = None
        time_col_name = None

        core_pids_to_check = column_list[:-2] # Exclude TIME_SEC and ENGINE_RUNTIME for now
        time_cols_to_check = column_list[-2:] # Check TIME_SEC then ENGINE_RUNTIME

        for col in core_pids_to_check:
            if col in all_columns:
                found_core_pids.append(col)
            else:
                missing_core_pids.append(col)

        if missing_core_pids:
            logging.warning(f"Missing core PIDs: {missing_core_pids}")
        else:
            logging.info(f"Found all required core PIDs: {found_core_pids}")

        # Check for time column
        for col in time_cols_to_check:
            if col in all_columns:
                found_time_col = True
                time_col_name = col
                logging.info(f"Found time column: '{time_col_name}'")
                break
        if not found_time_col:
            logging.warning(f"Did not find 'TIME_SEC' or 'ENGINE_RUNTIME' column.")
            time_col_name = None

        # Inspect the time column if found
        if time_col_name:
            logging.info(f"\nInspecting time column: '{time_col_name}'")
            df_time = pd.read_parquet(file_path, columns=[time_col_name])
            col_dtype = df_time[time_col_name].dtype
            logging.info(f"Data type: {col_dtype}")
            nan_count = df_time[time_col_name].isnull().sum()
            logging.info(f"NaN count: {nan_count}")

            if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
                logging.info(f"Attempting conversion to numeric (this might be slow)...")
                numeric_col = pd.to_numeric(df_time[time_col_name], errors='coerce')
                coerced_nan_count = numeric_col.isnull().sum()
                if coerced_nan_count > nan_count:
                    logging.warning(f"Numeric conversion would coerce {coerced_nan_count - nan_count} non-NaN values to NaN.")
                    unique_values = df_time[time_col_name][numeric_col.isnull() & df_time[time_col_name].notnull()].unique()
                    num_unique = len(unique_values)
                    logging.info(f"Showing first {min(num_unique, NUM_UNIQUE_TO_SHOW)} unique values causing coercion:")
                    unique_list = list(unique_values)
                    for i, val in enumerate(unique_list[:NUM_UNIQUE_TO_SHOW]):
                        logging.info(f"  [{i}]: {val} (Type: {type(val)})")
                else:
                    logging.info("Numeric conversion seems feasible without coercing valid values.")
            elif pd.api.types.is_numeric_dtype(col_dtype):
                logging.info("Time column is already numeric.")
            else:
                 logging.warning(f"Time column has unexpected data type: {col_dtype}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    inspect_columns(INPUT_PARQUET_PATH, COLUMNS_TO_CHECK) 