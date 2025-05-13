import pandas as pd
import sys
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) # Script is in scripts/

# Input file (original Volvo data, the fully processed one)
INPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/volvo_v40_full_raw_pids_final.parquet")

# Output file (Volvo data renamed for generic model, from original final)
OUTPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/volvo_v40_renamed_raw_pids_for_tier1_test.parquet")

# Define the standard PID names as found in the Volvo input file
# These seem to be standard OBD PIDs
STANDARD_PID_COLUMNS = [
    "ENGINE_RPM",                     # Matches target
    "COOLANT_TEMPERATURE",            # Needs rename
    "INTAKE_AIR_TEMPERATURE",         # Needs rename
    "THROTTLE_POSITION",            # Needs rename
    "VEHICLE_SPEED",                # Needs rename
    "CALCULATED_ENGINE_LOAD_VALUE", # Needs rename
]

# Define the target PID names expected by the generic tier1 model
# (Aligned with the combined general model training)
TARGET_PID_COLUMNS = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "INTAKE_AIR_TEMP", # Standardized
    "THROTTLE_POS",
    "VEHICLE_SPEED",   # Standardized
    "ENGINE_LOAD",
]

# Mapping from standard names to target names
COLUMN_RENAME_MAP = dict(zip(STANDARD_PID_COLUMNS, TARGET_PID_COLUMNS))

# Context columns to keep (must include TIME_SEC)
# Add other relevant columns found in the Volvo data
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
    # 'event_type' was noted as missing in inspection
    "make",
    "model",
    "fuel_type",
    # Add other potentially useful context if needed, e.g.:
    # "drive_mode", "from_location", "to_location"
]
# --- End Configuration ---

def prepare_renamed_volvo_data(input_path: str, output_path: str, rename_map: dict, context_cols: list[str]):
    """
    Reads the input Volvo parquet file, selects standard PID columns and context columns,
    renames the PIDs to match the generic Tier 1 model, and saves the result.
    """
    logging.info(f"Starting preparation of renamed Volvo data from: {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    standard_pids = list(rename_map.keys())
    target_pids_after_rename = list(rename_map.values())
    columns_to_read = standard_pids + context_cols
    columns_to_read = sorted(list(set(columns_to_read))) # Ensure unique and read only needed

    try:
        logging.info(f"Reading input file columns: {columns_to_read}")
        # Read only necessary columns to save memory
        df = pd.read_parquet(input_path, columns=columns_to_read)
        logging.info(f"Successfully read input file. Shape: {df.shape}")
    except Exception as e:
        # Log specific columns that might be missing
        try:
            # Check available columns without loading all data
            all_cols = pd.read_parquet(input_path, columns=[]).columns.tolist()
            missing = [c for c in columns_to_read if c not in all_cols]
            logging.error(f"Error reading Parquet file {input_path}. Missing columns might be: {missing}. Original error: {e}")
        except Exception as inner_e:
            logging.error(f"Error reading Parquet file {input_path}: {e}. Inner error checking columns: {inner_e}")
        sys.exit(1)

    # Verify required columns exist after loading
    actual_columns = df.columns.tolist()
    missing_cols = [col for col in columns_to_read if col not in actual_columns]
    if missing_cols:
        logging.error(f"Error: Input file is missing required columns after loading: {missing_cols}")
        sys.exit(1)

    # Rename columns
    logging.info(f"Renaming columns: {rename_map}")
    df_renamed = df.rename(columns=rename_map)

    # Verify rename worked - check if target PID names are present
    final_columns_expected = target_pids_after_rename + context_cols
    final_columns_expected = sorted(list(set(final_columns_expected))) # Ensure unique

    missing_after_rename = [col for col in target_pids_after_rename if col not in df_renamed.columns]
    if missing_after_rename:
         logging.error(f"Error: Columns missing after renaming: {missing_after_rename}. Check rename map.")
         sys.exit(1)

    # Select final columns in a consistent order
    # Ensure columns actually exist before selecting
    final_columns_present = [col for col in final_columns_expected if col in df_renamed.columns]
    df_output = df_renamed[final_columns_present]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"Saving renamed Volvo data for Tier 1 model (shape: {df_output.shape}) to: {output_path}")
        df_output.to_parquet(output_path, index=False)
        logging.info(f"Successfully saved renamed data.")
    except Exception as e:
        logging.error(f"Error saving output Parquet file to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_renamed_volvo_data(INPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH, COLUMN_RENAME_MAP, CONTEXT_COLUMNS) 