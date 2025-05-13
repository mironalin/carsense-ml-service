import pandas as pd
import sys
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) # Script is in scripts/

# Input file (Romanian data - RAW PIDs final version)
INPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/romanian_driving_ds_raw_pids_final.parquet")

# Output file (Romanian data renamed for generic model - from RAW PIDs)
OUTPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/romanian_renamed_raw_pids_for_generic_tier1.parquet")

# Define the standard PID names as found in the Romanian input file (romanian_driving_ds_raw_pids_final.parquet)
# Assuming these are the verbose names from the full processing pipeline before final selection for generic model
STANDARD_PID_COLUMNS = [
    "ENGINE_RPM",                     # Matches (assuming this is the name in _raw_pids_final)
    "COOLANT_TEMPERATURE",            # Source name for ENGINE_COOLANT_TEMP
    "INTAKE_AIR_TEMPERATURE",         # Source name for INTAKE_AIR_TEMP
    "THROTTLE_POSITION",            # Source name for THROTTLE_POS
    "VEHICLE_SPEED",                # Source name for VEHICLE_SPEED
    "CALCULATED_ENGINE_LOAD_VALUE", # Source name for ENGINE_LOAD
]

# Define the target PID names expected by the NEW generic tier1 model (TIER1_CORE_PIDS)
TARGET_PID_COLUMNS = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "INTAKE_AIR_TEMP",      # CORRECTED
    "THROTTLE_POS",
    "VEHICLE_SPEED",        # CORRECTED
    "ENGINE_LOAD",
]

# Mapping from standard names to target names
COLUMN_RENAME_MAP = dict(zip(STANDARD_PID_COLUMNS, TARGET_PID_COLUMNS))

# Context columns to keep (must include TIME_SEC)
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
    "event_type",
    "make",
    "model",
    "fuel_type"
]
# --- End Configuration ---

def prepare_renamed_romanian_data(input_path: str, output_path: str, rename_map: dict, context_cols: list[str]):
    """
    Reads the input Romanian parquet file, selects standard PID columns and context columns,
    renames the PIDs to match the generic Tier 1 model, and saves the result.
    """
    logging.info(f"Starting preparation of renamed Romanian data from: {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    standard_pids = list(rename_map.keys())
    columns_to_read = standard_pids + context_cols
    columns_to_read = sorted(list(set(columns_to_read))) # Ensure unique

    try:
        logging.info(f"Reading input file columns: {columns_to_read}")
        df = pd.read_parquet(input_path, columns=columns_to_read)
        logging.info(f"Successfully read input file. Shape: {df.shape}")
    except Exception as e:
        # Log specific columns that might be missing
        try:
            all_cols = pd.read_parquet(input_path, columns=[]).columns.tolist()
            missing = [c for c in columns_to_read if c not in all_cols]
            logging.error(f"Error reading Parquet file {input_path}. Missing columns might be: {missing}. Original error: {e}")
        except Exception:
            logging.error(f"Error reading Parquet file {input_path}: {e}")
        sys.exit(1)

    # Verify required columns exist
    actual_columns = df.columns.tolist()
    missing_cols = [col for col in columns_to_read if col not in actual_columns]
    if missing_cols:
        logging.error(f"Error: Input file is missing required columns: {missing_cols}")
        sys.exit(1)

    # Rename columns
    logging.info(f"Renaming columns: {rename_map}")
    df_renamed = df.rename(columns=rename_map)

    # Verify rename worked - check if target names are present
    target_pids = list(rename_map.values())
    final_columns_expected = target_pids + context_cols
    final_columns_expected = sorted(list(set(final_columns_expected))) # Ensure unique

    missing_after_rename = [col for col in target_pids if col not in df_renamed.columns]
    if missing_after_rename:
         logging.error(f"Error: Columns missing after renaming: {missing_after_rename}. Check rename map.")
         sys.exit(1)

    # Select final columns in a consistent order
    df_output = df_renamed[final_columns_expected]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"Saving renamed Romanian data for Tier 1 model (shape: {df_output.shape}) to: {output_path}")
        # Use index=True if the original index is meaningful (e.g., timestamp)
        # Assuming index is not critical here based on previous saves.
        df_output.to_parquet(output_path, index=False)
        logging.info(f"Successfully saved renamed data.")
    except Exception as e:
        logging.error(f"Error saving output Parquet file to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_renamed_romanian_data(INPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH, COLUMN_RENAME_MAP, CONTEXT_COLUMNS) 