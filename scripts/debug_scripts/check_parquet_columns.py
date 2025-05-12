import pandas as pd
import sys
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Get path relative to the script directory or use an absolute path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..')) # Assumes script is in scripts/debug_scripts
PARQUET_FILE_PATH = os.path.join(project_root, "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet")

# Columns required for Tier 1 Anomaly Detection
TIER1_CORE_PIDS = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "AIR_INTAKE_TEMP",
    "THROTTLE_POS",
    "SPEED",
    "ENGINE_LOAD",
]

# Column required for segmentation
FUEL_TYPE_COLUMN = "FUEL_TYPE"
# --- End Configuration ---

def check_columns(file_path: str, required_pids: list[str], fuel_col: str):
    """
    Checks if a Parquet file contains the required columns.
    """
    logging.info(f"Checking columns in Parquet file: {file_path}")

    if not os.path.exists(file_path):
        logging.error(f"Error: File not found at {file_path}")
        sys.exit(1)

    actual_columns = []
    try:
        # Read just the first 5 rows to efficiently get columns and sample data
        df_sample = pd.read_parquet(file_path).head()
        actual_columns = df_sample.columns.tolist()
        logging.info(f"Successfully read sample. Found columns ({len(actual_columns)}): {actual_columns}")

    except Exception as e:
        # Fallback if reading head fails, try reading just schema (might depend on engine)
        try:
             # This requires pyarrow usually. If not installed, might fail.
             import pyarrow.parquet as pq
             schema = pq.read_schema(file_path)
             actual_columns = [field.name for field in schema]
             logging.info(f"Successfully read schema. Found columns ({len(actual_columns)}): {actual_columns}")
        except ImportError:
             logging.warning("pyarrow not found. Cannot read schema directly.")
             logging.error(f"Error reading Parquet file {file_path}: {e}")
             sys.exit(1)
        except Exception as schema_e:
             logging.error(f"Error reading Parquet file schema {file_path}: {schema_e}")
             logging.error(f"Original read error was: {e}")
             sys.exit(1)


    # Check for required PIDs
    missing_pids = [pid for pid in required_pids if pid not in actual_columns]
    found_pids = [pid for pid in required_pids if pid in actual_columns]

    if not missing_pids:
        logging.info(f"SUCCESS: All required TIER1_CORE_PIDS found: {required_pids}")
    else:
        logging.error(f"FAILED: Missing required TIER1_CORE_PIDS: {missing_pids}")
        if found_pids:
            logging.info(f"Found TIER1_CORE_PIDS: {found_pids}")

    # Check for fuel type column
    if fuel_col in actual_columns:
        logging.info(f"SUCCESS: Fuel type column '{fuel_col}' found.")
        # Check unique values in fuelType if found
        try:
            fuel_types_sample = df_sample[fuel_col].unique()
            logging.info(f"Sample unique values found in '{fuel_col}': {fuel_types_sample}")
            # Note: Reading unique values from the whole file might be slow
            # fuel_types_full = pd.read_parquet(file_path, columns=[fuel_col])[fuel_col].unique()
            # logging.info(f"Full unique values in '{fuel_col}': {fuel_types_full}")
        except Exception as e:
            logging.warning(f"Could not analyze unique values from '{fuel_col}': {e}")
    else:
        logging.error(f"FAILED: Fuel type column '{fuel_col}' not found.")

    if not missing_pids and fuel_col in actual_columns:
        logging.info("Conclusion: File appears suitable for Tier 1 data preparation.")
    else:
        logging.error("Conclusion: File is missing required columns for Tier 1 data preparation.")
        sys.exit(1) # Exit with error if checks fail

if __name__ == "__main__":
    check_columns(PARQUET_FILE_PATH, TIER1_CORE_PIDS, FUEL_TYPE_COLUMN)