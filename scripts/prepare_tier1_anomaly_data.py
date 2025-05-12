import pandas as pd
import sys
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) # Script is in scripts/

# Input file containing data from potentially mixed sources
INPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet")

# Output file containing only the selected Tier 1 PIDs for all rows
OUTPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/generic_tier1_data.parquet")

# Define the exact column names for the Tier 1 PIDs as found in the input file
TIER1_PID_COLUMNS = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "AIR_INTAKE_TEMP",
    "THROTTLE_POS",
    "SPEED",
    "ENGINE_LOAD",
]
# --- End Configuration ---

def prepare_generic_tier1_data(input_path: str, output_path: str, pid_columns: list[str]):
    """
    Reads the input parquet file, selects only the Tier 1 PID columns,
    and saves the result to the output path. Includes data from all fuel types.
    """
    logging.info(f"Starting preparation of generic Tier 1 data from: {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    try:
        logging.info(f"Reading input file: {input_path}")
        df = pd.read_parquet(input_path)
        logging.info(f"Successfully read input file. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error reading Parquet file {input_path}: {e}")
        sys.exit(1)

    # Verify required columns exist
    actual_columns = df.columns.tolist()
    missing_pids = [pid for pid in pid_columns if pid not in actual_columns]
    if missing_pids:
        logging.error(f"Error: Input file is missing required Tier 1 PID columns: {missing_pids}")
        sys.exit(1)
    
    logging.info(f"Selecting Tier 1 PID columns: {pid_columns}")
    df_tier1 = df[pid_columns]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"Saving generic Tier 1 data (shape: {df_tier1.shape}) to: {output_path}")
        df_tier1.to_parquet(output_path, index=False) # index=False might be suitable if index isn't needed for training
        logging.info(f"Successfully saved generic Tier 1 data.")
    except Exception as e:
        logging.error(f"Error saving output Parquet file to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_generic_tier1_data(INPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH, TIER1_PID_COLUMNS) 