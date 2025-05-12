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
    # Excluded CONTROL_MODULE_VOLTAGE as requested
]

# Define extra context columns to include in the output file
EXTRA_CONTEXT_COLUMNS = [
    "absolute_timestamp",
    "ENGINE_RUNTIME", # Will be renamed to TIME_SEC
    "hour",
    "dayofweek",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos"
]

# --- End Configuration ---

def prepare_generic_tier1_data(input_path: str, output_path: str, pid_columns: list[str], context_columns: list[str]):
    """
    Reads the input parquet file, selects Tier 1 PID columns and extra context columns,
    renames ENGINE_RUNTIME to TIME_SEC, and saves the result to the output path.
    """
    logging.info(f"Starting preparation of generic Tier 1 data from: {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    columns_to_read = pid_columns + context_columns
    # Ensure unique columns if overlap exists (though unlikely here)
    columns_to_read = sorted(list(set(columns_to_read)))

    try:
        logging.info(f"Reading input file columns: {columns_to_read}")
        # Read only necessary columns for efficiency
        df = pd.read_parquet(input_path, columns=columns_to_read)
        logging.info(f"Successfully read input file. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error reading Parquet file {input_path}: {e}")
        sys.exit(1)

    # Verify required columns exist (after reading)
    actual_columns = df.columns.tolist()
    missing_cols = [col for col in columns_to_read if col not in actual_columns]
    if missing_cols:
        logging.error(f"Error: Input file is missing required columns: {missing_cols}")
        sys.exit(1)

    # Rename ENGINE_RUNTIME to TIME_SEC if it exists and convert to total seconds
    if "ENGINE_RUNTIME" in df.columns:
        logging.info("Processing 'ENGINE_RUNTIME' column...")

        # Define a function to convert HH:MM:SS string or NaNs to total seconds
        def time_str_to_seconds(time_str):
            if pd.isna(time_str): # Handle existing NaNs
                return None
            if isinstance(time_str, str):
                try:
                    parts = list(map(int, time_str.split(':')))
                    if len(parts) == 3:
                        td = pd.Timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
                        return td.total_seconds()
                    else:
                        # Handle unexpected format
                        return None
                except (ValueError, AttributeError):
                    # Handle parsing errors or non-string types that are not NaN
                    return None
            elif isinstance(time_str, (int, float)): # If it's already numeric, keep it
                 return float(time_str)
            return None # Default case for other unexpected types

        logging.info("Applying conversion from HH:MM:SS string to total seconds...")
        df["TIME_SEC"] = df["ENGINE_RUNTIME"].apply(time_str_to_seconds)

        # Log conversion results
        original_nan_count = df["ENGINE_RUNTIME"].isnull().sum()
        final_nan_count = df["TIME_SEC"].isnull().sum()
        logging.info(f"Original NaNs in 'ENGINE_RUNTIME': {original_nan_count}")
        logging.info(f"Final NaNs in 'TIME_SEC' after conversion: {final_nan_count}")
        if final_nan_count > original_nan_count:
             logging.warning(f"{final_nan_count - original_nan_count} additional NaNs introduced during conversion (likely due to parsing errors).")

        # Remove the original ENGINE_RUNTIME column
        # df = df.drop(columns=["ENGINE_RUNTIME"])

        # Update context_columns list for downstream consistency checks
        if "ENGINE_RUNTIME" in context_columns:
            context_columns = [col if col != "ENGINE_RUNTIME" else "TIME_SEC" for col in context_columns]
            # Ensure ENGINE_RUNTIME is removed if it was in context_columns
            context_columns = [col for col in context_columns if col in df.columns or col == "TIME_SEC"]

    else:
        logging.warning("Column 'ENGINE_RUNTIME' not found, 'TIME_SEC' will not be created.")

    # Define final columns for the output file
    # Make sure we only include columns actually present in the DataFrame now
    available_pid_columns = [col for col in pid_columns if col in df.columns]
    available_context_columns = [col for col in context_columns if col in df.columns]
    final_columns = available_pid_columns + available_context_columns
    final_columns = sorted(list(set(final_columns))) # Ensure unique and consistent order

    logging.info(f"Selecting final columns for output: {final_columns}")
    df_output = df[final_columns]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"Saving generic Tier 1 data (shape: {df_output.shape}) to: {output_path}")
        df_output.to_parquet(output_path, index=False) # index=False might be suitable if index isn't needed for training
        logging.info(f"Successfully saved generic Tier 1 data.")
    except Exception as e:
        logging.error(f"Error saving output Parquet file to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_generic_tier1_data(INPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH, TIER1_PID_COLUMNS, EXTRA_CONTEXT_COLUMNS) 