import pandas as pd
import os
import logging
import sys
import argparse
import glob # For finding files

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define the target standard core PID names for the combined dataset
TARGET_CORE_PIDS = [
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP", # Standard name
    "INTAKE_AIR_TEMP",     # Standard name
    "THROTTLE_POS",        # Standard name
    "VEHICLE_SPEED",       # Standard name
    "ENGINE_LOAD",         # Standard name
]

# Define essential context columns to try and keep (and standardize their names)
TARGET_CONTEXT_COLUMNS = {
    "absolute_timestamp": "absolute_timestamp", # Already standard
    "make": "make",
    "model": "model",
    "fuel_type": "fuel_type",
    "vehicle_id": "vehicle_id", # For Kaggle
    # Romanian and Volvo use TIME_SEC, Kaggle doesn't have a direct equivalent after initial processing.
    # We will derive TIME_SEC for all from absolute_timestamp if it's missing.
    "TIME_SEC": "TIME_SEC"
}


# --- Dataset Specific Configurations ---
# These maps define how to rename original PIDs to TARGET_CORE_PIDS
# and original context columns to TARGET_CONTEXT_COLUMNS keys.

VOLVO_RAW_PID_MAP = {
    "ENGINE_RPM": "ENGINE_RPM",
    "COOLANT_TEMPERATURE": "ENGINE_COOLANT_TEMP",
    "INTAKE_AIR_TEMPERATURE": "INTAKE_AIR_TEMP",
    "THROTTLE_POSITION": "THROTTLE_POS", # Note: Volvo uses "THROTTLE_POSITION"
    "VEHICLE_SPEED": "VEHICLE_SPEED",
    "CALCULATED_ENGINE_LOAD_VALUE": "ENGINE_LOAD",
}
VOLVO_CONTEXT_MAP = {
    "absolute_timestamp": "absolute_timestamp",
    "TIME_SEC": "TIME_SEC",
    # 'make', 'model', 'fuel_type' are added programmatically
}

ROMANIAN_RAW_PID_MAP = {
    "ENGINE_RPM": "ENGINE_RPM",
    "COOLANT_TEMPERATURE": "ENGINE_COOLANT_TEMP",
    "INTAKE_AIR_TEMPERATURE": "INTAKE_AIR_TEMP",
    "THROTTLE_POSITION": "THROTTLE_POS",
    "VEHICLE_SPEED": "VEHICLE_SPEED",
    "CALCULATED_ENGINE_LOAD_VALUE": "ENGINE_LOAD",
}
ROMANIAN_CONTEXT_MAP = {
    "absolute_timestamp": "absolute_timestamp",
    "TIME_SEC": "TIME_SEC",
    # 'make', 'model', 'fuel_type' are added programmatically
}

# Kaggle PIDs in its 'raw_pids_model_input.parquet' are already mostly standard, but let's be explicit.
# The 'exp1_14drivers_14cars_dailyRoutes_raw_pids_model_input.parquet' has aggregated features.
# We need to ensure we pick the *original* raw PIDs, not aggregated ones if they exist.
# Based on preprocess_kaggle_dataset.py, the PIDs are:
KAGGLE_RAW_PID_MAP = {
    'ENGINE_RPM': 'ENGINE_RPM',
    'ENGINE_COOLANT_TEMP': 'ENGINE_COOLANT_TEMP',
    'AIR_INTAKE_TEMP': 'INTAKE_AIR_TEMP', # Kaggle uses AIR_INTAKE_TEMP
    'THROTTLE_POS': 'THROTTLE_POS',
    'SPEED': 'VEHICLE_SPEED',            # Kaggle uses SPEED
    'ENGINE_LOAD': 'ENGINE_LOAD',
}
KAGGLE_CONTEXT_MAP = {
    "absolute_timestamp": "absolute_timestamp",
    "MARK": "make",
    "MODEL": "model",
    "FUEL_TYPE": "fuel_type",
    "VEHICLE_ID": "vehicle_id"
    # TIME_SEC will be derived
}

DATASET_SOURCES = {
    "volvo": {
        "path": "data/model_input/volvo_v40_full_raw_pids_final.parquet",
        "pid_map": VOLVO_RAW_PID_MAP,
        "context_map": VOLVO_CONTEXT_MAP,
        "static_metadata": {"make": "Volvo", "model": "V40", "fuel_type": "Diesel"}
    },
    "romanian": {
        "path": "data/model_input/romanian_driving_ds_raw_pids_final.parquet",
        "pid_map": ROMANIAN_RAW_PID_MAP,
        "context_map": ROMANIAN_CONTEXT_MAP,
        "static_metadata": {"make": "Volkswagen", "model": "Passat", "fuel_type": "Diesel"}
    },
    "kaggle": {
        # This is the output from the 'kaggle_raw_pids' pipeline's aggregate step
        "path": "data/model_input/exp1_14drivers_14cars_dailyRoutes_raw_pids_model_input.parquet",
        "pid_map": KAGGLE_RAW_PID_MAP,
        "context_map": KAGGLE_CONTEXT_MAP,
        "static_metadata": {} # Make, model, fuel_type should be in the columns
    },
    "Toyota": {
        "path": os.path.join(project_root, "data/model_input/toyota_etios_raw_pids_final.parquet"),
        "pid_map": { # From toyota_etios_raw_pids_final.parquet -> Target Standard Name
            # "TIME_SEC": "TIME_SEC", # Removed from PIDs, will be handled by context_map
            "ENGINE_RPM": "ENGINE_RPM",
            "VEHICLE_SPEED": "VEHICLE_SPEED",
            # "ENGINE_COOLANT_TEMP": "ENGINE_COOLANT_TEMP", # Excluded due to suspicious values
            "THROTTLE_POS": "THROTTLE_POS",
            "ENGINE_LOAD": "ENGINE_LOAD",
            "INTAKE_AIR_TEMP": "INTAKE_AIR_TEMP",
        },
        "context_map": { # From toyota_etios_raw_pids_final.parquet -> Target Standard Name
            "absolute_timestamp": "absolute_timestamp",
            "TIME_SEC": "TIME_SEC", # Ensured TIME_SEC is here
            "make": "make",
            "model": "model",
            "fuel_type": "fuel_type",
            "source_file": "source_file" # Keep for traceability
        },
        "static_metadata": {}, # Toyota data already has make, model, fuel_type
        # "time_col": "TIME_SEC", # This was an old remnant, TIME_SEC is now in context_map
        # "is_volvo_format": False,
        # "needs_primary_key_generation": True # Multiple files, treat as distinct series
    }
}

def load_and_prepare_dataset(config: dict, dataset_name: str, project_root: str) -> pd.DataFrame | None:
    """Loads a single dataset, selects, renames, and adds context."""
    file_path = os.path.join(project_root, config["path"])
    logging.info(f"Processing {dataset_name} from {file_path}...")

    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}. Skipping {dataset_name}.")
        return None
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Loaded {dataset_name} with shape {df.shape}. Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}", exc_info=True)
        return None

    # --- PID Selection and Renaming ---
    selected_pids = {}
    missing_pids_in_source = []
    for original_pid, target_pid in config["pid_map"].items():
        if original_pid in df.columns:
            selected_pids[target_pid] = df[original_pid]
        else:
            missing_pids_in_source.append(original_pid)
            selected_pids[target_pid] = pd.Series(dtype='float64') # Add empty series to maintain structure

    if missing_pids_in_source:
        logging.warning(f"For {dataset_name}, PIDs missing in source file and will be NaN: {missing_pids_in_source}")

    pid_df = pd.DataFrame(selected_pids)

    # --- Context Column Selection and Renaming ---
    selected_context = {}
    missing_context_in_source = []
    for original_ctx, target_ctx_name in config["context_map"].items():
        if original_ctx in df.columns:
            selected_context[target_ctx_name] = df[original_ctx]
        else:
            # Don't warn for TIME_SEC yet as we can derive it
            if target_ctx_name != "TIME_SEC":
                missing_context_in_source.append(original_ctx)
            selected_context[target_ctx_name] = pd.Series(dtype='object') # Maintain structure

    if missing_context_in_source:
        logging.warning(f"For {dataset_name}, context columns missing/will be NaN: {missing_context_in_source}")

    context_df = pd.DataFrame(selected_context)

    # Add static metadata
    for col, val in config.get("static_metadata", {}).items():
        context_df[col] = val
        logging.info(f"Added static metadata for {dataset_name}: {col}={val}")

    # Ensure all target context columns are present, filling with NaN if necessary
    for target_ctx_name in TARGET_CONTEXT_COLUMNS.values():
        if target_ctx_name not in context_df.columns:
            context_df[target_ctx_name] = pd.Series(dtype='object' if target_ctx_name not in ["TIME_SEC"] else 'float64')


    # Combine PIDs and Context
    combined_df = pd.concat([context_df, pid_df], axis=1)

    # Derive TIME_SEC if 'absolute_timestamp' exists and 'TIME_SEC' is missing or all NaNs
    if 'absolute_timestamp' in combined_df.columns and combined_df['absolute_timestamp'].notna().any():
        if 'TIME_SEC' not in combined_df.columns or combined_df['TIME_SEC'].isnull().all():
            logging.info(f"Deriving TIME_SEC for {dataset_name} from absolute_timestamp.")
            # Ensure absolute_timestamp is datetime
            combined_df['absolute_timestamp'] = pd.to_datetime(combined_df['absolute_timestamp'], errors='coerce')
            # Calculate time difference from the start of each group (approximated by first timestamp)
            # This is tricky without a clear trip identifier, so we'll do it per file for now.
            # A more robust way would be to group by vehicle_id if present and continuous blocks of time.
            # For simplicity here, assume timestamps are sorted and calculate from the first valid one.
            first_valid_timestamp = combined_df['absolute_timestamp'].dropna().iloc[0] if combined_df['absolute_timestamp'].notna().any() else pd.NaT
            if pd.notna(first_valid_timestamp):
                combined_df['TIME_SEC'] = (combined_df['absolute_timestamp'] - first_valid_timestamp).dt.total_seconds()
            else:
                combined_df['TIME_SEC'] = pd.NA # or np.nan

    # Add a source column
    combined_df['dataset_source'] = dataset_name

    # Select only the defined target PIDs and context columns in the correct order
    final_columns = list(TARGET_CONTEXT_COLUMNS.values()) + TARGET_CORE_PIDS + ['dataset_source']
    # Filter out any columns that are not in combined_df (e.g., if a PID was missing everywhere)
    final_columns_present = [col for col in final_columns if col in combined_df.columns]

    processed_df = combined_df[final_columns_present].copy()

    logging.info(f"Finished preparing {dataset_name}. Shape: {processed_df.shape}. Columns: {processed_df.columns.tolist()}")
    return processed_df


def main(args):
    logging.info("Starting dataset combination for Tier 1 training.")

    all_dfs = []
    for name, config in DATASET_SOURCES.items():
        df = load_and_prepare_dataset(config, name, project_root)
        if df is not None and not df.empty:
            all_dfs.append(df)
        else:
            logging.warning(f"No data loaded for {name}, it will be excluded from the combined dataset.")

    if not all_dfs:
        logging.error("No dataframes to combine. Exiting.")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Successfully combined all datasets. Final shape: {combined_df.shape}")

    # Final check for column order and presence
    final_ordered_columns = list(TARGET_CONTEXT_COLUMNS.values()) + TARGET_CORE_PIDS + ['dataset_source']
    # Ensure all expected columns are present, add NaN columns if any are entirely missing
    for col in final_ordered_columns:
        if col not in combined_df.columns:
            logging.warning(f"Column '{col}' was not found in any source dataset and is being added as full NaN column.")
            combined_df[col] = pd.NA # or np.nan depending on expected dtype, but NA handles mixed types better

    combined_df = combined_df[final_ordered_columns] # Enforce order

    # Report missing values in core PIDs
    for pid in TARGET_CORE_PIDS:
        if pid in combined_df.columns:
            missing_percentage = combined_df[pid].isnull().mean() * 100
            if missing_percentage > 0:
                logging.warning(f"PID '{pid}' has {missing_percentage:.2f}% missing values in the combined dataset.")
        else:
            logging.error(f"Critical Error: Target PID '{pid}' is not in the final combined DataFrame columns!")


    # Save the combined dataset
    output_path = os.path.join(project_root, args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        combined_df.to_parquet(output_path, index=False, engine='pyarrow')
        logging.info(f"Successfully saved combined dataset to: {output_path}")
        logging.info(f"Final combined dataset columns: {combined_df.columns.tolist()}")
        logging.info(f"Final combined dataset info:")
        combined_df.info(verbose=True, show_counts=True)


    except Exception as e:
        logging.error(f"Error saving combined dataset: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Dataset combination finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine processed raw PID datasets for Tier 1 model training.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/model_input/combined_raw_pids_for_tier1_training.parquet",
        help="Path to save the combined Parquet file (relative to project root)."
    )
    # Add arguments for individual dataset paths if needed for more flexibility later,
    # but for now, paths are hardcoded in DATASET_SOURCES.

    args = parser.parse_args()
    main(args)