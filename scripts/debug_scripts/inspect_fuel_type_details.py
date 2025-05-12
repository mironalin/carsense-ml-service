import pandas as pd
import sys
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
PARQUET_FILE_PATH = os.path.join(project_root, "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet")

FUEL_TYPE_COL = "FUEL_TYPE"
VEHICLE_INFO_COLS = ["MARK", "MODEL", "ENGINE_POWER"]
# --- End Configuration ---

def inspect_fuel_types(file_path: str, fuel_col: str, info_cols: list[str]):
    """
    Inspects unique fuel types and shows associated vehicle make, model, and engine power.
    """
    logging.info(f"Inspecting fuel types in Parquet file: {file_path}")

    if not os.path.exists(file_path):
        logging.error(f"Error: File not found at {file_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(file_path, columns=[fuel_col] + info_cols)
        logging.info(f"Successfully read relevant columns. DataFrame shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error reading Parquet file {file_path}: {e}")
        sys.exit(1)

    # Handle potential explicit string 'null' or actual NaN/None values
    # Convert explicit 'null' strings to NaN to treat them uniformly with None/NaN
    df[fuel_col] = df[fuel_col].replace('null', pd.NA)
    unique_fuel_types = df[fuel_col].unique() # This will include pd.NA if NaNs are present

    logging.info(f"Found unique values in '{fuel_col}' (NaNs represented as pd.NA): {unique_fuel_types}")

    for fuel_type in unique_fuel_types:
        logging.info(f"\n--- Details for FUEL_TYPE: {fuel_type if pd.notna(fuel_type) else 'NULL_VALUE'} ---")
        
        if pd.isna(fuel_type):
            subset_df = df[df[fuel_col].isna()]
        else:
            subset_df = df[df[fuel_col] == fuel_type]
        
        if subset_df.empty:
            logging.info("No vehicles found for this fuel type category (this shouldn't happen if unique_fuel_types is derived from df)." )
            continue

        # Show top N unique combinations of make, model, engine_power
        # Group by all info_cols and take the head(1) of each group, then select top N of those unique groups
        # This ensures we see variety if multiple cars share the same fuel type.
        unique_vehicle_samples = subset_df[info_cols].drop_duplicates().head(5) # Show up to 5 unique samples
        
        if not unique_vehicle_samples.empty:
            logging.info(f"Sample vehicle details (up to 5 unique make/model/power combinations):")
            for _, row in unique_vehicle_samples.iterrows():
                details = ", ".join([f"{col}: {row[col]}" for col in info_cols])
                logging.info(f"  - {details}")
        else:
            logging.info("No vehicle details to display for this fuel type (empty subset or no unique info)." )

if __name__ == "__main__":
    inspect_fuel_types(PARQUET_FILE_PATH, FUEL_TYPE_COL, VEHICLE_INFO_COLS) 