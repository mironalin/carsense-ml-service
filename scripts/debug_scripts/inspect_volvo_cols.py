import pandas as pd
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Script is in scripts/debug_scripts/

# Input file
DATA_PATH = os.path.join(project_root, "data/model_input/volvo_v40_full_final.parquet")
# --- End Configuration ---

def inspect_volvo_columns(data_path: str):
    """
    Loads the Volvo Parquet file and prints column information.
    """
    logging.info(f"Attempting to load data from: {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Error: Input file not found at {data_path}")
        return

    try:
        df = pd.read_parquet(data_path)
        logging.info(f"Successfully loaded data. Shape: {df.shape}")

        logging.info("\n--- Columns and Data Types ---")
        logging.info(df.dtypes)

        potential_pid_cols = [
            "ENGINE_RPM", "RPM",
            "COOLANT_TEMPERATURE", "ENGINE_COOLANT_TEMP",
            "INTAKE_AIR_TEMPERATURE", "AIR_INTAKE_TEMP",
            "THROTTLE_POSITION", "THROTTLE_POS", "APP_D", "APP_E",
            "VEHICLE_SPEED", "SPEED",
            "CALCULATED_ENGINE_LOAD_VALUE", "ENGINE_LOAD", "ABSOLUTE_LOAD_VALUE",
            "TIME_SEC"
        ]

        logging.info("\n--- Descriptive Statistics for Potential PID Columns ---")
        for col in df.columns:
            if col in potential_pid_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    logging.info(f"\nStatistics for column: {col}")
                    logging.info(df[col].describe())
                    # Show a few unique values if discrete or few unique
                    if df[col].nunique() < 20:
                        logging.info(f"Unique values in {col}: {df[col].unique()}")
                else:
                    logging.info(f"\nColumn {col} is not numeric. Value counts (top 5):")
                    logging.info(df[col].value_counts().nlargest(5))
                    logging.info(f"Unique values in {col} (first 10): {df[col].unique()[:10]}")

        context_cols_to_check = ['make', 'model', 'fuel_type', 'event_type', 'source_file']
        logging.info("\n--- Unique Values for Context Columns ---")
        for col in context_cols_to_check:
            if col in df.columns:
                logging.info(f"Unique values in '{col}' (first 10): {df[col].unique()[:10]}")
            else:
                logging.info(f"Context column '{col}' not found.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    inspect_volvo_columns(DATA_PATH) 