import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
INPUT_PARQUET_PATH = os.path.join(project_root, "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet")
COLUMN_TO_INSPECT = "ENGINE_RUNTIME"
NUM_UNIQUE_TO_SHOW = 50 # Limit output for brevity

def inspect_column(file_path: str, column_name: str):
    logging.info(f"Inspecting column '{column_name}' in file: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    try:
        # Read only the specific column to save memory/time
        df = pd.read_parquet(file_path, columns=[column_name])
        logging.info(f"Successfully read column '{column_name}'. Total rows: {len(df)}")

        col_dtype = df[column_name].dtype
        logging.info(f"Data type of column '{column_name}': {col_dtype}")

        unique_values = df[column_name].unique()
        num_unique = len(unique_values)
        logging.info(f"Number of unique values in '{column_name}': {num_unique}")

        logging.info(f"Showing first {min(num_unique, NUM_UNIQUE_TO_SHOW)} unique values:")
        # Convert to list for consistent slicing and printing
        unique_list = list(unique_values)
        for i, val in enumerate(unique_list[:NUM_UNIQUE_TO_SHOW]):
            logging.info(f"  [{i}]: {val} (Type: {type(val)})")

        if num_unique > NUM_UNIQUE_TO_SHOW:
            logging.info(f"... (omitting remaining {num_unique - NUM_UNIQUE_TO_SHOW} unique values)")

        # Check for NaNs/None
        nan_count = df[column_name].isnull().sum()
        logging.info(f"Number of NaN/None values in '{column_name}': {nan_count}")

        # Attempt numeric conversion to see what fails
        numeric_col = pd.to_numeric(df[column_name], errors='coerce')
        coerced_nan_count = numeric_col.isnull().sum()
        if coerced_nan_count > nan_count:
             logging.info(f"Attempting numeric conversion resulted in {coerced_nan_count} NaNs (originally {nan_count}).")
             # Find some examples that failed conversion
             failed_values = df[column_name][numeric_col.isnull() & df[column_name].notnull()].unique()
             num_failed_unique = len(failed_values)
             logging.info(f"Showing first {min(num_failed_unique, NUM_UNIQUE_TO_SHOW)} unique values that failed numeric conversion:")
             failed_list = list(failed_values)
             for i, val in enumerate(failed_list[:NUM_UNIQUE_TO_SHOW]):
                 logging.info(f"  [{i}]: {val} (Type: {type(val)})")
             if num_failed_unique > NUM_UNIQUE_TO_SHOW:
                 logging.info(f"... (omitting remaining {num_failed_unique - NUM_UNIQUE_TO_SHOW} failed unique values)")
        elif coerced_nan_count == nan_count:
             logging.info("Attempting numeric conversion did not introduce additional NaNs.")

    except KeyError:
        logging.error(f"Column '{column_name}' not found in the Parquet file.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_column(INPUT_PARQUET_PATH, COLUMN_TO_INSPECT) 