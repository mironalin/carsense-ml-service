import pandas as pd
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Script is in scripts/debug_scripts/

# Input file (renamed Romanian data for generic model)
DATA_PATH = os.path.join(project_root, "data/model_input/romanian_renamed_for_generic_tier1.parquet")
# --- End Configuration ---

def inspect_throttle_data(data_path: str):
    """
    Loads the specified Parquet file, filters for VW Passat Diesel,
    and prints statistics for the THROTTLE_POS column.
    """
    logging.info(f"Attempting to load data from: {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Error: Input file not found at {data_path}")
        return

    try:
        df = pd.read_parquet(data_path)
        logging.info(f"Successfully loaded data. Shape: {df.shape}")
        logging.info(f"Available columns: {df.columns.tolist()}")

        # Filter for VW Passat Diesel
        # Ensure column names here match those in the Parquet file
        vw_filter = df['make'].astype(str).str.contains('Volkswagen', case=False, na=False)
        passat_filter = df['model'].astype(str).str.contains('Passat', case=False, na=False)
        diesel_filter = df['fuel_type'].astype(str).str.contains('Diesel', case=False, na=False)

        df_passat_diesel = df[vw_filter & passat_filter & diesel_filter]

        if df_passat_diesel.empty:
            logging.warning("No VW Passat Diesel data found with the specified filters (make='Volkswagen', model='Passat', fuel_type='Diesel').")
            logging.info(f"Make unique values: {df['make'].unique()}")
            logging.info(f"Model unique values (first 50): {df['model'].unique()[:50]}")
            logging.info(f"Fuel type unique values: {df['fuel_type'].unique()}")
            return

        logging.info(f"Found {len(df_passat_diesel)} rows for VW Passat Diesel.")

        if 'THROTTLE_POS' not in df_passat_diesel.columns:
            logging.error("Column 'THROTTLE_POS' not found in the DataFrame.")
            return

        logging.info("\\nDescriptive statistics for THROTTLE_POS (VW Passat Diesel):")
        # Convert to numeric if it's not already, coercing errors
        throttle_pos_series = pd.to_numeric(df_passat_diesel['THROTTLE_POS'], errors='coerce')
        logging.info(throttle_pos_series.describe())

        logging.info("\\nUnique values for THROTTLE_POS (VW Passat Diesel):")
        logging.info(throttle_pos_series.unique())

        # Check if all values are NaN after coercion
        if throttle_pos_series.isnull().all() and not df_passat_diesel['THROTTLE_POS'].isnull().all():
            logging.warning("THROTTLE_POS column for VW Passat Diesel might contain non-numeric data that couldn't be coerced.")
            logging.info(f"Original unique values before coercion: {df_passat_diesel['THROTTLE_POS'].unique()}")


    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    inspect_throttle_data(DATA_PATH) 