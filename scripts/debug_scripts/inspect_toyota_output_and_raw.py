import pandas as pd
import os
import logging
import sys
import numpy as np # For histogram

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# FINAL_PARQUET_PATH = os.path.join(project_root, "data/model_input/toyota_etios_raw_pids_final.parquet")
RAW_CSV_DRIVE1_PATH = os.path.join(project_root, "data/datasets/toyota_etios_obd/obdiidata/drive1.csv")
EXPECTED_RPM_HEADER = 'ENGINE_RPM ()' # As seen in raw file view

def inspect_drive1_very_raw():
    logging.info(f"--- Deep Inspecting: {RAW_CSV_DRIVE1_PATH} ---")
    if not os.path.exists(RAW_CSV_DRIVE1_PATH):
        logging.error(f"Raw CSV file not found: {RAW_CSV_DRIVE1_PATH}")
        return

    try:
        # 1. Read with no header inference to see the absolute raw cells
        logging.info("Reading CSV with header=None to see raw cell data...")
        df_very_raw = pd.read_csv(RAW_CSV_DRIVE1_PATH, header=None, low_memory=False, skipinitialspace=True)
        print("\\nFirst 5 rows, raw cells (no header inference):")
        print(df_very_raw.head())

        # From manual inspection of drive1.csv via read_file, ENGINE_RPM () is the second column (index 1)
        # Let's verify this by looking at the first row of df_very_raw
        header_row_values = df_very_raw.iloc[0].astype(str).str.strip().tolist()
        print(f"\\nDetected header row values: {header_row_values}")
        
        rpm_col_index = -1
        try:
            rpm_col_index = header_row_values.index(EXPECTED_RPM_HEADER)
            print(f"Found '{EXPECTED_RPM_HEADER}' at column index: {rpm_col_index}")
        except ValueError:
            print(f"ERROR: Expected RPM header '{EXPECTED_RPM_HEADER}' not found in first row: {header_row_values}")
            logging.error(f"Could not find expected RPM header. Raw headers: {header_row_values}")
            # Try to find it case-insensitively or with partial match if needed, but for now, exact match.
            # Fallback: try to guess from previous knowledge if the script fails here.
            # rpm_col_index = 1 # Risky guess
            # print(f"Attempting to proceed with guessed RPM column index: {rpm_col_index}")
            return # Stop if header not found as expected

        # 2. Now, re-read the CSV, but this time, tell pandas which row is the header
        # and then try to access the RPM column by its *name*.
        logging.info(f"Re-reading CSV, using row 0 as header.")
        df_with_header = pd.read_csv(RAW_CSV_DRIVE1_PATH, header=0, low_memory=False, skipinitialspace=True)
        df_with_header.columns = [col.strip() for col in df_with_header.columns]

        if EXPECTED_RPM_HEADER in df_with_header.columns:
            print(f"\\nAccessing column '{EXPECTED_RPM_HEADER}' by name after re-read with header=0:")
            rpm_series_by_name = df_with_header[EXPECTED_RPM_HEADER]
            print(f"Original dtype of series accessed by name: {rpm_series_by_name.dtype}")
            
            # Convert to numeric
            rpm_series_numeric = pd.to_numeric(rpm_series_by_name, errors='coerce')
            print("\\nAfter pd.to_numeric(errors='coerce'):")
            print(f"New dtype: {rpm_series_numeric.dtype}")
            print(f"Number of NaNs introduced: {rpm_series_numeric.isna().sum()}")
            
            print(f"\\nBasic description of '{EXPECTED_RPM_HEADER}' (numeric from named column):")
            print(rpm_series_numeric.describe(percentiles=[.01, .25, .5, .75, .95, .99]))

            print(f"\\nUnique values for '{EXPECTED_RPM_HEADER}' (numeric, first 50 sorted, excluding NaN):")
            unique_vals = sorted(rpm_series_numeric.dropna().unique())
            print(unique_vals[:50])
            if len(unique_vals) > 50:
                print(f"... and {len(unique_vals) - 50} more unique values.")
            
            test_high_values = [245.0, 245.25, 1128.0, 1000.0]
            print(f"\\nChecking for specific high values {test_high_values} in named column:")
            for val in test_high_values:
                if not rpm_series_numeric.isna().all() and rpm_series_numeric[rpm_series_numeric == val].any():
                    print(f"Value {val} FOUND. Count: {rpm_series_numeric[rpm_series_numeric == val].shape[0]}")
                else:
                    print(f"Value {val} NOT FOUND.")
        else:
            logging.error(f"Column '{EXPECTED_RPM_HEADER}' NOT FOUND in df_with_header. Columns are: {df_with_header.columns.tolist()}")

    except Exception as e:
        logging.error(f"Error during deep inspection: {e}", exc_info=True)

if __name__ == "__main__":
    inspect_drive1_very_raw() 