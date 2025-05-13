import pandas as pd
import os
import glob
import logging
import argparse
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define PIDs to select from Toyota dataset and their target names
# Based on toyota_etios_obd/README.md and observed CSV column names
TOYOTA_PID_MAP_PREPROCESS = {
    'ENGINE_RPM ()': 'ENGINE_RPM',
    'VEHICLE_SPEED ()': 'VEHICLE_SPEED',
    'THROTTLE ()': 'THROTTLE_POS', 
    'ENGINE_LOAD ()': 'ENGINE_LOAD', 
    'COOLANT_TEMPERATURE ()': 'ENGINE_COOLANT_TEMP',
    'INTAKE_AIR_TEMPERATURE ()': 'INTAKE_AIR_TEMP', # README uses INTAKE_AIR_TEMPERATURE, files might use INTAKE_AIR_TEMP ()
    'ENGINE_RUN_TIME ()': 'TIME_SEC' # README uses ENGINE_RUN_TIME, files might use ENGINE_RUN_TINE () or ENGINE_RUN_TIME ()
}

# Check for variations like INTAKE_AIR_TEMP vs INTAKE_AIR_TEMPERATURE and ENGINE_RUN_TINE vs ENGINE_RUN_TIME
# The script reads columns like: 'ENGINE_RUN_TINE ()', 'INTAKE_AIR_TEMP ()'
# So, we need to be precise with the keys based on file contents.

# Corrected map based on logs (example: 'ENGINE_RUN_TINE ()', 'INTAKE_AIR_TEMP ()')
TOYOTA_PID_MAP_PREPROCESS = {
    'ENGINE_RPM ()': 'ENGINE_RPM',
    'VEHICLE_SPEED ()': 'VEHICLE_SPEED',
    'THROTTLE ()': 'THROTTLE_POS',
    'ENGINE_LOAD ()': 'ENGINE_LOAD',
    'COOLANT_TEMPERATURE ()': 'ENGINE_COOLANT_TEMP',
    'INTAKE_AIR_TEMP ()': 'INTAKE_AIR_TEMP', # Log showed 'INTAKE_AIR_TEMP ()'
    'ENGINE_RUN_TINE ()': 'TIME_SEC'  # Log showed 'ENGINE_RUN_TINE ()'
}

# All PIDs listed in the README for potential future use or fuller dataset:
# ALL_TOYOTA_PIDS_FROM_README = [
#     "ENGINE_RUN_TIME", "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE", "ENGINE_LOAD",
#     "COOLANT_TEMPERATURE", "LONG_TERM_FUEL_TRIM_BANK_1", "SHORT_TERM_FUEL_TRIM_BANK_1",
#     "INTAKE_MANIFOLD_PRESSURE", "FUEL_TANK_LEVEL_INPUT", "ABSOLUTE_THROTTLE_B",
#     "ACCELERATOR_PEDAL_POSITION_D", "ACCELERATOR_PEDAL_POSITION_E",
#     "COMMANDED_THROTTLE_ACTUATOR", "FUEL_AIR_COMMANDED_EQUIV_RATIO",
#     "ABSOLUTE_BAROMETRIC_PRESSURE", "RELATIVE_THROTTLE_POSITION",
#     "INTAKE_AIR_TEMPERATURE", "TIMING_ADVANCE", "CATALYST_TEMPERATURE_BANK1_SENSOR1",
#     "CATALYST_TEMPERATURE_BANK1_SENSOR2", "CONTROL_MODULE_VOLTAGE",
#     "COMMANDED_EVAPORATIVE_PURGE", "TIME_RUN_WITH_MIL_ON",
#     "TIME_SINCE_TROUBLE_CODES_CLEARED", "DISTANCE_TRAVELED_WITH_MIL_ON",
#     "WARM_UPS_SINCE_CODES_CLEARED"
# ]

def preprocess_toyota_file(input_file_path: str, output_file: str):
    """
    Reads a single Toyota Etios raw CSV file, selects specific PIDs, renames them,
    calculates absolute timestamp, and saves to a temporary Parquet file.
    Handles potential trailing comma issues by explicitly using header names.
    """
    logging.info(f"Processing Toyota file: {input_file_path}")
    try:
        # Step 1: Read the header row to get the exact column names
        headers_df = pd.read_csv(input_file_path, nrows=0, skipinitialspace=True)
        header_names = [col.strip() for col in headers_df.columns]
        logging.debug(f"Detected headers: {header_names}")
        if not header_names:
            logging.warning(f"Could not read headers from {input_file_path}. Skipping file.")
            return None

        # Step 2: Read the data using the detected headers and usecols to handle potential extra columns
        df = pd.read_csv(input_file_path, header=0, names=header_names, usecols=header_names, 
                         low_memory=False, skipinitialspace=True, on_bad_lines='warn')
        # Original columns before rename: ['ENGINE_RUN_TINE ()', 'ENGINE_RPM ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()', 'COOLANT_TEMPERATURE ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()', 'SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'INTAKE_MANIFOLD_PRESSURE ()', 'FUEL_TANK ()', 'ABSOLUTE_THROTTLE_B ()', 'PEDAL_D ()', 'PEDAL_E ()', 'COMMANDED_THROTTLE_ACTUATOR ()', 'FUEL_AIR_COMMANDED_EQUIV_RATIO ()', 'ABSOLUTE_BAROMETRIC_PRESSURE ()', 'RELATIVE_THROTTLE_POSITION ()', 'INTAKE_AIR_TEMP ()', 'TIMING_ADVANCE ()', 'CATALYST_TEMPERATURE_BANK1_SENSOR1 ()', 'CATALYST_TEMPERATURE_BANK1_SENSOR2 ()', 'CONTROL_MODULE_VOLTAGE ()', 'COMMANDED_EVAPORATIVE_PURGE ()', 'TIME_RUN_WITH_MIL_ON ()', 'TIME_SINCE_TROUBLE_CODES_CLEARED ()', 'DISTANCE_TRAVELED_WITH_MIL_ON ()', 'WARM_UPS_SINCE_CODES_CLEARED ()']
        
        # Check if essential raw columns exist based on TOYOTA_PID_MAP_PREPROCESS keys
        required_raw_cols = list(TOYOTA_PID_MAP_PREPROCESS.keys())
        missing_req_cols = [col for col in required_raw_cols if col not in df.columns]
        if missing_req_cols:
            logging.warning(f"File {input_file_path} is missing required raw columns: {missing_req_cols}. Available: {df.columns.tolist()}. Skipping file.")
            return None

        # Select and rename columns based on the map
        df_selected = df[required_raw_cols].copy()
        df_selected.rename(columns=TOYOTA_PID_MAP_PREPROCESS, inplace=True)

        # Drop duplicate rows based on all selected columns, keeping the first occurrence
        initial_rows = len(df_selected)
        df_selected.drop_duplicates(subset=df_selected.columns.tolist(), keep='first', inplace=True)
        dropped_rows = initial_rows - len(df_selected)
        if dropped_rows > 0:
            logging.info(f"Dropped {dropped_rows} duplicate rows from {os.path.basename(input_file_path)}.")

        # Convert selected PID columns to numeric, coercing errors
        numeric_cols = [col for col in df_selected.columns if col != 'TIME_SEC'] # Exclude time column for now
        for col in numeric_cols:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

        # Convert TIME_SEC separately
        df_selected['TIME_SEC'] = pd.to_numeric(df_selected['TIME_SEC'], errors='coerce')

        # Report missing percentages *after* numeric conversion
        logging.info(f"Missing values analysis for {os.path.basename(input_file_path)} (after selection/rename/coerce):")
        missing_percentages = (df_selected.isnull().sum() * 100 / len(df_selected)).round(2)
        for col, percentage in missing_percentages.items():
            if percentage > 0:
                logging.info(f"  PID {col} has {percentage}% missing values")
            else:
                 logging.debug(f"  PID {col} has 0.00% missing values") # Only log non-zero

        # Calculate absolute timestamp
        # Extract date from filename (e.g., drive1_20230115.csv -> 2023-01-15)
        # This part needs refinement based on actual filenames if they contain dates
        base_filename = os.path.basename(input_file_path)
        # Assuming filename format like 'drive1.csv' or 'abel_obd.csv' - no date info here.
        # If date info *is* available elsewhere (e.g., file modification time, or metadata file),
        # it should be incorporated here.
        # For now, we'll have to proceed without absolute date, which limits time-series analysis.
        # If TIME_SEC represents seconds from start of drive, it's still useful.
        # We need an 'absolute_timestamp' column for consistency, even if it's relative for now.

        if 'TIME_SEC' in df_selected.columns and not df_selected['TIME_SEC'].isnull().all():
            # Use a dummy start date if no real date is available
            # WARNING: This timestamp will NOT be accurate in absolute terms!
            # It assumes TIME_SEC is seconds from the start of the drive.
            start_timestamp = pd.Timestamp('2020-01-01') # Arbitrary start
            df_selected['absolute_timestamp'] = start_timestamp + pd.to_timedelta(df_selected['TIME_SEC'], unit='s', errors='coerce')
            missing_abs_time = df_selected['absolute_timestamp'].isnull().sum()
            if missing_abs_time > 0:
                 logging.warning(f"Could not calculate absolute_timestamp for {missing_abs_time} rows due to invalid TIME_SEC values.")
        else:
            logging.warning(f"Cannot create absolute_timestamp as 'TIME_SEC' is missing or all null.")
            df_selected['absolute_timestamp'] = pd.NaT # Add column with Not-a-Time

        # Add context columns (assuming a single vehicle for this dataset for now)
        df_selected['make'] = 'Toyota'
        df_selected['model'] = 'Etios'
        df_selected['year'] = 2014
        df_selected['fuel_type'] = 'Gasoline' # Based on README
        df_selected['source_file'] = base_filename # Keep track of origin

        logging.info(f"Processed {os.path.basename(input_file_path)}. Shape: {df_selected.shape}")
        return df_selected

    except Exception as e:
        logging.error(f"Error processing file {input_file_path}: {e}", exc_info=True)
        return None

def main(input_dir: str, output_file: str):
    logging.info(f"Starting Toyota dataset preprocessing.")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output file: {output_file}")

    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        logging.error(f"No CSV files found in {input_dir}")
        sys.exit(1)

    logging.info(f"Found {len(all_files)} CSV files to process.")

    all_dfs = []
    for file_path in all_files:
        processed_df = preprocess_toyota_file(file_path, output_file) # Pass output_file for logging context maybe?
        if processed_df is not None and not processed_df.empty:
            all_dfs.append(processed_df)

    if not all_dfs:
        logging.error("No dataframes were successfully processed. Exiting.")
        sys.exit(1)

    logging.info(f"Concatenating {len(all_dfs)} processed dataframes...")
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Final check for missing core PIDs in the combined dataframe
    final_core_pids = [
        'ENGINE_RPM', 'ENGINE_COOLANT_TEMP', 'INTAKE_AIR_TEMP',
        'THROTTLE_POS', 'VEHICLE_SPEED', 'ENGINE_LOAD'
    ]
    logging.info("Final combined dataset missing values analysis:")
    final_missing = (final_df.isnull().sum() * 100 / len(final_df)).round(2)
    for col in final_core_pids + ['TIME_SEC', 'absolute_timestamp']:
        if col in final_df.columns:
            percentage = final_missing.get(col, 100.0) # Default to 100% if somehow missing after concat
            if percentage > 0:
                logging.info(f"  Column '{col}' has {percentage}% missing values.")
            else:
                logging.info(f"  Column '{col}' has 0.00% missing values.")
        else:
            logging.warning(f"  Column '{col}' is missing from the final dataframe!")


    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the final dataframe
    try:
        final_df.to_parquet(output_file, index=False)
        logging.info(f"Successfully saved combined Toyota data to {output_file}. Final shape: {final_df.shape}")
    except Exception as e:
        logging.error(f"Failed to save final parquet file: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Toyota Etios OBD raw CSV data.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the raw Toyota CSV files.")
    parser.add_argument("--output_file", required=True, help="Path to save the final processed Parquet file.")

    args = parser.parse_args()

    main(args.input_dir, args.output_file)