#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import sys
import re
import logging
from typing import List, Any

# Add project root to sys.path to allow importing app modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Reuse existing components where possible
    from app.preprocessing.data_cleaning import (
        report_missing_values, handle_missing_values,
        apply_rolling_mean, handle_outliers_iqr, apply_scaling # Add others if needed
    )
    from app.preprocessing.feature_engineering import (
        add_time_features, add_cyclical_features
    )
    # Import the DTC parser specific to Kaggle data
    from app.preprocessing.kaggle_dtc_preprocessor import parse_dtc_string
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly or script is run from project root.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def preprocess_kaggle_file(input_file_path: str, output_dir: str):
    """
    Preprocesses a single raw Kaggle DTC dataset CSV file.
    - Loads data
    - Parses timestamps
    - Parses DTCs
    - Applies cleaning, feature engineering, scaling
    - Saves processed file to output directory.
    """
    filename = os.path.basename(input_file_path)
    logging.info(f"\n--- Processing Kaggle file: {filename} ---")

    try:
        # 1. Load Data (Handle comma decimals)
        df = pd.read_csv(
            input_file_path,
            low_memory=False,
            encoding='utf-8',
            decimal=',' # Specify comma as decimal separator
        )
        if df.empty:
            logging.warning(f"Skipping empty file: {filename}")
            return
        logging.info(f"Loaded {filename} with shape: {df.shape}. Columns: {df.columns.tolist()}")

        # 2. Timestamp Processing
        time_col = 'TIMESTAMP' # Corrected for exp1 file
        if time_col in df.columns:
            try:
                # Ensure the column is treated as string initially if it contains E notation
                time_series_str = df[time_col].astype(str)

                # Convert from potential scientific notation string (after handling comma decimal via read_csv)
                # to numeric (float) representation of milliseconds
                time_series_numeric = pd.to_numeric(time_series_str, errors='coerce')

                # Convert numeric milliseconds to datetime
                df['absolute_timestamp'] = pd.to_datetime(time_series_numeric, unit='ms', errors='coerce')

                if df['absolute_timestamp'].isnull().any():
                    num_failed = df['absolute_timestamp'].isnull().sum()
                    logging.warning(f"{num_failed} timestamps in {filename} could not be parsed (NaT values produced). Check source data.")
                else:
                    logging.info(f"Successfully parsed '{time_col}' into 'absolute_timestamp'.")
            except Exception as e:
                logging.error(f"Error processing '{time_col}' column in {filename}: {e}. Skipping timestamp creation.", exc_info=True)
                # Ensure column doesn't exist if parsing failed partway
                if 'absolute_timestamp' in df.columns: df = df.drop(columns=['absolute_timestamp'])
        else:
            logging.warning(f"'{time_col}' (or 'TIME') column not found in {filename}. Cannot create absolute timestamp.")

        # 3. DTC Parsing
        dtc_column = 'TROUBLE_CODES'
        parsed_dtc_col_name = 'parsed_dtcs'
        if dtc_column in df.columns:
            # Clean the column before parsing (strip quotes/whitespace)
            df[dtc_column] = df[dtc_column].astype(str).str.strip(' "')
            df[parsed_dtc_col_name] = df[dtc_column].apply(parse_dtc_string)
            logging.info(f"Parsed DTCs into '{parsed_dtc_col_name}' column.")
        else:
            logging.warning(f"'TROUBLE_CODES' column not found. Skipping DTC parsing.")
            df[parsed_dtc_col_name] = [[] for _ in range(len(df))] # Add empty list column

        # 4. Feature Selection / PID Standardization
        metadata_cols = ['VEHICLE_ID', 'MARK', 'MODEL', 'CAR_YEAR', 'ENGINE_POWER', 'AUTOMATIC', 'FUEL_TYPE']
        pid_cols_original = [
            'BAROMETRIC_PRESSURE(KPA)', # Corrected for exp1 file
            'ENGINE_COOLANT_TEMP',
            'FUEL_LEVEL',
            'ENGINE_LOAD',
            'AMBIENT_AIR_TEMP',
            'ENGINE_RPM',
            'INTAKE_MANIFOLD_PRESSURE',
            'MAF',
            'LONG TERM FUEL TRIM BANK 2',
            'AIR_INTAKE_TEMP',
            'FUEL_PRESSURE',
            'SPEED',
            'SHORT TERM FUEL TRIM BANK 2',
            'SHORT TERM FUEL TRIM BANK 1',
            'ENGINE_RUNTIME',
            'THROTTLE_POS',
            'TIMING_ADVANCE',
            'EQUIV_RATIO'
        ]
        pid_cols_present = [col for col in pid_cols_original if col in df.columns]
        missing_pids = set(pid_cols_original) - set(pid_cols_present)
        if missing_pids:
            logging.warning(f"Missing expected PID columns: {missing_pids}")

        required_cols = ['absolute_timestamp', parsed_dtc_col_name]
        cols_to_keep = required_cols + metadata_cols + pid_cols_present
        cols_to_keep = [col for col in cols_to_keep if col in df.columns] # Keep only existing

        # No renames needed now based on corrected pid_cols_original
        logging.info(f"Selecting subset of columns. Keeping {len(cols_to_keep)} columns: {cols_to_keep}")
        df = df[cols_to_keep].copy()

        # 5. Time Feature Engineering (Requires 'absolute_timestamp')
        if 'absolute_timestamp' in df.columns:
            df = add_time_features(df, timestamp_col='absolute_timestamp')
            if 'hour' in df.columns: df = add_cyclical_features(df, column_name='hour', max_value=24.0)
            if 'dayofweek' in df.columns: df = add_cyclical_features(df, column_name='dayofweek', max_value=7.0)
            logging.info("Added time and cyclical features.")
        else:
            logging.info("Skipping time feature engineering ('absolute_timestamp' missing).")

        # 6. Cleaning Steps

        # NEW: Special cleaning for columns that might have '%' and comma decimals.
        # This must happen BEFORE we try to identify numeric_cols_for_cleaning.
        cols_with_potential_percentages = [
            'FUEL_LEVEL', 'ENGINE_LOAD', 'THROTTLE_POS', 'TIMING_ADVANCE', 'EQUIV_RATIO',
            'LONG TERM FUEL TRIM BANK 2', 'SHORT TERM FUEL TRIM BANK 2', 'SHORT TERM FUEL TRIM BANK 1'
        ]
        # Ensure we only process columns that are actually present (from pid_cols_present)
        pids_to_process_for_percentage = [col for col in pid_cols_present if col in cols_with_potential_percentages]

        if pids_to_process_for_percentage:
            logging.info(f"Cleaning '%' and converting to numeric for: {pids_to_process_for_percentage}")
            for col in pids_to_process_for_percentage:
                if col in df.columns: # df has already been filtered by cols_to_keep which includes pid_cols_present
                    original_nan_count = df[col].isnull().sum()
                    
                    # Convert to string, remove '%', replace decimal comma with period, then to_numeric
                    df[col] = df[col].astype(str) 
                    df[col] = df[col].str.replace('%', '', regex=False)
                    df[col] = df[col].str.replace(',', '.', regex=False) # Convert "1,23" to "1.23"
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    current_nan_count = df[col].isnull().sum()
                    if current_nan_count > original_nan_count:
                        logging.warning(f"Column '{col}': {current_nan_count - original_nan_count} new NaNs after removing '%' and converting to numeric. Original NaNs: {original_nan_count}, Current NaNs: {current_nan_count}.")
                    elif original_nan_count == current_nan_count:
                        logging.info(f"Column '{col}' cleaned (percentage format). NaN count remained: {current_nan_count}.")
                    else: # NaNs decreased - less likely if original was string with '%'
                        logging.info(f"Column '{col}' cleaned (percentage format). NaN count changed from {original_nan_count} to {current_nan_count}.")

        # ... (logic for identifying numeric_cols_for_cleaning remains the same) ...
        # ... (Ensure numeric conversion happens correctly after read_csv with decimal=',')
        numeric_cols_for_cleaning = df[pid_cols_present].select_dtypes(include=['number']).columns.tolist()
        time_feature_cols = ['hour', 'dayofweek', 'is_weekend',
                               'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
        cols_to_exclude_cleaning = [col for col in time_feature_cols if col in df.columns]
        numeric_cols_for_cleaning = [col for col in numeric_cols_for_cleaning
                                       if col not in cols_to_exclude_cleaning]

        if numeric_cols_for_cleaning:
            logging.info(f"Applying cleaning steps to {len(numeric_cols_for_cleaning)} numeric columns...")
            # report_missing_values(df[numeric_cols_for_cleaning]) # Can be verbose
            df = handle_missing_values(df, strategy='median', columns=numeric_cols_for_cleaning)
            # Uncomment other steps as needed
            df = apply_rolling_mean(df, columns=numeric_cols_for_cleaning, window_size=3)
            df = handle_outliers_iqr(df, columns=numeric_cols_for_cleaning, strategy='cap')

            # --- Scaling (Split between MinMax and Standard) ---
            cols_for_minmax = ['FUEL_LEVEL', 'ENGINE_LOAD', 'THROTTLE_POS', 'TIMING_ADVANCE', 'EQUIV_RATIO']
            
            # Ensure that the columns in cols_for_minmax are indeed numeric now and present
            actual_cols_to_minmax_scale = [col for col in numeric_cols_for_cleaning if col in cols_for_minmax]
            actual_cols_to_standard_scale = [col for col in numeric_cols_for_cleaning if col not in cols_for_minmax]

            if actual_cols_to_minmax_scale:
                logging.info(f"Applying MinMax scaling to: {actual_cols_to_minmax_scale}")
                df = apply_scaling(df, columns=actual_cols_to_minmax_scale, scaler_type='minmax')
            
            if actual_cols_to_standard_scale:
                logging.info(f"Applying Standard scaling to: {actual_cols_to_standard_scale}")
                df = apply_scaling(df, columns=actual_cols_to_standard_scale, scaler_type='standard')
            
            logging.info("Cleaning steps applied (median imputation, rolling mean, outlier capping, mixed scaling).")
        else:
             logging.warning("No numeric columns identified for cleaning steps.")

        # 7. Save Output
        base_filename = os.path.splitext(filename)[0]
        output_filename = f"{base_filename}_processed.parquet"
        output_file_path = os.path.join(output_dir, output_filename)
        df.to_parquet(output_file_path, index=False, engine='pyarrow')
        logging.info(f"Successfully saved processed file: {output_file_path}")

    except Exception as e:
        logging.error(f"An error occurred processing {filename}: {e}", exc_info=True)

def main(args):
    logging.info(f"Starting Kaggle dataset preprocessing.")
    logging.info(f"Input directory/file: {args.input_path}")
    logging.info(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input_path):
        logging.info("Processing directory...")
        for filename in os.listdir(args.input_path):
            if filename.endswith('.csv'): # Simple check for CSV files
                file_path = os.path.join(args.input_path, filename)
                preprocess_kaggle_file(file_path, args.output_dir)
            else:
                logging.info(f"Skipping non-CSV file: {filename}")
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.csv'):
        logging.info("Processing single file...")
        preprocess_kaggle_file(args.input_path, args.output_dir)
    else:
        logging.error(f"Error: Input path is neither a directory nor a valid CSV file: {args.input_path}")
        sys.exit(1)

    logging.info("Kaggle dataset preprocessing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Kaggle DTC dataset CSV files.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory containing raw Kaggle CSV files or path to a single CSV file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed Parquet files.")
    # Add other arguments if needed (e.g., specific columns, cleaning strategies)

    args = parser.parse_args()
    main(args)