#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import glob
import logging
from typing import List, Set
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_ORIGINAL_DATA_PATH = "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet"
DEFAULT_SYNTHETIC_DATA_DIR = "data/synthetic_dtc_samples"
DEFAULT_OUTPUT_PATH = "data/model_input/exp1_combined_with_synthetic.parquet"

def load_parquet_files_from_dir(directory: str) -> pd.DataFrame:
    """Loads all Parquet files from a directory and concatenates them."""
    all_files = glob.glob(os.path.join(directory, "synthetic_*.parquet"))
    if not all_files:
        logging.warning(f"No synthetic parquet files found in directory: {directory}")
        return pd.DataFrame() # Return empty DataFrame if no files

    df_list = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            # Extract DTC from filename if 'generated_dtc' column missing (fallback)
            if 'generated_dtc' not in df.columns:
                dtc_code = os.path.basename(f).split('_')[1] # e.g., synthetic_P0128_samples.parquet -> P0128
                logging.warning(f"File {f} missing 'generated_dtc' column. Inferring DTC '{dtc_code}' from filename.")
                df['generated_dtc'] = [[dtc_code]] * len(df)
            df_list.append(df)
            logging.debug(f"Loaded {f}, shape: {df.shape}")
        except Exception as e:
            logging.error(f"Error loading or processing file {f}: {e}")
            continue # Skip problematic files

    if not df_list:
        logging.error("Failed to load any synthetic data files.")
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined {len(df_list)} synthetic files. Total shape: {combined_df.shape}")
    return combined_df

def ensure_list_format(dtc_col: pd.Series) -> pd.Series:
    """Ensures the DTC column contains lists of strings, handling potential single strings or lists."""
    def format_item(item):
        if isinstance(item, list):
            # Ensure elements within the list are strings
            return [str(d) for d in item if pd.notna(d)]
        elif isinstance(item, np.ndarray): # Explicitly handle numpy arrays
            if item.size == 0: # Check if it's an empty array
                return []
            else: # If it's a non-empty array, convert its elements to string list
                return [str(d) for d in item if pd.notna(d)]
        elif pd.isna(item): # Handle scalar NaN or None
             return [] # Represent NaN/None as empty list
        else: # Wrap single string in a list
             return [str(item)]
    return dtc_col.apply(format_item)


def main():
    parser = argparse.ArgumentParser(description="Combine original model input data with synthetic DTC samples.")
    parser.add_argument("--original_data_path", type=str, default=DEFAULT_ORIGINAL_DATA_PATH,
                        help="Path to the original model input Parquet file.")
    parser.add_argument("--synthetic_data_dir", type=str, default=DEFAULT_SYNTHETIC_DATA_DIR,
                        help="Directory containing the synthetic data Parquet files.")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Path to save the combined Parquet file.")

    args = parser.parse_args()

    # --- Load Data ---
    logging.info(f"Loading original data from: {args.original_data_path}")
    try:
        df_original = pd.read_parquet(args.original_data_path)
        logging.info(f"Original data loaded. Shape: {df_original.shape}")
    except FileNotFoundError:
        logging.error(f"Original data file not found: {args.original_data_path}")
        return
    except Exception as e:
        logging.error(f"Error loading original data: {e}")
        return

    logging.info(f"Loading synthetic data from directory: {args.synthetic_data_dir}")
    df_synthetic = load_parquet_files_from_dir(args.synthetic_data_dir)
    if df_synthetic.empty:
        logging.error("No synthetic data loaded. Exiting.")
        return
    logging.info(f"Synthetic data loaded. Shape: {df_synthetic.shape}")


    # --- Align Schemas ---
    logging.info("Aligning schemas...")

    # 1. Rename synthetic DTC column
    if 'generated_dtc' in df_synthetic.columns:
        df_synthetic = df_synthetic.rename(columns={'generated_dtc': 'parsed_dtcs'})
        logging.debug("Renamed 'generated_dtc' to 'parsed_dtcs' in synthetic data.")
    elif 'parsed_dtcs' not in df_synthetic.columns:
        logging.error("Synthetic data missing 'generated_dtc' or 'parsed_dtcs' column after loading.")
        return

    # 2. Ensure consistent DTC format (list of strings)
    if 'parsed_dtcs' in df_original.columns:
        df_original['parsed_dtcs'] = ensure_list_format(df_original['parsed_dtcs'])
        logging.debug("Formatted 'parsed_dtcs' in original data.")
    else:
        logging.warning("Original data missing 'parsed_dtcs' column. Cannot ensure format consistency.")
        # Optionally create an empty list column if needed downstream
        # df_original['parsed_dtcs'] = [[] for _ in range(len(df_original))]

    df_synthetic['parsed_dtcs'] = ensure_list_format(df_synthetic['parsed_dtcs'])
    logging.debug("Formatted 'parsed_dtcs' in synthetic data.")


    # 3. Identify common columns (use original data as the reference)
    original_cols: Set[str] = set(df_original.columns)
    synthetic_cols: Set[str] = set(df_synthetic.columns)

    common_cols: List[str] = sorted(list(original_cols.intersection(synthetic_cols)))
    missing_in_synthetic: Set[str] = original_cols - synthetic_cols
    extra_in_synthetic: Set[str] = synthetic_cols - original_cols

    if not common_cols:
        logging.error("No common columns found between original and synthetic data. Cannot combine.")
        return

    logging.info(f"Found {len(common_cols)} common columns.")
    if missing_in_synthetic:
        logging.warning(f"Columns in original but not synthetic (will be dropped from original or added as NaN to synthetic): {missing_in_synthetic}")
        # Decide strategy: Add as NaN to synthetic or drop from original. Let's add to synthetic for consistency.
        for col in missing_in_synthetic:
             df_synthetic[col] = pd.NA # Add missing columns with Pandas NA
        # Update common_cols to include these now-added columns
        common_cols = sorted(list(original_cols)) # Now all original cols should be in synthetic
        logging.info(f"Added missing columns ({missing_in_synthetic}) to synthetic data with NA values.")


    if extra_in_synthetic:
        logging.warning(f"Columns in synthetic but not original (will be dropped from synthetic): {extra_in_synthetic}")
        # Drop extra columns from synthetic
        df_synthetic = df_synthetic[common_cols]

    # Ensure both dataframes have columns in the same order before concat
    df_original_aligned = df_original[common_cols].copy()
    df_synthetic_aligned = df_synthetic[common_cols].copy()


    # --- Concatenate Data ---
    logging.info("Concatenating original and synthetic data...")
    df_combined = pd.concat([df_original_aligned, df_synthetic_aligned], ignore_index=True)
    logging.info(f"Combined data shape: {df_combined.shape}")


    # --- Save Combined Data ---
    logging.info(f"Saving combined data to: {args.output_path}")
    try:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        df_combined.to_parquet(args.output_path, index=False)
        logging.info("Combined data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving combined data: {e}")

if __name__ == "__main__":
    main() 