#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import logging
import sys
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Add project root to sys.path to allow potential imports if needed later
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def calculate_aggregation_features(group: pd.DataFrame, pid_cols: List[str], lags: List[int], windows: List[int]) -> pd.DataFrame:
    """
    Calculates lag, difference, and rolling window features for numeric PID columns within a group (e.g., one vehicle).
    Assumes the group DataFrame is already sorted by time.
    """
    group = group.copy() # Work on a copy to avoid modifying the original slice

    for col in pid_cols:
        # Difference feature (change from previous time step)
        diff_col_name = f"{col}_diff"
        group[diff_col_name] = group[col].diff()

        # Lag features (value at previous time steps)
        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}"
            group[lag_col_name] = group[col].shift(lag)

        # Rolling window features
        for window in windows:
            rolling_obj = group[col].rolling(window=window, min_periods=1) # min_periods=1 handles start of series
            mean_col_name = f"{col}_roll_mean_{window}"
            std_col_name = f"{col}_roll_std_{window}"
            min_col_name = f"{col}_roll_min_{window}"
            max_col_name = f"{col}_roll_max_{window}"

            group[mean_col_name] = rolling_obj.mean()
            group[std_col_name] = rolling_obj.std()
            group[min_col_name] = rolling_obj.min()
            group[max_col_name] = rolling_obj.max()

    return group

def aggregate_data(input_file: str, output_file: str):
    """
    Loads a processed Kaggle data file, calculates aggregated time-series features
    grouped by vehicle, and saves the result.
    """
    filename = os.path.basename(input_file)
    logging.info(f"\n--- Starting Aggregation for: {filename} ---")

    try:
        df = pd.read_parquet(input_file)
        logging.info(f"Loaded {filename} with shape: {df.shape}. Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error loading Parquet file {input_file}: {e}")
        return

    # --- Configuration ---
    group_col = 'VEHICLE_ID'
    time_col = 'absolute_timestamp'
    dtc_col = 'parsed_dtcs'

    # Define PIDs to aggregate (typically numeric columns after preprocessing, excluding time features)
    # Let's dynamically select numeric columns, excluding known non-features/metadata
    cols_to_exclude_aggregation = [
        group_col, time_col, dtc_col,
        'hour', 'dayofweek', 'is_weekend', # Base time features
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos' # Cyclical time features
    ]
    metadata_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_exclude_aggregation.extend(metadata_cols)
    # Also exclude boolean if 'AUTOMATIC' is bool
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    cols_to_exclude_aggregation.extend(bool_cols)

    cols_to_exclude_aggregation = list(set(col for col in cols_to_exclude_aggregation if col in df.columns))

    pid_cols_to_aggregate = df.select_dtypes(include=np.number).columns.tolist()
    pid_cols_to_aggregate = [col for col in pid_cols_to_aggregate if col not in cols_to_exclude_aggregation]

    if not pid_cols_to_aggregate:
        logging.error("No numeric PID columns found to aggregate. Check preprocessing steps.")
        return
    logging.info(f"Identified {len(pid_cols_to_aggregate)} PID columns for aggregation: {pid_cols_to_aggregate}")

    # Define aggregation parameters
    lag_periods = [1, 2, 3]
    rolling_windows = [3, 5, 10]
    logging.info(f"Lag periods: {lag_periods}, Rolling windows: {rolling_windows}")

    # --- Pre-computation Checks ---
    if group_col not in df.columns:
        logging.error(f"Grouping column '{group_col}' not found in the DataFrame.")
        return
    if time_col not in df.columns:
        logging.error(f"Time column '{time_col}' not found. Cannot sort for aggregation.")
        return
    if df[time_col].isnull().any():
         logging.warning(f"Time column '{time_col}' contains NaNs. Sorting might be affected. Ensure timestamps are handled.")
         # Rows with NaT timestamps might group together, affecting diff/lag/rolling calculations at boundaries.

    # --- Sort Data ---
    logging.info(f"Sorting data by '{group_col}' and '{time_col}'...")
    # Handle NaT values during sort if necessary, place them at the beginning/end consistently
    df_sorted = df.sort_values(by=[group_col, time_col], na_position='first') # Place NaTs first within each group

    # --- Apply Aggregation ---
    logging.info("Applying aggregation features (lag, diff, rolling) group by group...")
    # Using apply can be slow on very large data, but is conceptually clear.
    # Consider dask or other optimizations if performance becomes an issue.
    df_aggregated = df_sorted.groupby(group_col, group_keys=False).apply(
        calculate_aggregation_features,
        pid_cols=pid_cols_to_aggregate,
        lags=lag_periods,
        windows=rolling_windows
    )
    logging.info(f"Aggregation complete. New shape: {df_aggregated.shape}")

    # --- Handle NaNs introduced by aggregations ---
    logging.info("Handling NaNs introduced by lag/diff/rolling features (bfill within group, then fillna(0))...")
    # Identify newly created columns
    agg_feature_cols = [col for col in df_aggregated.columns if any(k in col for k in ['_diff', '_lag_', '_roll_'])]
    if agg_feature_cols:
        # Use bfill within each group first, then fill any remaining NaNs (e.g., at start of group) with 0
        for col in agg_feature_cols:
             if col in df_aggregated.columns: # Check if column exists (it should)
                df_aggregated[col] = df_aggregated.groupby(group_col)[col].bfill()
        df_aggregated[agg_feature_cols] = df_aggregated[agg_feature_cols].fillna(0)
        logging.info(f"NaN handling complete for {len(agg_feature_cols)} aggregation features.")
    else:
        logging.warning("No aggregation feature columns found after calculation. Skipping NaN handling for them.")

    # --- Save Output ---
    output_file_path = output_file

    try:
        # Ensure output directory exists
        output_dir_for_file = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir_for_file):
            os.makedirs(output_dir_for_file)
            logging.info(f"Created output directory: {output_dir_for_file}")

        # Save the aggregated dataframe
        df_aggregated.to_parquet(output_file_path, index=False)
        logging.info(f"Successfully saved aggregated file: {output_file_path}")
    except Exception as e:
        logging.error(f"Error saving aggregated Parquet file {output_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate processed Kaggle OBD data with lag, diff, and rolling features.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the processed Parquet file (e.g., data/processed/kaggle_dtc/exp1_processed.parquet).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Full path to save the final model input Parquet file (e.g., data/model_input/kaggle_dtc/exp1_model_input.parquet).")
    # Add arguments for lags, windows if desired

    args = parser.parse_args()

    aggregate_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()