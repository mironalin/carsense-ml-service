import argparse
import os
import pandas as pd
import glob
import sys # Added sys

# Adjust import path to access app modules if running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from app.preprocessing.feature_engineering import add_lag_diff_features, RELEVANT_PIDS, add_rolling_window_features
except ImportError as e:
    print(f"Error importing feature_engineering: {e}. Ensure PYTHONPATH or script context is correct.")
    sys.exit(1)

def aggregate_processed_data(input_dir: str, output_file: str):
    """
    Aggregates all processed Parquet files from an input directory into a single Parquet file,
    adding lag and difference features per file before aggregation.

    Args:
        input_dir (str): Directory containing the processed Parquet files.
        output_file (str): Path to save the aggregated Parquet file.
    """
    search_pattern = os.path.join(input_dir, "*_processed.parquet")
    processed_files = glob.glob(search_pattern)

    if not processed_files:
        print(f"No processed Parquet files found in {input_dir} matching pattern \"{search_pattern}\".")
        return

    print(f"Found {len(processed_files)} processed Parquet files to aggregate.")

    # Define PIDs for which to create lag/diff features
    # Exclude TIME_SEC as it's an index/timestamp, not a sensor reading to lag directly in this context.
    pids_for_lag_diff = [pid for pid in RELEVANT_PIDS if pid != "TIME_SEC"]
    print(f"Will attempt to generate lag/diff features for: {pids_for_lag_diff}")

    all_dataframes = []
    for i, file_path in enumerate(processed_files):
        print(f"Loading file {i+1}/{len(processed_files)}: {file_path}...")
        try:
            df = pd.read_parquet(file_path)
            
            if 'source_file' not in df.columns:
                print(f"Warning: 'source_file' column not found in {file_path}. Lag/diff features might be incorrect or skipped.")
                # Optionally, assign filename if it's missing, though it should be there from preprocess_dataset.py
                # df['source_file'] = os.path.basename(file_path)
            
            # Add lag and difference features
            print(f"Adding lag/diff features for {file_path}...")
            df = add_lag_diff_features(
                df,
                group_by_col='source_file', 
                target_cols=pids_for_lag_diff, 
                lag_periods=[1, 2], # e.g., value 1 and 2 timesteps ago
                diff_periods=[1]    # e.g., change from 1 timestep ago
            )
            
            # Add rolling window features
            print(f"Adding rolling window features for {file_path}...")
            df = add_rolling_window_features(
                df,
                group_by_col='source_file',
                target_cols=pids_for_lag_diff # Using the same PIDs as for lag/diff
                # Default window_sizes=[3, 5] and aggregations=['mean', 'std'] will be used
            )
            
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}. Skipping this file.")
            continue

    if not all_dataframes:
        print("No dataframes were successfully loaded. Aggregation cannot proceed.")
        return

    print("\nConcatenating all loaded dataframes...")
    try:
        aggregated_df = pd.concat(all_dataframes, ignore_index=True)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        # Attempt to concatenate with more robust handling for differing dtypes/columns if possible
        # For now, we'll just report the error.
        # A more advanced version might try to align columns or infer types if schemas mismatch slightly,
        # but our current pipeline should produce consistent schemas for sensor PIDs.
        print("Concatenation failed. Please check if all Parquet files have compatible schemas.")
        # You might want to inspect individual dataframes' dtypes and columns here if debugging is needed.
        # for i, df_part in enumerate(all_dataframes):
        #     print(f"\nDataFrame {i} from {processed_files[i]} info:")
        #     df_part.info()
        return

    print(f"Aggregation complete. Final DataFrame shape: {aggregated_df.shape}")
    print(f"Columns in aggregated DataFrame: {aggregated_df.columns.tolist()}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir: # If output_file includes a directory path
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    print(f"\nSaving aggregated DataFrame to: {output_file}...")
    try:
        aggregated_df.to_parquet(output_file, index=False)
        print("Successfully saved aggregated Parquet file.")
    except Exception as e:
        print(f"Error saving aggregated Parquet file {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate processed Parquet files.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the processed Parquet files (e.g., data/processed/volvo_v40_full/).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the aggregated Parquet file (e.g., data/features/volvo_v40_aggregated.parquet).")

    args = parser.parse_args()
    aggregate_processed_data(args.input_dir, args.output_file)