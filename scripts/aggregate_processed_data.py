import argparse
import os
import pandas as pd
import glob

def aggregate_processed_data(input_dir: str, output_file: str):
    """
    Aggregates all processed Parquet files from an input directory into a single Parquet file.

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

    all_dataframes = []
    for i, file_path in enumerate(processed_files):
        print(f"Loading file {i+1}/{len(processed_files)}: {file_path}...")
        try:
            df = pd.read_parquet(file_path)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping this file.")
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