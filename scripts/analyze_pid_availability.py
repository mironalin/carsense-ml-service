import os
import argparse
import pandas as pd
from collections import defaultdict

def analyze_pid_availability(input_dir: str):
    """
    Analyzes PID availability across multiple processed Parquet files in a directory.

    Args:
        input_dir: The directory containing the processed Parquet files.
    """
    pid_non_null_counts = defaultdict(int)
    pid_total_counts = defaultdict(int)
    file_count = 0

    print(f"Analyzing PID availability in directory: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith("_processed.parquet"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing file: {file_path}...")
            try:
                #CHUNKSIZE = 100000  # Adjust chunk size based on memory and file size
                # Temporarily disabling chunking for simplicity, assuming files are manageable.
                # If memory errors occur, re-enable and test chunking.
                # chunks = []
                # for chunk in pd.read_csv(file_path, chunksize=CHUNKSIZE, low_memory=False):
                #     chunks.append(chunk)
                # df = pd.concat(chunks, ignore_index=True)

                df = pd.read_parquet(file_path, engine='pyarrow')

                # Exclude metadata columns from PID analysis for availability stats
                # Metadata columns are typically added at the end by the preprocessing script
                # A more robust way might be to get RELEVANT_PIDS from feature_engineering.py
                # but for this script, we'll infer PIDs as non-metadata columns.

                # Infer PIDs: columns that are not common metadata suffixes or exact names
                # This is a heuristic.
                potential_metadata_cols = {'make', 'model', 'year', 'fuel_type',
                                           'transmission', 'power_kw', 'weight_kg',
                                           'drive_mode', 'event_type', 'from_location',
                                           'to_location', 'file_description'}

                pid_columns = [col for col in df.columns if col not in potential_metadata_cols and col != 'TIME_SEC']

                for col in pid_columns:
                    pid_non_null_counts[col] += df[col].count() # count() gives non-NaN values
                    pid_total_counts[col] += len(df[col])
                file_count += 1
            except pd.errors.EmptyDataError:
                print(f"Warning: File {file_path} is empty or unreadable. Skipping.")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}. Skipping.")

    if file_count == 0:
        print("No processed Parquet files found in the directory.")
        return

    print(f"\n--- PID Availability Report (Processed {file_count} Parquet files) ---")
    # Calculate an approximate total number of rows processed for context.
    # This assumes PIDs are mostly consistent across files for the denominator of the sum.
    # A more precise total row count would be sum over all df lengths before PID selection.
    approx_total_rows_processed = 0
    if pid_total_counts:
        # Get the total rows from one of the PIDs (they should all have the same total count sum if present in all files)
        # This is just for a rough estimate display
        approx_total_rows_processed = next(iter(pid_total_counts.values()))
        # More accurately, it's the sum of individual file lengths if PIDs varied per file,
        # but pid_total_counts sums rows where each PID was *expected*.
        # For now, we'll just print based on a representative PID's total count sum or average.
        # Let's use the sum of counts for a representative PID if all PIDs were in all files.
        # Or, more simply, the sum of lengths of dataframes processed.
        # The current pid_total_counts[pid] sums len(df[col]) for each file a PID appears in.
        # So sum(pid_total_counts.values()) is complex. Let's print an average instead.

        # Sum of all row counts for each PID (can be large if many PIDs)
        # Let's consider the total rows for a PID that is likely present in all selected PIDs.
        # For instance, 'ENGINE_RPM' if it's a key PID. Or average over PIDs.
        if 'ENGINE_RPM' in pid_total_counts: # Pick a common PID
            overall_row_count_for_rpm = pid_total_counts['ENGINE_RPM']
            print(f"Total rows where 'ENGINE_RPM' was expected (sum across files): {overall_row_count_for_rpm}")
        else: # Fallback if ENGINE_RPM is not found (should not happen with current RELEVANT_PIDS)
            if len(pid_total_counts) > 0:
                avg_rows_per_pid_column = sum(pid_total_counts.values()) / len(pid_total_counts)
                print(f"Average total rows per PID column (summed across files): {avg_rows_per_pid_column:.0f}")

    availability_report = []
    for pid, total_count_for_pid in pid_total_counts.items():
        if total_count_for_pid > 0:
            non_null_count = pid_non_null_counts.get(pid, 0)
            percentage_available = (non_null_count / total_count_for_pid) * 100
            availability_report.append({
                "PID": pid,
                "Non-Null Count": non_null_count,
                "Total Rows For PID": total_count_for_pid, # Renamed for clarity
                "Percentage Available": percentage_available
            })
        else:
            availability_report.append({
                "PID": pid,
                "Non-Null Count": 0,
                "Total Rows For PID": 0,
                "Percentage Available": 0.0
            })

    # Sort by PID name for consistent reporting
    availability_report.sort(key=lambda x: x["PID"])

    print("\n{:<40} | {:>15} | {:>20} | {:>15}".format(
        "PID Name", "Non-Nulls", "Total Rows For PID", "% Available"
    ))
    print("-" * 100) # Adjusted separator length
    for report_item in availability_report:
        print("{:<40} | {:>15} | {:>20} | {:>15.2f}%" .format(
            report_item["PID"],
            report_item["Non-Null Count"],
            report_item["Total Rows For PID"],
            report_item["Percentage Available"]
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PID availability in processed Parquet files.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the processed Parquet files (e.g., data/processed/volvo_v40_full/)"
    )
    args = parser.parse_args()
    analyze_pid_availability(args.input_dir)