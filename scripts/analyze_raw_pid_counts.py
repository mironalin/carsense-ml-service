import os
import argparse
import pandas as pd
from collections import defaultdict

def analyze_raw_pid_counts(input_dir: str):
    """
    Analyzes PID name occurrences and non-empty value counts directly from raw Volvo CSV files.

    Args:
        input_dir: The directory containing the raw Volvo CSV files (e.g., data/datasets/data_volvo_v40/).
    """
    pid_name_counts = defaultdict(int)
    pid_non_empty_value_counts = defaultdict(int)
    total_rows_processed = 0
    file_count = 0

    print(f"Analyzing raw PID counts in directory: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".csv"): # Ensure case-insensitivity for .csv
            file_path = os.path.join(input_dir, filename)
            print(f"Processing raw file: {file_path}...")
            try:
                df_long = pd.read_csv(file_path, delimiter=';', usecols=['PID', 'VALUE'], low_memory=False)

                if df_long.empty:
                    print(f"Warning: File {file_path} is empty or has no PID/VALUE columns. Skipping.")
                    continue

                total_rows_processed += len(df_long)

                # Count occurrences of each PID name
                for pid_name in df_long['PID'].unique():
                    if pd.notna(pid_name): # Ensure pid_name itself is not NaN
                        pid_name_counts[str(pid_name)] += len(df_long[df_long['PID'] == pid_name])

                # Count non-empty/non-NaN values for each PID
                # Group by PID and count non-NaN values in the 'VALUE' column
                non_empty_counts_for_file = df_long.groupby('PID')['VALUE'].count()

                for pid_name, count in non_empty_counts_for_file.items():
                    if pd.notna(pid_name):
                        pid_non_empty_value_counts[str(pid_name)] += count

                file_count += 1
            except pd.errors.EmptyDataError:
                print(f"Warning: File {file_path} is empty. Skipping.")
            except KeyError as ke:
                print(f"Warning: File {file_path} might be missing 'PID' or 'VALUE' column: {ke}. Skipping.")
            except Exception as e:
                print(f"Error processing raw file {file_path}: {e}. Skipping.")

    if file_count == 0:
        print("No raw CSV files found or processed in the directory.")
        return

    print(f"\n--- Raw PID Count Report (Processed {file_count} files, {total_rows_processed} total PID entries) ---")

    report_data = []
    all_observed_pids = set(pid_name_counts.keys()) | set(pid_non_empty_value_counts.keys())

    for pid in sorted(list(all_observed_pids)):
        total_occurrences = pid_name_counts.get(pid, 0)
        non_empty_values = pid_non_empty_value_counts.get(pid, 0)
        percentage_with_value = (non_empty_values / total_occurrences) * 100 if total_occurrences > 0 else 0

        report_data.append({
            "PID Name": pid,
            "Total Occurrences (Raw Logs)": total_occurrences,
            "Non-Empty Values (Raw Logs)": non_empty_values,
            "% With Value (Raw Logs)": percentage_with_value
        })

    print("\n{:<70} | {:>20} | {:>20} | {:>15}".format(
        "PID Name (from CarScanner)", "Total Occurrences", "Non-Empty Values", "% With Value"
    ))
    print("-" * 135)
    for item in report_data:
        print("{:<70} | {:>20} | {:>20} | {:>15.2f}%".format(
            item["PID Name"],
            item["Total Occurrences (Raw Logs)"],
            item["Non-Empty Values (Raw Logs)"],
            item["% With Value (Raw Logs)"]
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PID name occurrences and non-empty value counts from raw Volvo CSV files.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the raw Volvo CSV files (e.g., data/datasets/data_volvo_v40/)"
    )
    args = parser.parse_args()
    analyze_raw_pid_counts(args.input_dir)