import os
import sys
import pandas as pd
import argparse # For command-line arguments
from typing import Optional, Set # Import Optional and Set

# Adjust import path to access app modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from app.data_loader import load_volvo_v40_csv
    # Import the specific mapping for Volvo and the generic RELEVANT_PIDS list
    from app.preprocessing.feature_engineering import RELEVANT_PIDS, PID_COLUMN_MAPPING_VOLVO_V40
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly.")
    # Define fallbacks if imports fail, so script can at least try to run for diagnosis
    RELEVANT_PIDS = []
    PID_COLUMN_MAPPING_VOLVO_V40 = {}
    if "load_volvo_v40_csv" not in globals(): # If the main function didn't load
        print("Failed to import load_volvo_v40_csv. Exiting.")
        sys.exit(1)

def load_and_pivot_volvo_csv_for_pid_discovery(file_path: str) -> Optional[pd.DataFrame]:
    """Loads a Volvo CSV, pivots it, and returns the wide DataFrame for PID name discovery."""
    try:
        df_long = pd.read_csv(file_path, delimiter=';')
        expected_cols = ["SECONDS", "PID", "VALUE"]
        if not all(col in df_long.columns for col in expected_cols):
            print(f"Error: CSV {file_path} missing expected columns. Found: {df_long.columns.tolist()}")
            return None
        try:
            df_wide = df_long.pivot(index="SECONDS", columns="PID", values="VALUE")
        except ValueError as ve:
            if "Index contains duplicate entries, cannot reshape" in str(ve):
                df_long.drop_duplicates(subset=["SECONDS", "PID"], keep='last', inplace=True)
                df_wide = df_long.pivot(index="SECONDS", columns="PID", values="VALUE")
            else:
                raise
        return df_wide.reset_index() # Keep SECONDS as a column for now, not critical for PID names
    except Exception as e:
        print(f"Error loading/pivoting {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Discover unique PIDs in Volvo V40 CarScanner CSV files within a directory.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing Volvo V40 CSV files.")
    args = parser.parse_args()

    print(f"--- Discovering PIDs in directory: {args.directory_path} ---")

    all_pid_names: Set[str] = set()
    processed_files_count = 0
    csv_files_found = 0

    for root, _, files in os.walk(args.directory_path):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_files_found += 1
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}...")
                df_pivoted = load_and_pivot_volvo_csv_for_pid_discovery(file_path)

                if df_pivoted is not None:
                    processed_files_count += 1
                    current_file_pids = {col for col in df_pivoted.columns if col not in ["SECONDS", "Time (sec)"]}
                    all_pid_names.update(current_file_pids)
                else:
                    print(f"Could not process {file_path}")

    print(f"\n--- PID Discovery Summary ---")
    print(f"Found {csv_files_found} CSV files in the directory.")
    print(f"Successfully processed {processed_files_count} CSV files.")

    if all_pid_names:
        print(f"Found {len(all_pid_names)} unique PID names across all processed files:")
        for i, name in enumerate(sorted(list(all_pid_names))):
            print(f"  {i+1}. {name}")
    else:
        print("No PIDs found or no files could be processed.")

if __name__ == "__main__":
    main()