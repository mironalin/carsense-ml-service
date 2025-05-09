import argparse
import os
import pandas as pd
import sys
import re

# Adjust import paths based on project structure and how this script is run
# Assuming this script is in carsense-ml-service/scripts/ and app is a top-level directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from app.data_loader import load_romanian_dataset_csv, load_volvo_v40_csv
    from app.preprocessing.data_cleaning import (
        report_missing_values, handle_missing_values,
        apply_rolling_mean, apply_scaling, handle_outliers_iqr
    )
    from app.preprocessing.feature_engineering import (
        RELEVANT_PIDS, PID_COLUMN_MAPPING, PID_COLUMN_MAPPING_VOLVO_V40,
        get_vehicle_metadata
    )
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly or script is run from project root.")
    sys.exit(1)

def parse_volvo_filename(filename: str) -> dict:
    """Extracts mode, from, to, and description from Volvo V40 filenames."""
    # Format: YYYY-MM-DD hh-MM-ss_[mode]-[from]-[to]-[description].csv
    # The part after timestamp is optional.
    metadata = {"mode": None, "from_loc": None, "to_loc": None, "description": None}
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}_(.*)\.csv', filename)
    if match:
        parts_str = match.group(1)
        parts = parts_str.split('-')
        if len(parts) > 0: metadata["mode"] = parts[0] if parts[0] else None
        if len(parts) > 1: metadata["from_loc"] = parts[1] if parts[1] else None
        if len(parts) > 2: metadata["to_loc"] = parts[2] if parts[2] else None
        if len(parts) > 3: metadata["description"] = '-'.join(parts[3:]) if parts[3:] else None
    return metadata

def main(args):
    print(f"Starting preprocessing for dataset type: {args.dataset_type}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, filename)
        df = None
        event_type = None # For Romanian dataset style
        file_metadata = {} # For Volvo style filename metadata

        if args.dataset_type == "romanian":
            is_normal_behavior = 'normal-behavior' in filename or 'normal_behavior' in filename or \
                                (len(filename.split('_')) > 1 and 'normal' in filename.split('_')[1])
            is_intervention = 'intervention' in filename
            if not (filename.endswith('.csv') and (is_normal_behavior or is_intervention)):
                if filename.endswith('.csv'): print(f"Skipping unmatched Romanian CSV: {filename}")
                continue

            print(f"\n--- Processing Romanian file: {filename} ---")
            df = load_romanian_dataset_csv(file_path, pid_mapping=PID_COLUMN_MAPPING)
            if df is None: continue

            event_type = "normal_behavior"
            if is_intervention:
                match = re.search(r'intervention-[a-zA-Z0-9-]+(?=\.csv)', filename)
                event_type = match.group(0) if match else "intervention_unknown"
            df["event_type"] = event_type

        elif args.dataset_type == "volvo_v40":
            if not filename.endswith('.csv'):
                continue # Skip non-CSV files like READMEs

            print(f"\n--- Processing Volvo V40 file: {filename} ---")
            df = load_volvo_v40_csv(file_path, pid_mapping=PID_COLUMN_MAPPING_VOLVO_V40)
            if df is None: continue

            file_metadata = parse_volvo_filename(filename)
            df["drive_mode"] = file_metadata.get("mode")
            # We could add from_loc, to_loc, description as columns if needed
            # For now, drive_mode is similar to event_type

        else:
            print(f"Error: Unknown dataset_type '{args.dataset_type}'. Skipping.")
            continue

        if df is None or df.empty:
            print(f"No data loaded for {filename}. Skipping further processing.")
            continue

        print(f"Loaded {filename} with shape: {df.shape}")

        # --- Generic Cleaning and Preprocessing Steps ---
        print("\nReporting and handling missing values...")
        report_missing_values(df) # Prints report
        # For numeric columns that are not TIME_SEC or event_type/drive_mode like
        cols_for_imputation = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in ["TIME_SEC", "event_type", "drive_mode"]]
        df = handle_missing_values(df, strategy='median', columns=cols_for_imputation)

        default_rolling_window = 3
        print(f"\nApplying rolling mean with window_size={default_rolling_window}...")
        cols_for_rolling = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in ["TIME_SEC", "event_type", "drive_mode"]]
        df = apply_rolling_mean(df, columns=cols_for_rolling, window_size=default_rolling_window)

        print("\nHandling outliers with IQR (cap strategy)...")
        cols_for_outliers = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in ["TIME_SEC", "event_type", "drive_mode"]]
        df = handle_outliers_iqr(df, columns=cols_for_outliers, strategy='cap')

        print("\nApplying StandardScaler...")
        cols_for_scaling = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in ["TIME_SEC", "event_type", "drive_mode"]]
        df = apply_scaling(df, columns=cols_for_scaling, scaler_type='standard')

        # --- Save Processed File ---
        base_filename = os.path.splitext(filename)[0]
        output_filename = f"{base_filename}_processed.csv"
        output_file_path = os.path.join(args.output_dir, output_filename)
        try:
            df.to_csv(output_file_path, index=False)
            print(f"Successfully saved processed file: {output_file_path}")
        except Exception as e:
            print(f"Error saving processed file {output_file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess automotive sensor data.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["romanian", "volvo_v40"],
                        help="Type of the dataset to process (e.g., 'romanian', 'volvo_v40')")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw dataset CSV files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed CSV files.")

    args = parser.parse_args()
    main(args)