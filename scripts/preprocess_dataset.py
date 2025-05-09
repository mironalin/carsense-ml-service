import argparse
import os
import pandas as pd
import sys
import re
from typing import Optional
from datetime import datetime, timedelta

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
        get_vehicle_metadata,
        get_volvo_v40_static_metadata,
        VehicleMetadata,
        add_time_features
    )
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly or script is run from project root.")
    sys.exit(1)


def parse_volvo_filename(filename: str) -> dict:
    """Extracts mode, from, to, and description from Volvo V40 filenames."""
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

    static_vehicle_meta: Optional[VehicleMetadata] = None
    if args.dataset_type == "volvo_v40":
        readme_car_file = os.path.join(args.input_dir, "README-car.md")
        static_vehicle_meta = get_volvo_v40_static_metadata(readme_car_file)
        if static_vehicle_meta:
            print(f"Successfully parsed static Volvo V40 metadata: {static_vehicle_meta.model_dump(exclude_none=True)}")
        else:
            print("Warning: Could not load static Volvo V40 metadata. Static fields will be missing.")
    elif args.dataset_type == "romanian":
        static_vehicle_meta = get_vehicle_metadata()
        print(f"Using default static metadata for Romanian dataset: {static_vehicle_meta.model_dump(exclude_none=True)}")

    for filename in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, filename)
        df = None
        event_type = None
        file_metadata = {}
        current_file_vehicle_meta: Optional[VehicleMetadata] = None
        start_datetime: Optional[datetime] = None

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

            # --- Start: Added for Romanian datetime parsing ---
            date_match = re.search(r'_(\d{8})[-_]', filename)
            if date_match:
                date_str = date_match.group(1)
                try:
                    start_datetime = datetime.strptime(date_str, '%Y%m%d')
                    print(f"Parsed start datetime for {filename}: {start_datetime}")
                except ValueError as e:
                    print(f"Warning: Could not parse date from Romanian filename {filename}: {e}")
            else:
                print(f"Warning: Could not find date pattern in Romanian filename {filename}. Cannot create absolute_timestamp.")
            # --- End: Added for Romanian datetime parsing ---

            event_type = "normal_behavior"
            if is_intervention:
                match = re.search(r'intervention-[a-zA-Z0-9-]+(?=\.csv)', filename)
                event_type = match.group(0) if match else "intervention_unknown"
            df["event_type"] = event_type

            current_file_vehicle_meta = VehicleMetadata(
                make=static_vehicle_meta.make if static_vehicle_meta else "Volkswagen",
                model=static_vehicle_meta.model if static_vehicle_meta else "Passat",
                fuel_type=static_vehicle_meta.fuel_type if static_vehicle_meta else "Diesel",
                event_type=event_type
            )

        elif args.dataset_type == "volvo_v40":
            if not filename.endswith('.csv'):
                continue

            print(f"\n--- Processing Volvo V40 file: {filename} ---")
            df = load_volvo_v40_csv(file_path, pid_mapping=PID_COLUMN_MAPPING_VOLVO_V40)
            if df is None: continue

            # --- Start: Added for Volvo datetime parsing ---
            datetime_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})', filename)
            if datetime_match:
                datetime_str = datetime_match.group(1)
                try:
                    start_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H-%M-%S')
                    print(f"Parsed start datetime for {filename}: {start_datetime}")
                except ValueError as e:
                    print(f"Warning: Could not parse datetime from Volvo filename {filename}: {e}")
            else:
                print(f"Warning: Could not find datetime pattern in Volvo filename {filename}. Cannot create absolute_timestamp.")
            # --- End: Added for Volvo datetime parsing ---

            parsed_name_meta = parse_volvo_filename(filename)
            volvo_meta_fields = {}
            if static_vehicle_meta:
                volvo_meta_fields.update(static_vehicle_meta.model_dump(exclude_none=True))
            volvo_meta_fields['drive_mode'] = parsed_name_meta.get("mode")
            volvo_meta_fields['from_location'] = parsed_name_meta.get("from_loc")
            volvo_meta_fields['to_location'] = parsed_name_meta.get("to_loc")
            volvo_meta_fields['file_description'] = parsed_name_meta.get("description")
            current_file_vehicle_meta = VehicleMetadata(**volvo_meta_fields)

        else:
            print(f"Error: Unknown dataset_type '{args.dataset_type}'. Skipping.")
            continue

        if df is None or df.empty:
            print(f"No data loaded for {filename}. Skipping further processing.")
            continue

        # --- Start: Create absolute_timestamp column ---
        if start_datetime and 'TIME_SEC' in df.columns:
            try:
                # Ensure TIME_SEC is numeric and handle potential errors
                df['TIME_SEC'] = pd.to_numeric(df['TIME_SEC'], errors='coerce')
                # Drop rows where TIME_SEC could not be coerced to numeric, if any, or handle as NaN
                # For simplicity here, we'll proceed, and NaNs in TIME_SEC will result in NaT in absolute_timestamp
                df['absolute_timestamp'] = start_datetime + pd.to_timedelta(df['TIME_SEC'], unit='s', errors='coerce')
                if df['absolute_timestamp'].isnull().any():
                    print(f"Warning: Some 'absolute_timestamp' values are NaT for {filename} due to invalid 'TIME_SEC' values.")
                print(f"Successfully created 'absolute_timestamp' for {filename}.")
            except Exception as e:
                print(f"Error creating 'absolute_timestamp' for {filename}: {e}")
        elif not start_datetime:
            print(f"Skipping 'absolute_timestamp' creation for {filename} due to missing start_datetime.")
        elif 'TIME_SEC' not in df.columns:
            print(f"Skipping 'absolute_timestamp' creation for {filename} because 'TIME_SEC' column is missing.")
        # --- End: Create absolute_timestamp column ---

        # --- Start: Add time-based features ---
        if 'absolute_timestamp' in df.columns: # Check if it was successfully created
            df = add_time_features(df, timestamp_col='absolute_timestamp')
        else:
            print(f"Skipping time feature extraction for {filename} as 'absolute_timestamp' column is not available.")
        # --- End: Add time-based features ---

        print(f"Loaded {filename} with shape: {df.shape}. Columns: {df.columns.tolist()}")

        if current_file_vehicle_meta:
            meta_dict = current_file_vehicle_meta.model_dump(exclude_none=True)
            print(f"Adding metadata for {filename}: {meta_dict}")
            for meta_key, meta_val in meta_dict.items():
                if meta_val is not None:
                    if meta_key in df.columns and meta_key != "event_type" and meta_key != "drive_mode":
                        print(f"Warning: Metadata key '{meta_key}' clashes with existing PID column. Skipping adding this metadata column.")
                    else:
                        df[meta_key] = meta_val

        metadata_cols_to_exclude = list(meta_dict.keys()) if current_file_vehicle_meta else []
        cols_to_exclude_from_numeric_ops = ["TIME_SEC"] + metadata_cols_to_exclude
        if 'absolute_timestamp' in df.columns:
            cols_to_exclude_from_numeric_ops.append('absolute_timestamp')
        # Add new time features to exclusion list if they exist
        for col_name in ['hour', 'dayofweek', 'is_weekend']:
            if col_name in df.columns:
                cols_to_exclude_from_numeric_ops.append(col_name)

        print("\nReporting and handling missing values...")
        report_missing_values(df)
        cols_for_imputation = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in cols_to_exclude_from_numeric_ops]
        df = handle_missing_values(df, strategy='median', columns=cols_for_imputation)

        default_rolling_window = 3
        print(f"\nApplying rolling mean with window_size={default_rolling_window}...")
        cols_for_rolling = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in cols_to_exclude_from_numeric_ops]
        df = apply_rolling_mean(df, columns=cols_for_rolling, window_size=default_rolling_window)

        print("\nHandling outliers with IQR (cap strategy)...")
        cols_for_outliers = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in cols_to_exclude_from_numeric_ops]
        df = handle_outliers_iqr(df, columns=cols_for_outliers, strategy='cap')

        print("\nApplying StandardScaler...")
        cols_for_scaling = [col for col in df.select_dtypes(include=['number']).columns
                                if col not in cols_to_exclude_from_numeric_ops]
        df = apply_scaling(df, columns=cols_for_scaling, scaler_type='standard')

        base_filename = os.path.splitext(filename)[0]
        # Ensure consistent output format regardless of dataset type
        output_filename = f"{base_filename}_processed.parquet"
        output_file_path = os.path.join(args.output_dir, output_filename)
        try:
            df.to_parquet(output_file_path, index=False, engine='pyarrow')
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
                        help="Directory to save processed files.")

    args = parser.parse_args()
    main(args)