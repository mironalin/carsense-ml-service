import pandas as pd
import os
import glob
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# TARGET_CORE_PIDS from the combination script for reference
# TARGET_CORE_PIDS = [
#     "ENGINE_RPM", "ENGINE_COOLANT_TEMP", "INTAKE_AIR_TEMP",
#     "THROTTLE_POS", "VEHICLE_SPEED", "ENGINE_LOAD"
# ]

RAW_DATA_SOURCES = {
    "volvo": {
        "path_pattern": os.path.join(project_root, "data/datasets/data_volvo_v40/*.csv"),
        "pid_map": { # Actual pivoted column name -> Standard Name
            "Engine RPM": "ENGINE_RPM",
            "Engine coolant temperature": "ENGINE_COOLANT_TEMP",
            "Intake air temperature": "INTAKE_AIR_TEMP",
            "Throttle position": "THROTTLE_POS", # Note: The raw data might have 'Throttle position B' etc.
                                                # We are using the most generic one if available after pivot.
            "Vehicle speed": "VEHICLE_SPEED",
            "Calculated engine load value": "ENGINE_LOAD",
        },
        "csv_read_params": {'delimiter': ';'}
    },
    "romanian": {
        "path_pattern": os.path.join(project_root, "data/datasets/romanian_driving_ds/dataset/*.csv"),
        "pid_map": { # Exact, verbose names from Romanian CSVs -> Standard Name
            "Engine RPM (RPM)": "ENGINE_RPM", # Note: Stripped leading space
            "Engine coolant temperature (°C)": "ENGINE_COOLANT_TEMP", # Stripped leading space, also handling the other variant later
            "Intake air temperature (°C)": "INTAKE_AIR_TEMP", # Stripped leading space
            "Absolute throttle position B (%)": "THROTTLE_POS", # Stripped leading space
            "Vehicle speed (MPH)": "VEHICLE_SPEED",
            "Calculated load value (%)": "ENGINE_LOAD",
        },
        "csv_read_params": {} # Comma delimited by default
    },
    "kaggle": {
        "path_pattern": os.path.join(project_root, "data/datasets/kaggle_dtc_dataset/exp1_14drivers_14cars_dailyRoutes.csv"),
        "pid_map": { # Original Raw Name -> Standard Name
            'ENGINE_RPM': 'ENGINE_RPM',
            'ENGINE_COOLANT_TEMP': 'ENGINE_COOLANT_TEMP', # Note: Kaggle has this
            'AIR_INTAKE_TEMP': 'INTAKE_AIR_TEMP',
            'THROTTLE_POS': 'THROTTLE_POS',
            'SPEED': 'VEHICLE_SPEED',
            'ENGINE_LOAD': 'ENGINE_LOAD',
        },
        "csv_read_params": {'decimal': ',', 'low_memory': False},
        "percentage_cols": ['THROTTLE_POS', 'ENGINE_LOAD'] # Columns that might have '%' and comma decimal
    }
}

def clean_percentage_column(series: pd.Series) -> pd.Series:
    """Cleans a column that might contain percentages as strings (e.g., "75,5%")"""
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned_series = series.astype(str).str.replace('%', '', regex=False)
    cleaned_series = cleaned_series.str.replace(',', '.', regex=False)
    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
    return numeric_series

def main():
    logging.info("Starting inspection of raw dataset missingness...")

    summary_missingness = {}
    all_volvo_pivoted_columns = set()

    for name, config in RAW_DATA_SOURCES.items():
        logging.info(f"--- Processing dataset: {name} ---")
        file_paths = glob.glob(config["path_pattern"])

        if not file_paths:
            logging.warning(f"No files found for {name} with pattern {config['path_pattern']}. Skipping.")
            continue

        all_dfs_for_source = []
        for file_path in file_paths:
            try:
                df_single = pd.read_csv(file_path, **config["csv_read_params"])
                df_single.columns = [col.strip() for col in df_single.columns] # General stripping for all

                if name == "volvo":
                    # Volvo data is semicolon delimited. read_csv with delimiter=';' should parse columns like: "SECONDS", "PID", "VALUE", "UNITS"
                    # The quotes are part of the column names in the file header.
                    # We need to remove these quotes from the column names after loading.
                    df_single.columns = [col.replace('"', '') for col in df_single.columns]

                    col_seconds = 'SECONDS'
                    col_pid = 'PID'
                    col_value = 'VALUE'

                    if col_seconds in df_single.columns and col_pid in df_single.columns and col_value in df_single.columns:
                        logging.info(f"Pivoting Volvo data from file: {file_path}")
                        df_single[col_value] = pd.to_numeric(df_single[col_value], errors='coerce')
                        df_pivot = df_single.pivot_table(index=col_seconds,
                                                         columns=col_pid,
                                                         values=col_value,
                                                         aggfunc='first')
                        df_pivot.reset_index(inplace=True)
                        df_single = df_pivot
                        logging.info(f"Pivoted Volvo file {file_path}. New shape: {df_single.shape}, Columns: {df_single.columns.tolist()}")
                        if name == "volvo":
                            all_volvo_pivoted_columns.update(df_single.columns.tolist())
                    else:
                        logging.warning(f"Could not pivot Volvo file {file_path}. Key columns for pivot (SECONDS, PID, VALUE) not all found after cleaning. Columns present: {df_single.columns.tolist()}")

                all_dfs_for_source.append(df_single)
            except Exception as e:
                logging.error(f"Error loading raw file {file_path}: {e}", exc_info=True)
                continue

        if not all_dfs_for_source:
            logging.warning(f"No dataframes loaded for {name}. Skipping.")
            continue

        # Log all unique column names found in Volvo data after pivoting and before concatenation
        if name == "volvo" and all_volvo_pivoted_columns:
            logging.info(f"--- Unique column names found across all pivoted Volvo files ---")
            sorted_volvo_columns = sorted(list(all_volvo_pivoted_columns))
            for col_name in sorted_volvo_columns:
                logging.info(f"  Volvo Pivoted Column: {col_name}")
            logging.info(f"--- Total unique Volvo pivoted columns: {len(sorted_volvo_columns)} ---")

        df_source = pd.concat(all_dfs_for_source, ignore_index=True)
        logging.info(f"Loaded {name} with shape {df_source.shape}. Original columns after potential pivot: {df_source.columns.tolist()}")

        dataset_missing_stats = {}
        for raw_pid_name, standard_pid_name in config["pid_map"].items():
            # For Romanian, handle cases where multiple raw names map to one standard PID (e.g. coolant temp variants)
            # We take the first one found or the one with most data.
            # For simplicity here, just check if any of the mapped raw names are present.
            # A more robust way would be to coalesce them if multiple variants exist.

            actual_raw_pid_name_to_check = raw_pid_name
            if name == "romanian" and standard_pid_name == "ENGINE_COOLANT_TEMP":
                variant1 = "Engine coolant temperature (°C)" # Already stripped via df_single.columns.strip()
                # The pid_map for Romanian already has the two variants listed,
                # so this specific check might be redundant if pid_map is used carefully.
                # However, to be safe, we ensure we are checking against the known variants.
                if variant1 in df_source.columns:
                    actual_raw_pid_name_to_check = variant1
                # If variant1 is not found, actual_raw_pid_name_to_check remains the one from pid_map,
                # which could be the other variant or the first one listed if both map to same standard name.

            if actual_raw_pid_name_to_check in df_source.columns:
                pid_series = df_source[actual_raw_pid_name_to_check]

                if name == "kaggle" and actual_raw_pid_name_to_check in config.get("percentage_cols", []):
                    logging.info(f"Cleaning percentage-like column for Kaggle: {actual_raw_pid_name_to_check}")
                    pid_series = clean_percentage_column(pid_series.copy())

                if not pd.api.types.is_numeric_dtype(pid_series):
                    original_dtype = pid_series.dtype
                    pid_series_numeric = pd.to_numeric(pid_series, errors='coerce')
                    if pid_series_numeric.isnull().all() and not pid_series.isnull().all():
                         logging.warning(f"Column '{actual_raw_pid_name_to_check}' in {name} (dtype {original_dtype}) became all NaNs after to_numeric. Original had non-NaNs. Check data.")
                    pid_series = pid_series_numeric

                if pid_series.empty:
                     missing_percentage = 100.0
                     missing_count = 0 # Or len(pid_series) if it's truly empty with 0 rows
                     total_count = 0
                else:
                    missing_count = pid_series.isnull().sum()
                    total_count = len(pid_series)
                    missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0

                logging.info(f"  PID: {actual_raw_pid_name_to_check} (maps to {standard_pid_name}) - Missing: {missing_percentage:.2f}% ({missing_count}/{total_count})")
                dataset_missing_stats[standard_pid_name] = missing_percentage
            else:
                logging.warning(f"  Raw PID column '{actual_raw_pid_name_to_check}' not found in {name} dataset. Reporting as 100% missing.")
                dataset_missing_stats[standard_pid_name] = 100.0

        summary_missingness[name] = dataset_missing_stats

    logging.info("\n--- Summary of Raw PID Missingness (%) ---")
    summary_df = pd.DataFrame.from_dict(summary_missingness, orient='index')
    cols_order = ["ENGINE_RPM", "ENGINE_COOLANT_TEMP", "INTAKE_AIR_TEMP", "THROTTLE_POS", "VEHICLE_SPEED", "ENGINE_LOAD"]
    # Ensure all expected columns are present in the summary_df, adding them with NaN if not found, before reindexing
    for col in cols_order:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    summary_df = summary_df.reindex(columns=cols_order)

    print("\nRaw Dataset PID Missingness Report (%):")
    print(summary_df.to_string(float_format="%.2f"))

    logging.info("\n--- Comparison with Combined Dataset ---")
    logging.info("Combined (from previous logs): ENGINE_RPM: 17.53%, ENGINE_COOLANT_TEMP: 65.08%, INTAKE_AIR_TEMP: 71.58%, THROTTLE_POS: 20.03%, VEHICLE_SPEED: ~0.00%, ENGINE_LOAD: 68.00%")

if __name__ == "__main__":
    main()
