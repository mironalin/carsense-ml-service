import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
ANOMALOUS_DATA_PATH = os.path.join(project_root, "data/processed/generic_tier1_analysis_anomalous_data.parquet")

# Columns involved in heuristics (using scaled values)
HEURISTIC_COLUMNS = [
    "ENGINE_COOLANT_TEMP",
    "TIME_SEC", # For filtering coolant analysis
    "ENGINE_RPM",
    "SPEED",
    "ENGINE_LOAD",
    "THROTTLE_POS",
    "AIR_INTAKE_TEMP"
]

def analyze_anomaly_distributions(file_path: str, columns: list[str]):
    logging.info(f"Analyzing distributions in anomalous data file: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    try:
        df_anomalous = pd.read_parquet(file_path)
        logging.info(f"Successfully loaded anomalous data. Shape: {df_anomalous.shape}")

        # Ensure the required columns exist
        available_cols = [col for col in columns if col in df_anomalous.columns]
        missing_cols = list(set(columns) - set(available_cols))
        if missing_cols:
            logging.warning(f"Columns not found in anomalous data and will be skipped: {missing_cols}")
        if not available_cols:
            logging.error("None of the specified columns for analysis were found.")
            return

        logging.info("Descriptive statistics for relevant scaled features within anomalous data:")
        # Calculate description for available columns
        desc = df_anomalous[available_cols].describe()
        logging.info("\n" + desc.to_string())

        # --- Specific Analysis for Low Coolant Threshold ---
        coolant_col = "ENGINE_COOLANT_TEMP"
        time_col = "TIME_SEC"
        runtime_threshold_sec = 120
        if coolant_col in available_cols and time_col in available_cols:
            anomalies_after_warmup = df_anomalous[
                (df_anomalous[time_col].notna()) & (df_anomalous[time_col] > runtime_threshold_sec)
            ]
            if not anomalies_after_warmup.empty:
                logging.info(f"\nDescriptive statistics for '{coolant_col}' (scaled) among anomalies after {runtime_threshold_sec}s runtime:")
                coolant_desc = anomalies_after_warmup[coolant_col].describe()
                logging.info("\n" + coolant_desc.to_string())
            else:
                logging.info(f"\nNo anomalies found after {runtime_threshold_sec}s runtime for coolant analysis.")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    analyze_anomaly_distributions(ANOMALOUS_DATA_PATH, HEURISTIC_COLUMNS) 