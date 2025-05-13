import pandas as pd
import numpy as np
from joblib import load
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Script is in scripts/debug_scripts
sys.path.append(project_root)

try:
    from app.preprocessing.anomaly_detection import TIER1_CORE_PIDS
except ImportError as e:
    logging.error(f"Could not import TIER1_CORE_PIDS. Error: {e}")
    sys.exit(1)

# Paths
ROMANIAN_DATA_PATH = os.path.join(project_root, "data/model_input/romanian_renamed_raw_pids_for_generic_tier1.parquet")
COMBINED_TRAINING_DATA_PATH = os.path.join(project_root, "data/model_input/combined_raw_pids_for_tier1_training.parquet")
IMPUTER_PATH = os.path.join(project_root, "models/anomaly/tier1_combined_general_imputer.joblib")
SCALER_PATH = os.path.join(project_root, "models/anomaly/tier1_combined_general_scaler.joblib")
OUTPUT_PLOT_DIR = os.path.join(project_root, "eda_plots/romanian_vs_combined_scaled_distributions")
ROMANIAN_STATS_CSV_PATH = os.path.join(OUTPUT_PLOT_DIR, "romanian_scaled_stats.csv")
COMBINED_STATS_CSV_PATH = os.path.join(OUTPUT_PLOT_DIR, "combined_train_scaled_stats.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

def load_and_select_pids(data_path: str, pids: list, dataset_name: str) -> pd.DataFrame:
    logging.info(f"Loading {dataset_name} data from: {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"{dataset_name} data file not found: {data_path}")
        sys.exit(1)
    try:
        df = pd.read_parquet(data_path)
        logging.info(f"Successfully loaded {dataset_name} data with shape: {df.shape}")
        
        missing_pids = [pid for pid in pids if pid not in df.columns]
        if missing_pids:
            logging.error(f"{dataset_name} data is missing required PIDs: {missing_pids}. Available: {df.columns.tolist()}")
            sys.exit(1)
        return df[pids]
    except Exception as e:
        logging.error(f"Error loading or selecting PIDs from {dataset_name} data ({data_path}): {e}")
        sys.exit(1)

def apply_preprocessing(df: pd.DataFrame, imputer, scaler, pids: list) -> pd.DataFrame:
    logging.info(f"Applying imputer to PIDs: {pids}")
    try:
        df_imputed_values = imputer.transform(df)
        df_imputed = pd.DataFrame(df_imputed_values, columns=df.columns, index=df.index)
        logging.info("Imputation complete.")
    except Exception as e:
        logging.error(f"Error during imputation: {e}", exc_info=True)
        sys.exit(1)

    logging.info(f"Applying scaler to PIDs: {pids}")
    try:
        df_scaled_values = scaler.transform(df_imputed)
        df_scaled = pd.DataFrame(df_scaled_values, columns=df.columns, index=df.index)
        logging.info("Scaling complete.")
        return df_scaled
    except Exception as e:
        logging.error(f"Error during scaling: {e}", exc_info=True)
        sys.exit(1)

def main():
    logging.info("--- Starting Scaled Distribution Comparison ---")

    # 1. Load Imputer and Scaler
    logging.info(f"Loading imputer from: {IMPUTER_PATH}")
    imputer = load(IMPUTER_PATH)
    logging.info(f"Loading scaler from: {SCALER_PATH}")
    scaler = load(SCALER_PATH)
    
    # Verify scaler features
    if not hasattr(scaler, 'feature_names_in_') or set(TIER1_CORE_PIDS) != set(scaler.feature_names_in_):
        logging.error(f"Scaler features mismatch! Expected based on TIER1_CORE_PIDS: {TIER1_CORE_PIDS}, Scaler has: {getattr(scaler, 'feature_names_in_', 'N/A')}")
        # sys.exit(1) # Allow to proceed but with caution if TIER1_CORE_PIDS is the source of truth

    # 2. Load and preprocess Romanian data
    df_romanian_pids = load_and_select_pids(ROMANIAN_DATA_PATH, TIER1_CORE_PIDS, "Romanian")
    df_romanian_scaled = apply_preprocessing(df_romanian_pids.copy(), imputer, scaler, TIER1_CORE_PIDS)

    # 3. Load and preprocess Combined training data
    df_combined_pids = load_and_select_pids(COMBINED_TRAINING_DATA_PATH, TIER1_CORE_PIDS, "Combined Training")
    df_combined_scaled = apply_preprocessing(df_combined_pids.copy(), imputer, scaler, TIER1_CORE_PIDS)

    # 4. Generate Descriptive Statistics
    logging.info("Generating descriptive statistics for scaled PIDs...")
    stats_romanian = df_romanian_scaled.describe().transpose()
    stats_combined = df_combined_scaled.describe().transpose()

    # Save stats to CSV files
    try:
        stats_romanian.to_csv(ROMANIAN_STATS_CSV_PATH)
        logging.info(f"Saved Romanian scaled stats to: {ROMANIAN_STATS_CSV_PATH}")
        stats_combined.to_csv(COMBINED_STATS_CSV_PATH)
        logging.info(f"Saved Combined Training scaled stats to: {COMBINED_STATS_CSV_PATH}")
    except Exception as e:
        logging.error(f"Error saving stats to CSV: {e}")

    # 5. Plot and Save Histograms/Density Plots
    logging.info("Generating and saving comparison plots...")
    for pid in TIER1_CORE_PIDS:
        plt.figure(figsize=(12, 6))
        sns.histplot(df_combined_scaled[pid], color="skyblue", label="Combined Training Data (Scaled)", kde=True, stat="density", element="step", common_norm=False)
        sns.histplot(df_romanian_scaled[pid], color="orange", label="Romanian Data (Scaled)", kde=True, stat="density", element="step", common_norm=False)
        
        plt.title(f"Scaled Distribution Comparison for {pid}")
        plt.xlabel(f"Scaled {pid} Value")
        plt.ylabel("Density")
        plt.legend()
        
        plot_filename = os.path.join(OUTPUT_PLOT_DIR, f"scaled_distribution_{pid}.png")
        try:
            plt.savefig(plot_filename)
            logging.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logging.error(f"Error saving plot {plot_filename}: {e}")
        plt.close()

    logging.info("--- Scaled Distribution Comparison Finished ---")

if __name__ == "__main__":
    main() 