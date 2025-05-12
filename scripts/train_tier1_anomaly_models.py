import os
import subprocess
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Define the segments for which Tier 1 models should be trained
TARGET_SEGMENTS = ["gasoline", "diesel"] # Add more as needed (e.g., "hybrid", "electric" if applicable)

# Define the base path for input data. Assumes pre-filtered Parquet files.
# Example: data/model_input/gasoline_tier1_data.parquet
INPUT_DATA_BASE_DIR = "data/model_input"
INPUT_DATA_FILENAME_TEMPLATE = "{segment}_tier1_data.parquet"

# Define the output directory for trained models and scalers
MODEL_OUTPUT_BASE_DIR = "models/anomaly"

# Define the path to the anomaly detection training script
ANOMALY_SCRIPT_PATH = "app/preprocessing/anomaly_detection.py"

# Default contamination rate (can be overridden per segment if needed)
DEFAULT_CONTAMINATION = 0.02


def train_segment_model(segment_tag: str, contamination_rate: float) -> bool:
    """Trains an anomaly detection model for a specific vehicle segment."""
    logging.info(f"--- Attempting to train Tier 1 model for segment: {segment_tag} ---")

    input_filename = INPUT_DATA_FILENAME_TEMPLATE.format(segment=segment_tag)
    data_path = os.path.join(INPUT_DATA_BASE_DIR, input_filename)

    if not os.path.exists(data_path):
        logging.error(f"Data file for segment '{segment_tag}' not found at: {data_path}. Skipping training.")
        return False

    logging.info(f"Using data file: {data_path}")
    logging.info(f"Output directory: {MODEL_OUTPUT_BASE_DIR}")
    logging.info(f"Contamination rate for {segment_tag}: {contamination_rate}")

    command = [
        "python",
        ANOMALY_SCRIPT_PATH,
        "--data-path", data_path,
        "--output-dir", MODEL_OUTPUT_BASE_DIR,
        "--fuel-type-segment", segment_tag,
        "--contamination", str(contamination_rate)
        # Add --output-csv if you want to save the anomaly df from the script, e.g.:
        # "--output-csv", os.path.join(MODEL_OUTPUT_BASE_DIR, f"{segment_tag}_anomalies_output.csv")
    ]

    try:
        logging.info(f"Executing command: {' '.join(command)}")
        # Using check=True will raise CalledProcessError if the script returns a non-zero exit code
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Successfully trained Tier 1 model for segment: {segment_tag}")
        logging.debug(f"Script STDOUT for {segment_tag}:\n{process.stdout}")
        if process.stderr:
            logging.warning(f"Script STDERR for {segment_tag} (though successful):\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error training Tier 1 model for segment: {segment_tag}")
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to train model for segment {segment_tag}: {e}")
        return False


if __name__ == "__main__":
    logging.info("Starting Tier 1 Anomaly Model Training Orchestration...")
    successful_trainings = 0
    failed_trainings = 0

    for segment in TARGET_SEGMENTS:
        # Here you could potentially define different contamination rates per segment
        # For example:
        # cont_rate = 0.015 if segment == "gasoline" else DEFAULT_CONTAMINATION
        cont_rate = DEFAULT_CONTAMINATION

        if train_segment_model(segment, cont_rate):
            successful_trainings += 1
        else:
            failed_trainings += 1

    logging.info("--- Tier 1 Anomaly Model Training Orchestration Finished ---")
    logging.info(f"Successfully trained models for {successful_trainings} segment(s).")
    if failed_trainings > 0:
        logging.warning(f"Failed to train models for {failed_trainings} segment(s). Check logs for details.")
    else:
        logging.info("All targeted segments processed successfully (if data was found).") 