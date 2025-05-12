import argparse
import os
import subprocess
import sys

# Ensure the script uses the same Python interpreter for subprocesses
PYTHON_EXECUTABLE = sys.executable

# Define configurations for each dataset and step
DATASET_CONFIGS = {
    "romanian": {
        "preprocess": {
            "script": "scripts/preprocess_dataset.py",
            "args": [
                "--dataset_type", "romanian",
                "--input_dir", "data/datasets/romanian_driving_ds/dataset",
                "--output_dir", "data/processed/romanian_driving_ds"
            ],
            "output_check_dir": "data/processed/romanian_driving_ds" # Dir to check if step can be skipped
        },
        "aggregate": {
            "script": "scripts/aggregate_processed_data.py",
            "args": [
                "--input_dir", "data/processed/romanian_driving_ds",
                "--output_file", "data/features/romanian_driving_ds_aggregated.parquet"
            ],
            "output_check_file": "data/features/romanian_driving_ds_aggregated.parquet"
        },
        "finalize": {
            "script": "scripts/finalize_features.py",
            "args": [
                "--input_file", "data/features/romanian_driving_ds_aggregated.parquet",
                "--output_file", "data/model_input/romanian_driving_ds_final.parquet"
            ],
            "output_check_file": "data/model_input/romanian_driving_ds_final.parquet"
        }
    },
    "volvo": {
        "preprocess": {
            "script": "scripts/preprocess_dataset.py",
            "args": [
                "--dataset_type", "volvo_v40",
                "--input_dir", "data/datasets/data_volvo_v40",
                "--output_dir", "data/processed/volvo_v40_full"
            ],
            "output_check_dir": "data/processed/volvo_v40_full"
        },
        "aggregate": {
            "script": "scripts/aggregate_processed_data.py",
            "args": [
                "--input_dir", "data/processed/volvo_v40_full",
                "--output_file", "data/features/volvo_v40_full_aggregated.parquet"
            ],
            "output_check_file": "data/features/volvo_v40_full_aggregated.parquet"
        },
        "finalize": {
            "script": "scripts/finalize_features.py",
            "args": [
                "--input_file", "data/features/volvo_v40_full_aggregated.parquet",
                "--output_file", "data/model_input/volvo_v40_full_final.parquet"
            ],
            "output_check_file": "data/model_input/volvo_v40_full_final.parquet"
        }
    },
    "kaggle": {
        "preprocess": {
            "script": "scripts/preprocess_kaggle_dataset.py",
            "args": [
                "--input_path", "data/datasets/kaggle_dtc_dataset/exp1_14drivers_14cars_dailyRoutes.csv",
                "--output_dir", "data/processed/kaggle_dtc"
            ],
            "output_check_file": "data/processed/kaggle_dtc/exp1_14drivers_14cars_dailyRoutes_processed.parquet"
        },
        "aggregate": {
            "script": "scripts/aggregate_kaggle_data.py",
            "args": [
                "--input_file", "data/processed/kaggle_dtc/exp1_14drivers_14cars_dailyRoutes_processed.parquet",
                "--output_file", "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet"
            ],
            "output_check_file": "data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet"
        }
    }
}

def run_step(step_name: str, config: dict, force_run: bool = False):
    """Runs a single processing step using subprocess."""
    script_path = config["script"]
    args = config["args"]
    cmd = [PYTHON_EXECUTABLE, script_path] + args

    output_path = config.get("output_check_file") or config.get("output_check_dir")

    if not force_run and output_path:
        if os.path.isfile(output_path) or (os.path.isdir(output_path) and os.listdir(output_path)):
            print(f"--- Output for {step_name} (e.g. {output_path}) already exists. Skipping. Use --force_run_{step_name.split('_')[0]} to override. ---")
            return True # Indicates step was skipped but considered successful for flow

    print(f"--- Running {step_name}: {' '.join(cmd)} ---")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Output for {step_name}:")
        print(process.stdout)
        if process.stderr:
            print(f"Stderr for {step_name}:")
            print(process.stderr)
        print(f"--- {step_name} completed successfully. ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Error during {step_name} !!!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout:")
        print(e.stdout)
        print(f"Stderr:")
        print(e.stderr)
        print(f"--- {step_name} failed. Halting pipeline for this dataset. ---")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline for specified datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CONFIGS.keys(),
        default=list(DATASET_CONFIGS.keys()),
        help="Which datasets to process (e.g., romanian volvo). Defaults to all."
    )
    parser.add_argument(
        "--force_run_preprocess",
        action="store_true",
        help="Force run the preprocessing step even if output exists."
    )
    parser.add_argument(
        "--force_run_aggregate",
        action="store_true",
        help="Force run the aggregation step even if output exists."
    )
    parser.add_argument(
        "--force_run_finalize",
        action="store_true",
        help="Force run the finalization step even if output exists."
    )
    # Individual skip flags can be added if more granular control than force_run is needed
    # e.g. --skip_step1_preprocessing, etc. For now, force_run gives control.

    args = parser.parse_args()

    for dataset_name in args.datasets:
        print(f"\n=== Processing dataset: {dataset_name.upper()} ===")
        config = DATASET_CONFIGS[dataset_name]

        # Step 1: Preprocessing
        step_config = config.get("preprocess")
        if step_config:
            if not run_step("step1_preprocess", step_config, force_run=args.force_run_preprocess):
                continue # Skip to next dataset if this step fails
        else:
            print(f"Skipping preprocess step for {dataset_name} (not defined in config).")

        # Step 2: Aggregation
        step_config = config.get("aggregate")
        if step_config:
            if not run_step("step2_aggregate", step_config, force_run=args.force_run_aggregate):
                continue # Skip to next dataset if this step fails
        else:
            print(f"Skipping aggregate step for {dataset_name} (not defined in config).")

        # Step 3: Finalization
        step_config = config.get("finalize")
        if step_config:
            if not run_step("step3_finalize", step_config, force_run=args.force_run_finalize):
                continue # Skip to next dataset if this step fails
        else:
            print(f"Skipping finalize step for {dataset_name} (not defined in config).")

        print(f"=== Successfully completed defined steps for dataset: {dataset_name.upper()} ===")

if __name__ == "__main__":
    main()