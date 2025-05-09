import argparse
import pandas as pd
import numpy as np # For np.nan in potential future use, and consistent with other scripts

def generate_eda_summary(file_path: str):
    """
    Loads a Parquet file and prints a basic EDA summary.

    Args:
        file_path (str): Path to the aggregated Parquet file.
    """
    print(f"--- EDA Summary for: {file_path} ---\n")

    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded Parquet file.\n")
    except Exception as e:
        print(f"Error loading Parquet file {file_path}: {e}")
        return

    # 1. Print DataFrame shape
    print("--- 1. DataFrame Shape ---")
    print(df.shape)
    print("\n")

    # 2. Print DataFrame info (verbose to show all columns, dtypes, and non-null counts)
    print("--- 2. DataFrame Info ---")
    df.info(verbose=True, show_counts=True)
    print("\n")

    # 3. Print descriptive statistics for all columns
    print("--- 3. Descriptive Statistics (all columns) ---")
    # .T transposes for better readability with many columns
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df.describe(include='all').T)
    print("\n")

    # 4. Calculate and print missing value percentages
    print("--- 4. Missing Value Percentages (Sorted) ---")
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_percentages_sorted = missing_percentages.sort_values(ascending=False)
    
    with pd.option_context('display.max_rows', None):
        print(missing_percentages_sorted[missing_percentages_sorted > 0]) # Only print if there are missing values
        if missing_percentages_sorted.sum() == 0:
            print("No missing values found in the dataset.")
    print("\n")
    
    print(f"--- End of EDA Summary for: {file_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a basic EDA summary for an aggregated Parquet file.")
    parser.add_argument("input_file", type=str,
                        help="Path to the aggregated Parquet file (e.g., data/features/romanian_driving_ds_aggregated.parquet).")
    
    args = parser.parse_args()
    generate_eda_summary(args.input_file) 