import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

# Adjust import path to access app modules if running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# We might not need imports from app.preprocessing.data_cleaning if we implement directly
# from app.preprocessing.data_cleaning import apply_scaling # apply_scaling from data_cleaning.py also handles NaNs before scaling. We'll do it more explicitly.

def identify_derived_features(df_columns: list) -> list:
    '''Identifies columns that are lag, difference, or rolling window features.'''
    derived_feature_patterns = ['_lag_', '_diff_', '_rol_']
    derived_cols = [col for col in df_columns if any(pattern in col for pattern in derived_feature_patterns)]
    print(f"Identified {len(derived_cols)} potential derived features for imputation.")
    return derived_cols

def finalize_features(input_file: str, output_file: str):
    '''
    Loads an aggregated feature file, imputes NaNs in derived features,
    applies global scaling, and saves the result.
    '''
    print(f"--- Starting Final Feature Processing for: {input_file} ---")

    try:
        df = pd.read_parquet(input_file)
        print(f"Successfully loaded {input_file}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading Parquet file {input_file}: {e}")
        return

    # 1. Impute NaNs in Derived Features (lag, diff, rolling window features)
    print("\n--- Step 1: Imputing NaNs in Derived Features ---")
    derived_feature_columns = identify_derived_features(df.columns.tolist())

    if not derived_feature_columns:
        print("No derived feature columns identified for imputation. Skipping imputation.")
    else:
        group_col = 'source_file'
        if group_col not in df.columns:
            print(f"Error: Grouping column '{group_col}' not found. Cannot perform grouped imputation.")
            # Potentially fall back to a global imputation or skip, but for these features, grouped is key.
            # For now, we'll assume 'source_file' is present as per our pipeline.
        else:
            print(f"Imputing NaNs for {len(derived_feature_columns)} columns, grouped by '{group_col}' using bfill then fillna(0).")
            for col in derived_feature_columns:
                if col in df.columns:
                    # Ensure the column is numeric before attempting numeric operations, though they should be
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df.groupby(group_col)[col].bfill()
                        df[col] = df[col].fillna(0) # Fallback for any full-group NaNs
                        # print(f"  Imputed: {col}") # Verbose
                    else:
                        print(f"  Warning: Column {col} is not numeric. Skipping imputation for it.")
                else:
                    print(f"  Warning: Expected derived column {col} not found in DataFrame. Skipping.")
            print("Finished imputing NaNs in derived features.")

    # 2. Identify Columns for Global Scaling
    print("\n--- Step 2: Identifying Columns for Global Scaling ---")

    # Columns to exclude from scaling
    # 'source_file' is critical, 'absolute_timestamp' is datetime.
    # Original 'TIME_SEC' if it exists and isn't considered a feature.
    # Metadata columns that are objects/categories.
    cols_to_exclude_from_scaling = ['source_file', 'absolute_timestamp', 'TIME_SEC']

    # Add any non-numeric metadata columns that might have been added
    metadata_object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_exclude_from_scaling.extend(metadata_object_cols)
    cols_to_exclude_from_scaling = list(set(cols_to_exclude_from_scaling)) # Ensure uniqueness

    numerical_cols_for_scaling = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols_for_scaling = [col for col in numerical_cols_for_scaling if col not in cols_to_exclude_from_scaling]

    if not numerical_cols_for_scaling:
        print("No numerical columns identified for scaling. Skipping scaling step.")
    else:
        print(f"Identified {len(numerical_cols_for_scaling)} numerical columns for global scaling.")
        # print(f"Columns to be scaled: {numerical_cols_for_scaling}") # Very verbose

        # 3. Apply Global StandardScaler
        print("\n--- Step 3: Applying Global StandardScaler ---")
        scaler = StandardScaler()

        # Fit and transform
        # Important: Scaler operates on a NumPy array. Store original index and columns.
        df_scaled_values = scaler.fit_transform(df[numerical_cols_for_scaling])

        # Create a new DataFrame for scaled values to preserve dtypes and handle column assignment safely
        df_scaled = pd.DataFrame(df_scaled_values, index=df.index, columns=numerical_cols_for_scaling)

        # Update the original DataFrame with scaled values
        for col in numerical_cols_for_scaling:
            df[col] = df_scaled[col]

        print("Finished applying Global StandardScaler.")

    # 4. Save Output
    print("\n--- Step 4: Saving Finalized DataFrame ---")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"Successfully saved finalized Parquet file: {output_file}")
        print(f"Final DataFrame shape: {df.shape}")
        print(f"Columns in final DataFrame: {df.columns.tolist()}")

    except Exception as e:
        print(f"Error saving finalized Parquet file {output_file}: {e}")

    print(f"--- Final Feature Processing Complete for: {input_file} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finalizes aggregated features by imputing NaNs in derived time-series features and applying global scaling."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the aggregated Parquet file (e.g., data/features/dataset_aggregated.parquet)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the finalized Parquet file (e.g., data/model_input/dataset_final.parquet)."
    )

    args = parser.parse_args()
    finalize_features(args.input_file, args.output_file)