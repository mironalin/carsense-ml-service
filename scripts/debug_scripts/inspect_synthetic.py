import pandas as pd
import numpy as np
import sys # Keep sys just in case

SYNTHETIC_FILE = 'data/synthetic_dtc_samples/synthetic_P0121_samples.parquet' # Inspect P0121
ORIGINAL_FILE = 'data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet'

def check_empty_dtc(x):
    """Check if the DTC entry (list, array, nan) represents an empty set."""
    if isinstance(x, list):
        return len(x) == 0
    elif isinstance(x, np.ndarray):
        # Check if it's an empty array
        return x.size == 0
    elif pd.isna(x): # Handle NaN or None
         return True
    else: # Treat other types (like scalar strings if they exist) as non-empty
         return False

print(f"--- Inspecting Synthetic Data: {SYNTHETIC_FILE} ---")
try:
    df_synthetic = pd.read_parquet(SYNTHETIC_FILE)
    print(f"Shape: {df_synthetic.shape}")
    # print(f"Columns: {df_synthetic.columns.tolist()}") # Maybe too verbose
    print("\nRelevant Columns Head:")
    # Base relevant columns + potential new ones
    relevant_cols = ["absolute_timestamp", "ENGINE_COOLANT_TEMP", "SPEED", "ENGINE_RPM",
                     "MAF", "ENGINE_LOAD", "INTAKE_MANIFOLD_PRESSURE",
                     "SHORT TERM FUEL TRIM BANK 1", "SHORT TERM FUEL TRIM BANK 2", "LONG TERM FUEL TRIM BANK 2",
                     "THROTTLE_POS", "generated_dtc"]
    existing_cols_synthetic = [col for col in relevant_cols if col in df_synthetic.columns]
    if existing_cols_synthetic:
        print(df_synthetic[existing_cols_synthetic].head())
    else:
        print("Relevant columns not found in synthetic data.")

    print("\nRelevant Columns Description:")
    # Base desc cols + potential new ones
    desc_cols = ["ENGINE_COOLANT_TEMP", "SPEED", "ENGINE_RPM", "MAF", "ENGINE_LOAD", "INTAKE_MANIFOLD_PRESSURE",
                 "SHORT TERM FUEL TRIM BANK 1", "SHORT TERM FUEL TRIM BANK 2", "LONG TERM FUEL TRIM BANK 2",
                 "THROTTLE_POS"]
    existing_desc_cols_synthetic = [col for col in desc_cols if col in df_synthetic.columns]
    if existing_desc_cols_synthetic:
        print(df_synthetic[existing_desc_cols_synthetic].describe())
    else:
        print("Description columns not found in synthetic data.")

    print("\nGenerated DTCs:")
    if 'generated_dtc' in df_synthetic.columns:
        # Handle potential list or scalar values in 'generated_dtc'
        unique_dtcs = df_synthetic['generated_dtc'].apply(lambda x: tuple(x) if isinstance(x, list) else x).astype(str).unique()
        print(unique_dtcs.tolist())
    else:
        print("'generated_dtc' column not found.")

except FileNotFoundError:
    print(f"Error: Synthetic data file not found at {SYNTHETIC_FILE}")
except Exception as e:
    print(f"Error reading or processing synthetic data: {e}")


print(f"\n--- Inspecting Original Data (Normal Segments): {ORIGINAL_FILE} ---")
try:
    df_original = pd.read_parquet(ORIGINAL_FILE)
    # Filter out rows with existing DTCs for a fairer comparison baseline
    # Use the robust check_empty_dtc function
    try:
        if 'parsed_dtcs' in df_original.columns:
             is_normal = df_original['parsed_dtcs'].apply(check_empty_dtc)
        else:
             print("Warning: 'parsed_dtcs' column missing. Showing stats for all original data.")
             is_normal = pd.Series(True, index=df_original.index)
    except Exception as e: # Catch any unexpected error during apply
        print(f"Warning: Error applying check_empty_dtc: {e}. Showing stats for all original data.")
        is_normal = pd.Series(True, index=df_original.index)

    df_original_normal = df_original[is_normal].copy()

    print(f"Shape (Normal): {df_original_normal.shape}")

    print("\nRelevant Columns Head (Normal):")
    # Base relevant cols + potential new ones
    relevant_cols_original = ['absolute_timestamp', 'ENGINE_COOLANT_TEMP', 'SPEED', 'ENGINE_RPM',
                              'MAF', 'ENGINE_LOAD', 'INTAKE_MANIFOLD_PRESSURE',
                              'SHORT TERM FUEL TRIM BANK 1', 'SHORT TERM FUEL TRIM BANK 2', 'LONG TERM FUEL TRIM BANK 2',
                              'THROTTLE_POS', 'parsed_dtcs']
    existing_cols_original = [col for col in relevant_cols_original if col in df_original_normal.columns]
    if existing_cols_original:
        # Show more rows to see variation
        print(df_original_normal[existing_cols_original].head(10))
    else:
        print("Relevant columns not found in original data.")


    print("\nRelevant Columns Description (Normal):")
    # Base desc cols + potential new ones
    desc_cols_original = ['ENGINE_COOLANT_TEMP', 'SPEED', 'ENGINE_RPM', 'MAF', 'ENGINE_LOAD', 'INTAKE_MANIFOLD_PRESSURE',
                        'SHORT TERM FUEL TRIM BANK 1', 'SHORT TERM FUEL TRIM BANK 2', 'LONG TERM FUEL TRIM BANK 2',
                        'THROTTLE_POS']
    existing_desc_cols_original = [col for col in desc_cols_original if col in df_original_normal.columns]
    if existing_desc_cols_original:
        print(df_original_normal[existing_desc_cols_original].describe())
    else:
        print("Relevant description columns not found in original data.")


except FileNotFoundError:
    print(f"Error: Original data file not found at {ORIGINAL_FILE}")
except Exception as e:
    print(f"Error reading or processing original data: {e}", file=sys.stderr)