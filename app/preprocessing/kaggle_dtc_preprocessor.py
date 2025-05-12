#!/usr/bin/env python3

import pandas as pd
import re
import logging
import argparse
from typing import List, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def parse_dtc_string(dtc_string: Any) -> List[str]:
    """
    Parses a string potentially containing concatenated DTC codes into a list of
    standard DTC codes (Pxxxx, Cxxxx, Bxxxx, Uxxxx).

    Handles non-string inputs and empty strings gracefully.
    """
    if not isinstance(dtc_string, str) or not dtc_string:
        return []

    # Regular expression to find standard DTC codes
    # P: Powertrain, C: Chassis, B: Body, U: Network
    dtc_pattern = r'[PCBU]\d{4}'

    # Find all matches
    found_codes = re.findall(dtc_pattern, dtc_string)

    # Return the list of found codes, ensuring uniqueness just in case
    return sorted(list(set(found_codes)))

def load_and_preprocess_kaggle_dtc(file_path: str,
                                   dtc_column: str = 'TROUBLE_CODES',
                                   parsed_dtc_col_name: str = 'parsed_dtcs') -> pd.DataFrame:
    """
    Loads a Kaggle DTC dataset CSV, parses the specified DTC column,
    and returns the DataFrame with a new column containing lists of parsed DTCs.
    """
    logging.info(f"Loading and preprocessing Kaggle DTC data from: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        logging.info(f"Successfully read {len(df)} rows from {file_path}.")

        if dtc_column not in df.columns:
            logging.error(f"Column '{dtc_column}' not found in the CSV.")
            logging.info(f"Available columns: {df.columns.tolist()}")
            # Return an empty DataFrame or raise an error, depending on desired handling
            return pd.DataFrame()

        # Apply the parsing function
        df[parsed_dtc_col_name] = df[dtc_column].apply(parse_dtc_string)

        logging.info(f"Applied DTC parsing to column '{dtc_column}', new column is '{parsed_dtc_col_name}'.")
        return df

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred during loading/preprocessing: {e}")
        return pd.DataFrame()


# --- Main execution block (for testing) ---
if __name__ == "__main__":
    # Test the loading and preprocessing function
    # test_file_path = "data/datasets/kaggle_dtc_dataset/exp3_4drivers_1car_1route.csv"
    test_file_path_large = "data/datasets/kaggle_dtc_dataset/exp1_14drivers_14cars_dailyRoutes.csv"

    logging.info(f"--- Testing load_and_preprocess_kaggle_dtc with {test_file_path_large} ---")
    df_processed = load_and_preprocess_kaggle_dtc(test_file_path_large)

    if not df_processed.empty:
        logging.info(f"Processed DataFrame head (relevant columns):\n")
        dtc_related_cols = ['TROUBLE_CODES', 'parsed_dtcs']
        if 'DTC_NUMBER' in df_processed.columns:
            dtc_related_cols.insert(0, 'DTC_NUMBER')

        print(df_processed[dtc_related_cols].head(20)) # Show top 20 to see if any DTCs appear early

        logging.info("\n--- Checking rows where parsing might have occurred ---")
        parsed_rows = df_processed[df_processed['parsed_dtcs'].apply(lambda x: len(x) > 0)]
        if not parsed_rows.empty:
            logging.info(f"Found {len(parsed_rows)} rows with parsed DTCs. Examples:")
            print(parsed_rows[dtc_related_cols].head(10)) # Show first 10 parsed
            logging.info("... and a few from the tail in case DTCs are sparse:")
            print(parsed_rows[dtc_related_cols].tail(10)) # Show last 10 parsed

        else:
            logging.info("No rows with parsed DTCs found.")

        all_parsed_dtcs = set()
        for dtc_list in df_processed['parsed_dtcs']:
            for dtc in dtc_list:
                all_parsed_dtcs.add(dtc)
        if all_parsed_dtcs:
            logging.info(f"\nAll unique parsed DTCs from '{test_file_path_large}': {sorted(list(all_parsed_dtcs))}")
        else:
            logging.info(f"\nNo DTCs were parsed from the TROUBLE_CODES column in '{test_file_path_large}'.")

    # --- Previous test for parse_dtc_string (can be commented out if verbose) ---
    # logging.info("\n--- Testing parse_dtc_string function directly ---")
    # test_strings = [
    #     None,
    #     "",
    #     "P0133",
    #     "C0300",
    #     "4300",
    #     "P0078B0004P3000",
    #     "P0078B0004P3000P0078",
    #     "Some text P0128 and U0100 mixed in",
    #     "P007EP2036P18D0",
    # ]
    # for test_str in test_strings:
    #     parsed = parse_dtc_string(test_str)
    #     logging.info(f"Input: {repr(test_str)} -> Output: {parsed}")