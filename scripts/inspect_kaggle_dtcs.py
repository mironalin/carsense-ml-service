import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def inspect_dtcs(file_path: str, dtc_column: str = 'TROUBLE_CODES'):
    """
    Reads a CSV file and prints the unique non-empty values from the specified DTC column.
    Handles potential quote characters and comma decimal separators.
    """
    logging.info(f"Reading file: {file_path}")
    try:
        # Attempt to read, handling potential quote issues and comma decimals
        df = pd.read_csv(
            file_path,
            low_memory=False,
            encoding='utf-8', # Try common encodings if default fails
            # encoding='latin-1',
            # decimal=',', # If numbers use comma as decimal separator
            # quotechar='"', # Explicitly set quote character
            # escapechar='\\' # Handle potential escaped quotes
        )
        logging.info(f"Successfully read {len(df)} rows.")

        # Instead of checking DTCs, just print the columns
        logging.info("Available columns in the CSV:")
        print(df.columns.tolist())
        return # Exit after printing columns

        # --- Original DTC inspection logic (commented out) ---
        # if dtc_column not in df.columns:
        #     logging.error(f"Column '{dtc_column}' not found in the CSV.")
        #     logging.info(f"Available columns: {df.columns.tolist()}")
        #     return
        # ... rest of DTC logic ...

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        logging.error("Try adjusting encoding, delimiter, quotechar, or handling bad lines.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect unique DTC codes in a Kaggle dataset CSV.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    # Remove the DTC column argument as we are just listing columns now
    # parser.add_argument('--dtc-column', type=str, default='TROUBLE_CODES', help='Name of the column containing DTCs.')

    args = parser.parse_args()

    # Call the function without the dtc_column argument
    inspect_dtcs(args.file_path) 