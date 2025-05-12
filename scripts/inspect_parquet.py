import pandas as pd
import sys

def inspect_parquet_file(file_path):
    """
    Reads a Parquet file and prints its schema information and the first 5 rows.

    Args:
        file_path (str): The path to the Parquet file.
    """
    try:
        print(f"Attempting to read Parquet file: {file_path}")
        df = pd.read_parquet(file_path)

        print("\\nDataFrame Info:")
        print("--------------------------------------------------")
        df.info()
        print("--------------------------------------------------")

        print("\\n\\nDataFrame Head (first 5 rows):")
        print("--------------------------------------------------")
        print(df.head().to_string())
        print("--------------------------------------------------")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading or processing the Parquet file: {e}")

if __name__ == "__main__":
    # Default file path, can be overridden by command line argument
    default_file_path = "../data/model_input/romanian_driving_ds_final.parquet" # Adjusted path relative to scripts/

    if len(sys.argv) > 1:
        file_to_inspect = sys.argv[1]
    else:
        # Attempt to construct the path relative to this script's location if no arg is given
        # This is a common pattern but might need adjustment based on execution context.
        # For simplicity, we'll keep the direct relative path from workspace root in the default.
        # A more robust way would involve os.path.dirname(__file__) etc.
        # but given the execution context, this might be simpler.
        print(f"No file path provided as a command-line argument. Using default: {default_file_path}")
        file_to_inspect = default_file_path


    # Try to make the path relative to the workspace root if it's a relative path starting with "data/" or "../data/"
    # This assumes the script is run from the workspace root or the path is already correct.
    # Given the initial default_file_path, it assumes execution from workspace root for that default.
    # If run from scripts/ folder directly, the path needs to be ../data/...
    # The default_file_path is now set to be relative from scripts/

    # Let's adjust the path to be relative from the workspace root for clarity if the script is in scripts/
    # However, the user will likely run `python scripts/inspect_parquet.py` from the root,
    # so the default path should be 'data/model_input/romanian_driving_ds_final.parquet'

    # Re-evaluating the pathing:
    # If the script is `scripts/inspect_parquet.py` and run from workspace root:
    # `python scripts/inspect_parquet.py` -> file path is 'data/model_input/...'
    # If run from `scripts/` dir:
    # `python inspect_parquet.py` -> file path is '../data/model_input/...'

    # Let's assume the user runs `python scripts/inspect_parquet.py` from the workspace root.
    # The argument parser will handle if they provide a different path.

    final_file_path = file_to_inspect
    if file_to_inspect == default_file_path and not file_to_inspect.startswith("/"): # if it's the default relative path
        # This logic is a bit complex for a simple script.
        # Let's simplify and assume the path provided is correct or the default works from `scripts/`
        pass # Use the path as is for now.

    inspect_parquet_file(final_file_path)

print("\\nTo run this script from the workspace root:")
print("python scripts/inspect_parquet.py")
print("Or, to specify a different file:")
print("python scripts/inspect_parquet.py path/to/your/file.parquet")