import json
import os
import sys

# Global variable to hold the loaded DTC data (acts as a simple cache)
_dtc_data_cache = None

def load_dtc_data(dtc_file_path: str = "dtc.json") -> dict:
    """
    Loads the DTC data from the specified JSON file into a dictionary keyed by DTC code.

    Args:
        dtc_file_path (str): Path to the DTC JSON file.

    Returns:
        dict: A dictionary where keys are DTC codes (str) and values are the
              corresponding DTC objects (dict), or None if loading fails.

    Raises:
        FileNotFoundError: If the dtc_file_path does not exist.
        MemoryError: If the file is too large to fit into memory (potentially).
        json.JSONDecodeError: If the file is not valid JSON.
    """
    global _dtc_data_cache
    if _dtc_data_cache is not None:
        print("Using cached DTC data.")
        return _dtc_data_cache

    print(f"Attempting to load DTC data from: {dtc_file_path}")
    if not os.path.exists(dtc_file_path):
        raise FileNotFoundError(f"DTC file not found at: {dtc_file_path}")

    dtc_map = {}
    try:
        with open(dtc_file_path, 'r') as f:
            # Attempt to load the entire JSON array
            # For very large files, this might cause MemoryError
            # TODO: Consider streaming parser (e.g., ijson) if memory becomes an issue
            data = json.load(f)

            if not isinstance(data, list):
                print(f"Warning: Expected a JSON list in {dtc_file_path}, but got {type(data)}.")
                return None # Or raise an error

            for item in data:
                if isinstance(item, dict) and 'code' in item:
                    dtc_map[item['code']] = item
                else:
                    print(f"Warning: Skipping invalid item in DTC data: {item}")

        _dtc_data_cache = dtc_map # Cache the loaded data
        print(f"Successfully loaded and mapped {len(dtc_map)} DTC codes.")
        return dtc_map

    except MemoryError as me:
        print(f"Error: MemoryError loading {dtc_file_path}. File might be too large.", file=sys.stderr)
        raise me # Re-raise the MemoryError
    except json.JSONDecodeError as jde:
        print(f"Error: Invalid JSON in {dtc_file_path}. {jde}", file=sys.stderr)
        raise jde # Re-raise the decode error
    except Exception as e:
        print(f"An unexpected error occurred during DTC loading: {e}", file=sys.stderr)
        raise e # Re-raise other exceptions

def get_dtc_description(dtc_code: str, dtc_data: dict = None) -> dict | None:
    """
    Looks up the description object for a given DTC code.

    Args:
        dtc_code (str): The DTC code to look up (e.g., "P0118").
        dtc_data (dict, optional): The pre-loaded DTC data map. If None,
                                   it attempts to load data using load_dtc_data().

    Returns:
        dict | None: The dictionary object for the DTC code if found, otherwise None.
    """
    if dtc_data is None:
        try:
            dtc_data = load_dtc_data() # Use default path "dtc.json"
        except Exception:
            print("Failed to load DTC data for lookup.", file=sys.stderr)
            return None

    if dtc_data is None:
        return None

    return dtc_data.get(dtc_code)

# Example Usage:
if __name__ == "__main__":
    print("Testing DTC Lookup...")
    try:
        # Load the data first (can take time/memory)
        loaded_data = load_dtc_data()

        if loaded_data:
            # Test some codes
            test_codes = ["P0101", "P0300", "U0073", "B0001", "C0035", "INVALID_CODE"]
            for code in test_codes:
                description = get_dtc_description(code, loaded_data)
                if description:
                    # Print only relevant fields if they exist
                    fault = description.get('fault', 'N/A')
                    desc = description.get('description', 'N/A')
                    print(f"  {code}: Fault: {fault} | Description: {desc}")
                else:
                    print(f"  {code}: Not found.")
        else:
            print("DTC data could not be loaded.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except MemoryError:
        print("Error: Could not load DTC file due to memory constraints.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}") 