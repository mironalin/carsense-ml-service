RELEVANT_PIDS = [
    "ENGINE_LOAD",
    "ENGINE_RPM",
    "COOLANT_TEMPERATURE",
    "SHORT_TERM_FUEL_TRIM_BANK_1",
    "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_2",
    "LONG_TERM_FUEL_TRIM_BANK_2",
    "INTAKE_MANIFOLD_ABSOLUTE_PRESSURE",
    "INTAKE_AIR_TEMPERATURE",
    "MASS_AIR_FLOW",
    "THROTTLE_POSITION",
    "OXYGEN_SENSOR_1_VOLTAGE_BANK_1",
    "OXYGEN_SENSOR_2_VOLTAGE_BANK_1",
    "OXYGEN_SENSOR_1_VOLTAGE_BANK_2",
    "OXYGEN_SENSOR_2_VOLTAGE_BANK_2",
    "FUEL_RAIL_PRESSURE",
    "TIMING_ADVANCE",
    "CATALYST_TEMPERATURE_BANK_1_SENSOR_1",
    "CATALYST_TEMPERATURE_BANK_2_SENSOR_1",
    "CONTROL_MODULE_VOLTAGE",
    "ENGINE_FUEL_RATE",
    "VEHICLE_SPEED",
    "FUEL_LEVEL_INPUT",
    "AMBIENT_AIR_TEMPERATURE",
    "ENGINE_OIL_TEMPERATURE",
]

# TODO: Add functions for feature engineering tasks
# - Process DTCs with severity classification
# - Create metadata features from vehicle info
# - Normalize data

import json
from enum import Enum

class DTCSeverity(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"

# Initial placeholder for DTC severity mapping.
# This should be expanded with ~100-150 common DTCs and their severities.
# We can load a more comprehensive list from dtc.json and then work on classifying them,
# or use a pre-selected list of common DTCs if available.
DTC_SEVERITY_MAPPING = {
    # Example Critical DTCs (Engine/Transmission major failures, safety systems)
    "P0300": DTCSeverity.CRITICAL,  # Random/Multiple Cylinder Misfire Detected
    "P0700": DTCSeverity.CRITICAL,  # Transmission Control System Malfunction
    "U0100": DTCSeverity.CRITICAL,  # Lost Communication With ECM/PCM "A"

    # Example High Severity DTCs (Can affect drivability or cause further damage)
    "P0171": DTCSeverity.HIGH,    # System Too Lean (Bank 1)
    "P0172": DTCSeverity.HIGH,    # System Too Rich (Bank 1)
    "P0420": DTCSeverity.HIGH,    # Catalyst System Efficiency Below Threshold (Bank 1)
    "P0011": DTCSeverity.HIGH,    # "A" Camshaft Position - Timing Over-Advanced or System Performance (Bank 1)
    "P0128": DTCSeverity.HIGH,    # Coolant Thermostat (Coolant Temperature Below Thermostat Regulating Temperature)


    # Example Medium Severity DTCs (Affect performance, emissions, fuel economy)
    "P0442": DTCSeverity.MEDIUM,  # Evaporative Emission System Leak Detected (Small Leak)
    "P0135": DTCSeverity.MEDIUM,  # O2 Sensor Heater Circuit Malfunction (Bank 1 Sensor 1)
    "P0101": DTCSeverity.MEDIUM,  # Mass or Volume Air Flow Circuit Range/Performance Problem
    "P0507": DTCSeverity.MEDIUM,  # Idle Air Control System RPM Higher Than Expected

    # Example Low Severity DTCs (Minor issues, often emissions related, less impact on immediate drivability)
    "P0456": DTCSeverity.LOW,     # Evaporative Emission System Leak Detected (Very Small Leak)
    "C0000": DTCSeverity.LOW,     # Placeholder for a generic Chassis code example
}

def get_dtc_severity(dtc_code: str) -> DTCSeverity:
    """
    Classifies the severity of a given DTC code based on a predefined mapping.
    """
    return DTC_SEVERITY_MAPPING.get(dtc_code, DTCSeverity.UNKNOWN)

# We will need a function to load and select relevant DTCs from dtc.json
# For now, this is a placeholder.
def load_and_select_dtcs(dtc_file_path: str = "dtc.json", num_codes: int = 150):
    """
    Placeholder function to load DTCs from the json file.
    In a real scenario, this would involve parsing the large JSON
    and potentially using frequency data or other heuristics to select the
    most common/relevant DTCs.
    """
    # This is a simplified approach.
    # Given the file size, we would typically stream-process it or use a more robust method.
    # For now, let's assume we have a way to get a list of (code, description) tuples.
    # We'll simulate by just returning our current mapped DTCs as a list of objects
    # This part will need significant refinement based on dtc.json actual full structure
    # and how we want to select the "most common" ones.

    selected_dtcs = []
    # This is just for demonstration with the small mapped list.
    # In reality, you'd load from dtc.json and filter/select.
    try:
        with open(dtc_file_path, 'r') as f:
            # This will fail for a very large file, illustrative purpose only
            all_dtcs = json.load(f)
            # Assuming dtc.json is a list of objects like: [{"code": "P0001", "description": "..."}]
            # This selection logic is naive and needs to be improved (e.g. based on frequency)
            for i, dtc_entry in enumerate(all_dtcs):
                if i < num_codes: # Select the first num_codes as a simple example
                    selected_dtcs.append({
                        "code": dtc_entry.get("code"),
                        "description": dtc_entry.get("description", "N/A"),
                        "severity": get_dtc_severity(dtc_entry.get("code")).value
                    })
                else:
                    break
        return selected_dtcs
    except FileNotFoundError:
        print(f"Warning: DTC file {dtc_file_path} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {dtc_file_path}. Is it a valid JSON array of DTC objects?")
        # Fallback to our manually mapped DTCs if json loading fails
        # This is a fallback for the placeholder nature of this function
        for code, severity_enum in DTC_SEVERITY_MAPPING.items():
            selected_dtcs.append({
                "code": code,
                "description": "Description not available in simplified example", # Placeholder
                "severity": severity_enum.value
            })
        return selected_dtcs[:num_codes]


if __name__ == "__main__":
    # Example usage:
    print(f"Severity of P0300: {get_dtc_severity('P0300').value}")
    print(f"Severity of P0456: {get_dtc_severity('P0456').value}")
    print(f"Severity of P9999 (Unknown): {get_dtc_severity('P9999').value}")

    # Example of loading selected DTCs (using the placeholder logic)
    # common_dtcs = load_and_select_dtcs(num_codes=10)
    # print("\\nSelected Common DTCs with Severity (Example):")
    # for dtc_info in common_dtcs:
    # print(f"  Code: {dtc_info['code']}, Severity: {dtc_info['severity']}, Description: {dtc_info['description']}")

    # To properly implement load_and_select_dtcs for a large dtc.json,
    # we'd need to process it in chunks or use a library that can handle large JSON files efficiently.
    # The selection of "common" DTCs would also require a data source for DTC frequency.
    # For now, the DTC_SEVERITY_MAPPING serves as our primary source for severity.
    pass