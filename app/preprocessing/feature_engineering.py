import json
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict
import pandas as pd

# Refined based on raw data occurrence analysis (threshold >= 500 raw occurrences)
RELEVANT_PIDS = [
    "TIME_SEC",  # Handled separately by data loaders
    "ENGINE_RPM",
    "COOLANT_TEMPERATURE",
    "INTAKE_MANIFOLD_ABSOLUTE_PRESSURE",
    "INTAKE_AIR_TEMPERATURE",
    "MASS_AIR_FLOW",
    "THROTTLE_POSITION",
    "FUEL_RAIL_PRESSURE",
    "TIMING_ADVANCE",
    "CONTROL_MODULE_VOLTAGE",
    "ENGINE_FUEL_RATE",
    "VEHICLE_SPEED",
    "FUEL_LEVEL_INPUT",
    "AMBIENT_AIR_TEMPERATURE",
    "BAROMETRIC_PRESSURE",
    "CALCULATED_ENGINE_LOAD_VALUE", # This is the primary load PID with decent raw counts
    "ABSOLUTE_LOAD_VALUE",
    "EGR_ERROR",
    # PIDs like fuel trims, O2 sensors, detailed catalyst/DPF/turbo/EGR PIDs were too infrequent (<500 raw occurrences)
]

# Mapping from our RELEVANT_PIDS to the column names in the Romanian Driving Dataset CSVs
# This list should also be reviewed against RELEVANT_PIDS if used for Romanian dataset.
PID_COLUMN_MAPPING = {
    "TIME_SEC": "Time (sec)",
    "ENGINE_RPM": "Engine RPM (RPM)",
    "COOLANT_TEMPERATURE": "Engine coolant temperature (°C)",
    "INTAKE_MANIFOLD_ABSOLUTE_PRESSURE": "Intake manifold absolute pressure (bar)",
    "INTAKE_AIR_TEMPERATURE": "Intake air temperature (°C)",
    "MASS_AIR_FLOW": "Mass air flow rate (g/s)",
    "THROTTLE_POSITION": "Absolute throttle position B (%)",
    "FUEL_RAIL_PRESSURE": None,
    "TIMING_ADVANCE": None,
    "CONTROL_MODULE_VOLTAGE": "Control module voltage (V)",
    "ENGINE_FUEL_RATE": "Engine Fuel Rate (g/s)",
    "VEHICLE_SPEED": "Vehicle speed (MPH)",
    "FUEL_LEVEL_INPUT": None,
    "AMBIENT_AIR_TEMPERATURE": "Ambient air temperature (°C)",
    "BAROMETRIC_PRESSURE": "Barometric pressure (bar)",
    "CALCULATED_ENGINE_LOAD_VALUE": "Calculated load value (%)",
    "ABSOLUTE_LOAD_VALUE": None,
    "EGR_ERROR": None,
}

# --- Volvo V40 (CarScanner) Specific PID Mapping ---
# Updated based on raw data occurrence threshold (>= 500) and specific CarScanner names
PID_COLUMN_MAPPING_VOLVO_V40 = {
    "TIME_SEC": "Time (sec)",  # Handled by loader

    "ENGINE_RPM": "Engine RPM",
    "COOLANT_TEMPERATURE": "Engine coolant temperature",
    "INTAKE_MANIFOLD_ABSOLUTE_PRESSURE": "Intake manifold absolute pressure",
    "INTAKE_AIR_TEMPERATURE": "Intake air temperature",
    "MASS_AIR_FLOW": "MAF air flow rate",
    "THROTTLE_POSITION": "Absolute pedal position D", # Very frequent (208k occurrences)
    "FUEL_RAIL_PRESSURE": "Fuel rail pressure (absolute)", # 561 occurrences
    "TIMING_ADVANCE": "Fuel injection timing", # 895 occurrences (was "Timing advance" with 90)
    "CONTROL_MODULE_VOLTAGE": "Control module voltage", # 1148 occurrences (alt "OBD Module Voltage" 1377)
    "ENGINE_FUEL_RATE": "Engine fuel rate", # 456k occurrences (alt "Calculated instant fuel rate" 456k)
    "VEHICLE_SPEED": "Vehicle speed",
    "FUEL_LEVEL_INPUT": "Fuel level input",
    "AMBIENT_AIR_TEMPERATURE": "Ambient air temperature",
    "BAROMETRIC_PRESSURE": "Barometric pressure",
    "CALCULATED_ENGINE_LOAD_VALUE": "Calculated engine load value", # 2.7k occurrences
    "ABSOLUTE_LOAD_VALUE": "Absolute load value", # 559 occurrences
    "EGR_ERROR": "EGR error", # 563 occurrences

    # Explicitly not including PIDs with < 500 raw occurrences, e.g.:
    # ENGINE_LOAD (generic, use CALCULATED_ENGINE_LOAD_VALUE)
    # SHORT_TERM_FUEL_TRIM_BANK_1, LONG_TERM_FUEL_TRIM_BANK_1, etc.
    # OXYGEN_SENSOR_1_VOLTAGE_BANK_1, etc.
    # CATALYST_TEMPERATURE_BANK_1_SENSOR_1, etc.
    # ENGINE_OIL_TEMPERATURE
    # EXHAUST_GAS_TEMPERATURE_B1S1, etc.
    # DPF_DELTA_PRESSURE_B1, DPF_INLET_TEMP_B1, DPF_OUTLET_TEMP_B1
    # BOOST_PRESSURE_A, TURBO_A_RPM, ACTUAL_EGR_A
    # DRIVER_DEMAND_ENGINE_TORQUE, ACTUAL_ENGINE_TORQUE
}

# TODO: Add functions for feature engineering tasks
# - Process DTCs with severity classification
# - Create metadata features from vehicle info
# - Normalize data

# --- Start: New function for time-based features ---
def add_time_features(df, timestamp_col: str = 'absolute_timestamp'):
    """
    Adds time-based features (hour, dayofweek, is_weekend) to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): Name of the column containing absolute timestamps.
                            This column should be of datetime64[ns] type or coercible.

    Returns:
        pd.DataFrame: DataFrame with added time features, or original df if error.
    """
    if timestamp_col not in df.columns:
        print(f"Warning: Timestamp column '{timestamp_col}' not found. Skipping time feature extraction.")
        return df

    # Ensure the timestamp column is in datetime format
    try:
        # Check if it's already datetime, if not, try to convert
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Check for NaT values after potential conversion
        if df[timestamp_col].isnull().all():
            print(f"Warning: Timestamp column '{timestamp_col}' contains all NaT values after conversion. Skipping time feature extraction.")
            return df
        elif df[timestamp_col].isnull().any():
            print(f"Warning: Timestamp column '{timestamp_col}' contains some NaT values. Features will be NaN for these rows.")

    except Exception as e:
        print(f"Error converting column '{timestamp_col}' to datetime: {e}. Skipping time feature extraction.")
        return df

    print(f"Extracting time features from '{timestamp_col}'...")
    df['hour'] = df[timestamp_col].dt.hour
    df['dayofweek'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    print(f"Added time features: hour, dayofweek, is_weekend")
    return df
# --- End: New function for time-based features ---

class VehicleMetadata(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None # For Volvo, this will be 'V40 D2 R-Design'
    year: Optional[int] = None # Can be derived from 'Delivery date'
    mileage_km: Optional[float] = Field(default=None, alias="Vehicle Odometer Reading (km)") # Not in Volvo README-car
    fuel_type: Optional[str] = None # e.g., Diesel
    transmission: Optional[str] = None # e.g., Manual
    power_kw: Optional[float] = None   # e.g., 88
    weight_kg: Optional[float] = None  # e.g., 1292

    # Fields from filename parsing (can be per-file)
    drive_mode: Optional[str] = None       # e.g., eco, normal, rush (from Volvo filename)
    event_type: Optional[str] = None       # e.g., normal_behavior, intervention-* (from Romanian filename)
    from_location: Optional[str] = None    # from Volvo filename
    to_location: Optional[str] = None        # from Volvo filename
    file_description: Optional[str] = None # from Volvo filename

def get_vehicle_metadata(
    # Default values based on the Romanian Driving Dataset (VW Passat, likely Diesel)
    make: str = "Volkswagen",
    model: str = "Passat",
    fuel_type: str = "Diesel",
    year: Optional[int] = None, # Year is unknown for this dataset
    csv_row: Optional[dict] = None # Allow passing a data row to extract mileage
) -> VehicleMetadata:
    """
    Creates a VehicleMetadata object.
    Uses defaults for the Romanian Driving Dataset and can extract mileage if a data row is provided.
    """
    mileage = None
    if csv_row and "Vehicle Odometer Reading (km)" in csv_row:
        try:
            mileage = float(csv_row["Vehicle Odometer Reading (km)"])
        except (ValueError, TypeError):
            mileage = None # Or handle error appropriately

    return VehicleMetadata(
        make=make,
        model=model,
        year=year,
        mileage_km=mileage,
        fuel_type=fuel_type
    )

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

# NEW function for Volvo V40 static metadata
def get_volvo_v40_static_metadata(readme_car_path: str) -> Optional[VehicleMetadata]:
    """Parses the Volvo V40 README-car.md to extract static vehicle metadata."""
    metadata_dict = {}
    try:
        with open(readme_car_path, 'r') as f:
            lines = f.readlines()

        # Expecting a markdown table format: | Specification | Value |
        #                                     | --- | --- |
        #                                     | Brand | Volvo |
        for line in lines:
            if line.strip().startswith('|') and line.count('|') == 3:
                parts = [p.strip() for p in line.strip().split('|') if p.strip()]
                if len(parts) == 2:
                    key, value = parts[0], parts[1]
                    # Basic normalization of keys
                    if key.lower() == 'brand': metadata_dict['make'] = value
                    elif key.lower() == 'type': metadata_dict['model'] = value
                    elif key.lower() == 'delivery date':
                        try:
                            # Attempt to get year, e.g., from DD-MM-YYYY
                            year_str = value.split('-')[-1]
                            metadata_dict['year'] = int(year_str)
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse year from delivery date: {value}")
                            metadata_dict['year'] = None
                    elif key.lower() == 'transmission': metadata_dict['transmission'] = value
                    elif key.lower() == 'fuel': metadata_dict['fuel_type'] = value
                    elif key.lower() == 'power':
                        try:
                            # e.g., "88 kW (120 pk)" -> extract 88
                            power_val = value.split(' ')[0]
                            metadata_dict['power_kw'] = float(power_val)
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse power_kw from: {value}")
                    elif key.lower() == 'weight':
                        try:
                            # e.g., "1292 kg" -> extract 1292
                            weight_val = value.split(' ')[0]
                            metadata_dict['weight_kg'] = float(weight_val)
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse weight_kg from: {value}")

        if not metadata_dict:
            print(f"Warning: No metadata extracted from {readme_car_path}. Is format as expected?")
            return None

        return VehicleMetadata(**metadata_dict)

    except FileNotFoundError:
        print(f"Error: Volvo README-car.md not found at {readme_car_path}")
        return None
    except Exception as e:
        print(f"Error parsing Volvo README-car.md: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    print(f"Severity of P0300: {get_dtc_severity('P0300').value}")
    print(f"Severity of P0456: {get_dtc_severity('P0456').value}")
    print(f"Severity of P9999 (Unknown): {get_dtc_severity('P9999').value}")

    # Example of getting vehicle metadata
    example_data_row = {"Vehicle Odometer Reading (km)": "150000.5"}
    metadata = get_vehicle_metadata(csv_row=example_data_row)
    print("\\nVehicle Metadata Example:")
    print(f"  Make: {metadata.make}")
    print(f"  Model: {metadata.model}")
    print(f"  Fuel Type: {metadata.fuel_type}")
    print(f"  Mileage: {metadata.mileage_km} km")

    metadata_no_row = get_vehicle_metadata()
    print("\\nVehicle Metadata Example (no data row):")
    print(f"  Make: {metadata_no_row.make}")
    print(f"  Mileage: {metadata_no_row.mileage_km}") # Will be None

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