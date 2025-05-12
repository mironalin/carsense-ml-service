#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import logging
import sys
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Constants ---
DEFAULT_OUTPUT_DIR = "data/synthetic_dtc_samples"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Heuristic Definitions (Placeholders - Mirror logic from anomaly detection) ---

# Example: Map DTC to required PIDs and modification logic function
HEURISTIC_MAP = {
    "P0101": { # MAF Range/Performance
        "required_pids": ["MAF", "ENGINE_LOAD"],
        "modification_func": lambda df: modify_maf_load(df, direction='low')
    },
    "P0102": { # MAF Low Input
        "required_pids": ["MAF", "ENGINE_LOAD"],
        "modification_func": lambda df: modify_maf_load(df, direction='low')
    },
    "P0106": { # MAP Range/Performance
        "required_pids": ["INTAKE_MANIFOLD_PRESSURE", "ENGINE_LOAD"],
        "modification_func": lambda df: modify_map_load(df, direction='low')
    },
    "P0128": { # Coolant Thermostat (Temperature Below Regulating)
        "required_pids": ["ENGINE_COOLANT_TEMP", "SPEED", "ENGINE_RPM"],
        "modification_func": lambda df: modify_coolant_thermostat(df)
    },
    # "P0404": { # EGR Control Circuit Range/Performance - COMMENTED OUT as EGR_ERROR column is missing
    #     "required_pids": ["EGR_ERROR", "ENGINE_RPM", "ENGINE_LOAD"], # Corrected ENGINE_LOAD
    #     "modification_func": lambda df: modify_egr(df, error_type='range')
    # },

    # --- Fuel Trim Heuristics ---
    "P0171": { # System Too Lean (Bank 1) - Use available trims
        "required_pids": ["SHORT TERM FUEL TRIM BANK 1", "SHORT TERM FUEL TRIM BANK 2", "LONG TERM FUEL TRIM BANK 2"],
        "modification_func": lambda df: modify_fuel_trim(df, direction='lean')
    },
    "P0174": { # System Too Lean (Bank 2) - Use available trims
        "required_pids": ["SHORT TERM FUEL TRIM BANK 1", "SHORT TERM FUEL TRIM BANK 2", "LONG TERM FUEL TRIM BANK 2"],
        "modification_func": lambda df: modify_fuel_trim(df, direction='lean')
    },
    "P0172": { # System Too Rich (Bank 1) - Use available trims
        "required_pids": ["SHORT TERM FUEL TRIM BANK 1", "SHORT TERM FUEL TRIM BANK 2", "LONG TERM FUEL TRIM BANK 2"],
        "modification_func": lambda df: modify_fuel_trim(df, direction='rich')
    },
    "P0175": { # System Too Rich (Bank 2) - Use available trims
        "required_pids": ["SHORT TERM FUEL TRIM BANK 1", "SHORT TERM FUEL TRIM BANK 2", "LONG TERM FUEL TRIM BANK 2"],
        "modification_func": lambda df: modify_fuel_trim(df, direction='rich')
    },

    # --- Throttle Position Sensor Heuristics ---
    "P0121": { # TPS Performance/Range
        "required_pids": ["THROTTLE_POS", "ENGINE_RPM", "SPEED"],
        "modification_func": lambda df: modify_tps_performance(df)
    },
    "P0122": { # TPS Low Input
        "required_pids": ["THROTTLE_POS", "ENGINE_RPM", "SPEED"],
        "modification_func": lambda df: modify_tps_low(df)
    },
    "P0123": { # TPS High Input
        "required_pids": ["THROTTLE_POS", "ENGINE_RPM", "SPEED"],
        "modification_func": lambda df: modify_tps_high(df)
    },

    # Add other heuristics for P0078, P1004, U1004, etc. if possible
    # These might be harder if they relate to circuit issues rather than sensor performance
}

# --- Modification Functions (Placeholders) ---

def modify_maf_load(segment: pd.DataFrame, direction: str = 'low', load_threshold: float = 60.0, change_duration_steps: int = 5, noise_level: float = 0.05) -> pd.DataFrame:
    """Simulates MAF sensor issues relative to load, applying change gradually with noise."""
    modified_segment = segment.copy()
    logging.debug(f"Applying MAF modification (direction: {direction})")
    
    load_col = 'ENGINE_LOAD'
    maf_col = 'MAF'
    
    if load_col not in modified_segment.columns or maf_col not in modified_segment.columns:
        logging.warning(f"Missing required columns ({load_col}, {maf_col}) for MAF modification. Skipping.")
        return segment

    high_load_indices = modified_segment.index[modified_segment[load_col] > load_threshold].tolist()

    if not high_load_indices:
        logging.debug("No high load periods found in segment, no MAF modification applied.")
        return modified_segment

    # Find the start of the first high-load period in the segment
    start_high_load_index = high_load_indices[0]
    # Ensure we don't go past the end of the segment
    end_change_index = min(start_high_load_index + change_duration_steps, segment.index.max() + 1)

    # Indices where the gradual change will happen
    change_indices = segment.index[start_high_load_index:end_change_index]
    # Indices after the gradual change (where effect is stable)
    stable_indices = segment.index[end_change_index:]
    # Filter stable indices to only include those actually meeting the high load condition
    stable_high_load_indices = stable_indices.intersection(high_load_indices)

    if direction == 'low':
        target_reduction_factor = np.random.uniform(0.2, 0.6) # Target reduction (e.g., 40-80% lower)
        # Apply gradual change
        for i, idx in enumerate(change_indices):
            current_factor = 1.0 - (1.0 - target_reduction_factor) * (i + 1) / len(change_indices)
            modified_segment.loc[idx, maf_col] *= current_factor

        # Apply stable change
        if not stable_high_load_indices.empty:
             modified_segment.loc[stable_high_load_indices, maf_col] *= target_reduction_factor

        # Add noise relative to the modified value, only where modified
        all_modified_indices = change_indices.union(stable_high_load_indices)
        if not all_modified_indices.empty:
            noise = np.random.normal(loc=0, scale=noise_level * modified_segment.loc[all_modified_indices, maf_col].mean(), size=len(all_modified_indices))
            modified_segment.loc[all_modified_indices, maf_col] += noise
            # Ensure MAF doesn't go below 0 after adding noise
            modified_segment[maf_col] = modified_segment[maf_col].clip(lower=0)

    # Add logic for 'high' direction if needed

    return modified_segment

def modify_map_load(segment: pd.DataFrame, direction: str = 'low', load_threshold: float = 60.0, change_duration_steps: int = 5, noise_level: float = 0.05, min_map_value: float = 10.0) -> pd.DataFrame:
    """Simulates MAP sensor issues relative to load, applying change gradually with noise."""
    modified_segment = segment.copy()
    logging.debug(f"Applying MAP modification (direction: {direction})")
    
    load_col = 'ENGINE_LOAD'
    map_col = 'INTAKE_MANIFOLD_PRESSURE'
    
    if load_col not in modified_segment.columns or map_col not in modified_segment.columns:
        logging.warning(f"Missing required columns ({load_col}, {map_col}) for MAP modification. Skipping.")
        return segment

    high_load_indices = modified_segment.index[modified_segment[load_col] > load_threshold].tolist()

    if not high_load_indices:
        logging.debug("No high load periods found in segment, no MAP modification applied.")
        return modified_segment

    # Find the start of the first high-load period in the segment
    start_high_load_index = high_load_indices[0]
    end_change_index = min(start_high_load_index + change_duration_steps, segment.index.max() + 1)

    change_indices = segment.index[start_high_load_index:end_change_index]
    stable_indices = segment.index[end_change_index:]
    stable_high_load_indices = stable_indices.intersection(high_load_indices)

    if direction == 'low':
        target_reduction_factor = np.random.uniform(0.3, 0.7) # Target reduction (e.g., 30-70% lower)

        # Apply gradual change
        for i, idx in enumerate(change_indices):
            current_factor = 1.0 - (1.0 - target_reduction_factor) * (i + 1) / len(change_indices)
            modified_segment.loc[idx, map_col] *= current_factor

        # Apply stable change
        if not stable_high_load_indices.empty:
             modified_segment.loc[stable_high_load_indices, map_col] *= target_reduction_factor

        # Add noise relative to the modified value, only where modified
        all_modified_indices = change_indices.union(stable_high_load_indices)
        if not all_modified_indices.empty:
            noise_scale = noise_level * modified_segment.loc[all_modified_indices, map_col].mean()
            # Ensure noise scale is positive, handle case where mean might be ~0
            if noise_scale <= 0:
                 noise_scale = noise_level * 10 # Assign a small default scale if mean is zero/negative
            noise = np.random.normal(loc=0, scale=noise_scale, size=len(all_modified_indices))
            modified_segment.loc[all_modified_indices, map_col] += noise

        # Ensure MAP doesn't go below a plausible minimum
        modified_segment[map_col] = modified_segment[map_col].clip(lower=min_map_value)

    # Add logic for 'high' direction if needed

    return modified_segment

def modify_coolant_thermostat(segment: pd.DataFrame, warmup_steps: int = 20, noise_level: float = 0.1) -> pd.DataFrame:
    """Simulates coolant not reaching temperature after a warmup period, operating in scaled domain."""
    modified_segment = segment.copy()
    logging.debug("Applying Coolant Thermostat (P0128) modification (scaled domain logic)")

    col_name = 'ENGINE_COOLANT_TEMP'
    if col_name not in modified_segment.columns:
        logging.warning(f"'{col_name}' not found in segment. Cannot apply P0128 modification.")
        return segment # Return original if column is missing

    if len(segment) <= warmup_steps:
        logging.debug("Segment shorter than warmup steps, no P0128 modification applied.")
        return modified_segment

    # Target scaled temperature range for "stuck open thermostat"
    # Aim for values typically seen during initial warmup or below normal operating median (which is ~0.22)
    # Let's target a range like -1.0 to 0.0 in the scaled domain.
    target_low_scaled_temp_upper_bound = np.random.uniform(-0.5, 0.0) # Keep it below typical warm median
    # The actual value will be a single value chosen for the segment, then noise added.
    chosen_target_scaled_temp = np.random.uniform(-1.0, target_low_scaled_temp_upper_bound)

    post_warmup_indices = segment.index[warmup_steps:]

    # For rows after warmup, set temp to the chosen low scaled value
    if not post_warmup_indices.empty:
        modified_segment.loc[post_warmup_indices, col_name] = chosen_target_scaled_temp
        
        # Add some noise
        noise = np.random.normal(loc=0, scale=noise_level, size=len(post_warmup_indices))
        modified_segment.loc[post_warmup_indices, col_name] += noise
        
        # Clip to ensure it stays within a somewhat reasonable low scaled range
        # e.g., not excessively negative due to noise, and not above our intended low range.
        modified_segment.loc[post_warmup_indices, col_name] = modified_segment.loc[post_warmup_indices, col_name].clip(
            lower=-2.5, # Arbitrary reasonable scaled minimum
            upper=target_low_scaled_temp_upper_bound + (noise_level * 2) # Allow slight overshoot from noise but keep low
        )
    else:
        logging.debug("No post-warmup indices found, P0128 modification might not be effective.")

    return modified_segment

def modify_egr(segment: pd.DataFrame, error_type: str = 'range', min_error_pct: float = 15.0, max_error_pct: float = 50.0, fault_duration_steps: int = 10, noise_level: float = 2.0) -> pd.DataFrame:
    """Simulates EGR errors (currently P0404 range), applying error over a sub-duration with noise."""
    modified_segment = segment.copy()
    logging.debug(f"Applying EGR modification (type: {error_type})")
    
    egr_col = 'EGR_ERROR'
    rpm_col = 'ENGINE_RPM'
    load_col = 'ENGINE_LOAD'

    if egr_col not in modified_segment.columns:
        logging.warning(f"'{egr_col}' column not found, cannot apply EGR modification. Returning original segment.")
        return segment
        
    if rpm_col not in modified_segment.columns or load_col not in modified_segment.columns:
        logging.warning(f"Missing required columns ({rpm_col}, {load_col}) for EGR modification context. Skipping EGR modification.")
        return segment

    # Define operating conditions where EGR error might manifest (mid-range RPM/Load)
    mid_rpm_load_mask = (modified_segment[rpm_col] > 1500) & \
                        (modified_segment[rpm_col] < 3500) & \
                        (modified_segment[load_col] > 20) & \
                        (modified_segment[load_col] < 80)

    eligible_indices = modified_segment.index[mid_rpm_load_mask].tolist()

    if not eligible_indices:
        logging.debug("No eligible mid-range RPM/Load conditions found in segment.")
        return modified_segment

    if error_type == 'range': # P0404
        # Choose a random start point within the eligible period for the fault
        if len(eligible_indices) <= fault_duration_steps:
            start_fault_index_loc = 0 # Apply to the whole eligible period if shorter than fault duration
        else:
            start_fault_index_loc = np.random.randint(0, len(eligible_indices) - fault_duration_steps + 1)

        start_fault_actual_index = eligible_indices[start_fault_index_loc]
        end_fault_actual_index = min(start_fault_actual_index + fault_duration_steps, segment.index.max() + 1)

        # Indices where the fault is active
        fault_indices = segment.index[start_fault_actual_index:end_fault_actual_index].intersection(eligible_indices) # Ensure still within eligible condition

        if not fault_indices.empty:
            logging.debug(f"Applying P0404 fault to indices: {fault_indices.min()} to {fault_indices.max()}")
            # Assign high error percentage
            error_value = np.random.uniform(min_error_pct, max_error_pct, size=len(fault_indices))
            modified_segment.loc[fault_indices, egr_col] = error_value

            # Add noise
            noise = np.random.normal(loc=0, scale=noise_level, size=len(fault_indices))
            modified_segment.loc[fault_indices, egr_col] += noise
            # Clip error percentage (e.g., between -100 and 100, or other plausible range)
            modified_segment.loc[fault_indices, egr_col] = modified_segment.loc[fault_indices, egr_col].clip(lower=-100, upper=100)
        else:
            logging.debug("Could not find suitable sub-duration to apply P0404 fault.")

    # Add logic for P0401 (Insufficient Flow) or P0402 (Excessive Flow) if possible
    # This might involve modifying EGR_COMMANDED or setting EGR_ERROR to specific negative/positive values

    return modified_segment

# --- NEW Modification Functions --- 

def modify_fuel_trim(segment: pd.DataFrame, direction: str = 'lean', trim_threshold: float = 15.0, noise_level: float = 1.0, application_fraction: float = 0.6) -> pd.DataFrame:
    """Simulates lean or rich fuel trim conditions by shifting available trims."""
    modified_segment = segment.copy()
    logging.debug(f"Applying Fuel Trim modification (direction: {direction})")

    stft1_col = 'SHORT TERM FUEL TRIM BANK 1'
    stft2_col = 'SHORT TERM FUEL TRIM BANK 2'
    ltft2_col = 'LONG TERM FUEL TRIM BANK 2'

    available_trims = [col for col in [stft1_col, stft2_col, ltft2_col] if col in modified_segment.columns]
    if not available_trims:
        logging.warning("No fuel trim columns found for modification. Skipping.")
        return segment

    # Apply modification to a fraction of the segment to simulate intermittent issue
    n_rows = len(modified_segment)
    n_apply = int(n_rows * application_fraction)
    start_index = np.random.randint(0, n_rows - n_apply + 1)
    apply_indices = modified_segment.index[start_index : start_index + n_apply]

    if direction == 'lean':
        target_shift = np.random.uniform(trim_threshold, trim_threshold + 10) # Shift towards positive
        logging.debug(f" Applying lean shift: {target_shift:.2f}")
        for col in available_trims:
            noise = np.random.normal(loc=0, scale=noise_level, size=n_apply)
            modified_segment.loc[apply_indices, col] = modified_segment.loc[apply_indices, col] + target_shift + noise
            # Clip to prevent excessively high values (e.g., max 50%)
            modified_segment.loc[apply_indices, col] = modified_segment.loc[apply_indices, col].clip(upper=50)
    elif direction == 'rich':
        target_shift = np.random.uniform(- (trim_threshold + 10), -trim_threshold) # Shift towards negative
        logging.debug(f" Applying rich shift: {target_shift:.2f}")
        for col in available_trims:
            noise = np.random.normal(loc=0, scale=noise_level, size=n_apply)
            modified_segment.loc[apply_indices, col] = modified_segment.loc[apply_indices, col] + target_shift + noise
            # Clip to prevent excessively low values (e.g., min -50%)
            modified_segment.loc[apply_indices, col] = modified_segment.loc[apply_indices, col].clip(lower=-50)

    return modified_segment

def modify_tps_performance(segment: pd.DataFrame, noise_level: float = 5.0, flatten_prob: float = 0.3, application_fraction: float = 0.5) -> pd.DataFrame:
    """Simulates TPS range/performance issues by adding noise or flattening signal intermittently."""
    modified_segment = segment.copy()
    logging.debug("Applying TPS Performance (P0121) modification")
    tps_col = 'THROTTLE_POS'
    if tps_col not in modified_segment.columns:
        logging.warning(f"'{tps_col}' not found. Skipping P0121 modification.")
        return segment

    n_rows = len(modified_segment)
    n_apply = int(n_rows * application_fraction)
    start_index = np.random.randint(0, n_rows - n_apply + 1)
    apply_indices = modified_segment.index[start_index : start_index + n_apply]

    if np.random.rand() < flatten_prob: # Chance to flatten the signal
        flatten_value = modified_segment.loc[apply_indices, tps_col].mean() + np.random.normal(0, noise_level/2) # Flatten around mean + noise
        modified_segment.loc[apply_indices, tps_col] = flatten_value
        logging.debug(" P0121: Flattening TPS signal")
    else: # Add significant noise
        noise = np.random.normal(loc=0, scale=noise_level, size=len(apply_indices))
        modified_segment.loc[apply_indices, tps_col] += noise
        logging.debug(" P0121: Adding noise to TPS signal")

    # Clip TPS between 0 and 100
    modified_segment[tps_col] = modified_segment[tps_col].clip(lower=0, upper=100)
    return modified_segment

def modify_tps_low(segment: pd.DataFrame, stuck_value_max: float = 5.0, application_fraction: float = 0.7) -> pd.DataFrame:
    """Simulates TPS Low Input (P0122) by clipping the value near zero."""
    modified_segment = segment.copy()
    logging.debug("Applying TPS Low Input (P0122) modification")
    tps_col = 'THROTTLE_POS'
    if tps_col not in modified_segment.columns:
        logging.warning(f"'{tps_col}' not found. Skipping P0122 modification.")
        return segment

    n_rows = len(modified_segment)
    n_apply = int(n_rows * application_fraction)
    start_index = np.random.randint(0, n_rows - n_apply + 1)
    apply_indices = modified_segment.index[start_index : start_index + n_apply]

    stuck_value = np.random.uniform(0, stuck_value_max)
    modified_segment.loc[apply_indices, tps_col] = stuck_value # Force low value
    logging.debug(f" P0122: Forcing TPS low ({stuck_value:.2f})")

    # Clip just in case, although stuck_value should be low anyway
    modified_segment[tps_col] = modified_segment[tps_col].clip(lower=0, upper=100)
    return modified_segment

def modify_tps_high(segment: pd.DataFrame, stuck_value_min: float = 90.0, application_fraction: float = 0.7) -> pd.DataFrame:
    """Simulates TPS High Input (P0123) by clipping the value near 100."""
    modified_segment = segment.copy()
    logging.debug("Applying TPS High Input (P0123) modification")
    tps_col = 'THROTTLE_POS'
    if tps_col not in modified_segment.columns:
        logging.warning(f"'{tps_col}' not found. Skipping P0123 modification.")
        return segment

    n_rows = len(modified_segment)
    n_apply = int(n_rows * application_fraction)
    start_index = np.random.randint(0, n_rows - n_apply + 1)
    apply_indices = modified_segment.index[start_index : start_index + n_apply]

    stuck_value = np.random.uniform(stuck_value_min, 100.0)
    modified_segment.loc[apply_indices, tps_col] = stuck_value # Force high value
    logging.debug(f" P0123: Forcing TPS high ({stuck_value:.2f})")

    # Clip just in case
    modified_segment[tps_col] = modified_segment[tps_col].clip(lower=0, upper=100)
    return modified_segment

# --- Core Functions ---

def load_source_data(data_path: str, required_pids: List[str], target_col: str = 'parsed_dtcs') -> pd.DataFrame:
    """Loads data from parquet, checks for required PIDs, and filters out rows with existing DTCs."""
    logging.info(f"Loading source data from: {data_path}")
    try:
        # Load required PIDs plus the target column to check for existing DTCs
        columns_to_load = list(set(required_pids + [target_col]))
        df = pd.read_parquet(data_path, columns=columns_to_load)
        logging.info(f"Source data loaded. Shape: {df.shape}")

        # Basic check for required columns (PIDs only)
        missing_pids = [pid for pid in required_pids if pid not in df.columns]
        if missing_pids:
            raise ValueError(f"Source data missing required PIDs for heuristic: {missing_pids}")

        # Filter out rows that already have DTCs in the target column
        if target_col in df.columns:
            # Ensure the target column is list-like (handle potential numpy arrays)
            def is_empty_list_or_array(item):
                if isinstance(item, (list, np.ndarray)):
                    return len(item) == 0
                return True # Treat non-list/array types or NaNs as empty

            original_count = len(df)
            no_dtc_mask = df[target_col].apply(is_empty_list_or_array)
            df_filtered = df[no_dtc_mask].copy()
            filtered_count = len(df_filtered)
            logging.info(f"Filtered out {original_count - filtered_count} rows with existing DTCs. Remaining rows: {filtered_count}")

            if filtered_count == 0:
                 raise ValueError("No rows without existing DTCs found in the source data.")

            # Drop the target column now that filtering is done
            df_filtered = df_filtered.drop(columns=[target_col])
            return df_filtered
        else:
            logging.warning(f"Target column '{target_col}' not found in source data. Assuming no pre-existing DTCs.")
            return df # Return original df if target column doesn't exist

    except Exception as e:
        logging.error(f"Error loading source data: {e}")
        sys.exit(1)

def select_normal_segments(df: pd.DataFrame, num_segments: int, segment_length: int, min_speed_threshold: float = 1.0, min_moving_fraction: float = 0.1) -> List[pd.DataFrame]:
    """Selects random segments of 'normal' driving data, ensuring some vehicle movement."""
    segments = []
    max_start_index = len(df) - segment_length
    if max_start_index < 0:
        logging.error(f"Segment length ({segment_length}) is greater than total data length ({len(df)}).")
        return []

    logging.info(f"Selecting {num_segments} segments of length {segment_length}, ensuring some movement...")
    attempts = 0
    while len(segments) < num_segments and attempts < num_segments * 10: # Increased attempts for stricter filtering
        start_index = np.random.randint(0, max_start_index + 1)
        segment = df.iloc[start_index : start_index + segment_length]

        # Check for vehicle movement
        if 'SPEED' in segment.columns:
            moving_fraction = (segment['SPEED'] > min_speed_threshold).mean()
            if moving_fraction < min_moving_fraction:
                attempts += 1
                continue # Skip segment if not enough movement
        else:
            logging.warning("SPEED column not found, cannot check for movement.")

        segments.append(segment.reset_index(drop=True))
        attempts += 1

    if len(segments) < num_segments:
         logging.warning(f"Could only extract {len(segments)} segments meeting criteria, requested {num_segments}.")

    return segments


def generate_synthetic_data(source_df: pd.DataFrame, target_dtc: str, num_samples: int, segment_length: int) -> List[pd.DataFrame]:
    """Generates synthetic data samples for a given DTC."""
    if target_dtc not in HEURISTIC_MAP:
        logging.error(f"Heuristic for DTC {target_dtc} not found in HEURISTIC_MAP.")
        return []

    heuristic_info = HEURISTIC_MAP[target_dtc]
    required_pids = heuristic_info["required_pids"]
    modification_func = heuristic_info["modification_func"]

    # Ensure all required PIDs for the heuristic are in the source_df
    missing_pids_for_heuristic = [pid for pid in required_pids if pid not in source_df.columns]
    if missing_pids_for_heuristic:
        logging.error(f"Source data for DTC {target_dtc} generation is missing PIDs required by its heuristic: {missing_pids_for_heuristic}. Skipping.")
        return []

    logging.info(f"Selecting normal segments for DTC: {target_dtc}")
    normal_segments = select_normal_segments(source_df, num_segments=num_samples * 2, segment_length=segment_length) # Select more to increase chances

    if not normal_segments:
        logging.warning(f"No suitable normal segments found to generate synthetic data for {target_dtc}.")
        return []

    generated_samples_dfs = []
    for i in range(num_samples):
        if not normal_segments: # Should not happen if initial check passed, but good practice
            logging.warning(f"Ran out of normal segments while generating sample {i+1} for {target_dtc}.")
            break
        
        segment_df = normal_segments.pop(np.random.randint(len(normal_segments)))
        
        logging.debug(f"Modifying segment for sample {i+1} of {target_dtc} (Original shape: {segment_df.shape})")
        
        # Apply the modification
        modified_segment_df = modification_func(segment_df.copy()) # Pass a copy

        if modified_segment_df is None or modified_segment_df.empty:
            logging.warning(f"Modification function for {target_dtc} returned empty or None for sample {i+1}. Original segment shape: {segment_df.shape}")
            continue

        # Add the generated DTC as a column. Store as a list of lists for consistency with MultiLabelBinarizer.
        # Each row in the segment gets the same target DTC.
        modified_segment_df['generated_dtc'] = [[target_dtc]] * len(modified_segment_df)
        
        # TODO: Consider adding original_vehicle_id and original_segment_indices if needed later for traceability
        # For now, just the modified data with the DTC
        generated_samples_dfs.append(modified_segment_df)
        logging.debug(f"Generated sample {i+1}/{num_samples} for {target_dtc}. Shape: {modified_segment_df.shape}")

    logging.info(f"Successfully generated {len(generated_samples_dfs)} DataFrame samples for {target_dtc}.")
    return generated_samples_dfs


def save_synthetic_samples(samples_dfs: List[pd.DataFrame], target_dtc: str, output_dir: str):
    """Saves a list of synthetic sample DataFrames to a single Parquet file."""
    if not samples_dfs:
        logging.warning(f"No synthetic samples (DataFrames) provided for {target_dtc}, nothing to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Concatenate all sample DataFrames
        combined_df = pd.concat(samples_dfs, ignore_index=True)
    except Exception as e:
        logging.error(f"Error concatenating DataFrames for {target_dtc}: {e}")
        # Log details of the DFs if possible
        for i, df_s in enumerate(samples_dfs):
            logging.debug(f"Sample DF {i} columns: {df_s.columns.tolist()}, Shape: {df_s.shape}, Dtypes: {df_s.dtypes.to_dict()}")
        return

    if combined_df.empty:
        logging.warning(f"Concatenated DataFrame for {target_dtc} is empty. Nothing to save.")
        return

    file_name = f"synthetic_{target_dtc}_samples.parquet"
    file_path = os.path.join(output_dir, file_name)
    
    try:
        combined_df.to_parquet(file_path, index=False)
        logging.info(f"Successfully saved {len(combined_df)} rows ({len(samples_dfs)} original segments) for {target_dtc} to {file_path}")
    except Exception as e:
        logging.error(f"Error saving combined Parquet file for {target_dtc} to {file_path}: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DTC data based on heuristics.")
    parser.add_argument("--source_data_path", type=str, default="data/model_input/exp1_14drivers_14cars_dailyRoutes_model_input.parquet",
                        help="Path to the source Parquet file (cleaned, aggregated data).")
    parser.add_argument("--dtc_codes", type=str, nargs='+', default=["P0128"], # Example: P0101 P0128
                        help="List of DTC codes to generate data for.")
    parser.add_argument("--num_samples_per_dtc", type=int, default=50,
                        help="Number of synthetic samples to generate per DTC.")
    parser.add_argument("--segment_length", type=int, default=100,
                        help="Length of data segments (number of timesteps) for each sample.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the generated synthetic samples.")

    args = parser.parse_args()

    # Load base data once
    # Determine all PIDs required by all requested DTCs to load efficiently
    all_required_pids = set()
    for dtc_code in args.dtc_codes:
        if dtc_code in HEURISTIC_MAP:
            all_required_pids.update(HEURISTIC_MAP[dtc_code]["required_pids"])
        else:
            logging.warning(f"DTC {dtc_code} not found in HEURISTIC_MAP. Cannot determine required PIDs.")
    
    if not all_required_pids:
        logging.error("No PIDs identified for the requested DTCs. Exiting.")
        return

    # Add other columns that might be useful for segment selection or are implicitly needed
    # e.g., VEHICLE_ID, TIMESTAMP, SPEED, and the target column for filtering normal data
    utility_cols = ['VEHICLE_ID', 'absolute_timestamp', 'SPEED', 'parsed_dtcs'] # Corrected timestamp column name
    columns_to_load_from_source = list(all_required_pids.union(utility_cols))


    # Check if source_df needs to be loaded or if it can be passed to generate_synthetic_data
    # For now, assume generate_synthetic_data will handle its own PID checks from a potentially larger df
    
    full_source_df = load_source_data(args.source_data_path, required_pids=columns_to_load_from_source, target_col='parsed_dtcs')

    if full_source_df.empty:
        logging.error(f"Failed to load or process source data from {args.source_data_path}. Exiting.")
        return

    for dtc_code in args.dtc_codes:
        logging.info(f"--- Generating synthetic data for DTC: {dtc_code} ---")
        
        # Filter the full_source_df to only include PIDs relevant for this specific DTC + utility cols for selection
        current_dtc_pids = []
        if dtc_code in HEURISTIC_MAP:
            current_dtc_pids.extend(HEURISTIC_MAP[dtc_code]["required_pids"])
        
        cols_for_current_dtc_generation = list(set(current_dtc_pids + utility_cols + list(all_required_pids))) # Ensure all modified PIDs are there plus selection cols
        # Ensure no KeyError by checking columns exist in full_source_df before subselecting
        cols_present_in_source = [col for col in cols_for_current_dtc_generation if col in full_source_df.columns]
        
        source_df_for_dtc = full_source_df[cols_present_in_source].copy()


        generated_sample_dfs = generate_synthetic_data(
            source_df=source_df_for_dtc, # Pass the potentially filtered df
            target_dtc=dtc_code,
            num_samples=args.num_samples_per_dtc,
            segment_length=args.segment_length
        )

        if generated_sample_dfs:
            save_synthetic_samples(generated_sample_dfs, dtc_code, args.output_dir)
        else:
            logging.warning(f"No samples were generated for DTC {dtc_code}.")

    logging.info("Synthetic data generation process completed.")

if __name__ == "__main__":
    main()