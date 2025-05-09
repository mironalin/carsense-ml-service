import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import scalers
import numpy as np # Import NumPy

def report_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Reports the percentage of missing values for each column in the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        A pandas Series with column names as index and percentage of missing values as values.
    """
    missing_percentage = df.isnull().mean() * 100
    print("Missing values percentage per column:")
    print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))
    return missing_percentage

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: List[str] = None) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame using a specified strategy.
    Placeholder: More sophisticated strategies can be added.

    Args:
        df: The input DataFrame.
        strategy: The imputation strategy. Currently supports 'mean', 'median', 'mode', or 'drop_rows'.
                    'drop_rows' will drop rows with any missing values in the specified columns (or all columns if None).
        columns: List of columns to apply the strategy. If None, applies to all numeric columns for 'mean'/'median'/'mode',
                    or all columns for 'drop_rows'.

    Returns:
        DataFrame with missing values handled.
    """
    df_cleaned = df.copy()

    if columns is None:
        target_columns = df_cleaned.select_dtypes(include=['number']).columns.tolist() if strategy in ['mean', 'median', 'mode'] else df_cleaned.columns.tolist()
    else:
        target_columns = columns

    print(f"Handling missing values with strategy: '{strategy}' for columns: {target_columns}")

    for col in target_columns:
        if col not in df_cleaned.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        if df_cleaned[col].isnull().any():
            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    # df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                else:
                    print(f"Warning: Column '{col}' is not numeric. Skipping mean imputation.")
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    # df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    print(f"Warning: Column '{col}' is not numeric. Skipping median imputation.")
            elif strategy == 'mode':
                mode_values = df_cleaned[col].mode()
                if not mode_values.empty:
                    # df_cleaned[col].fillna(mode_values.iloc[0], inplace=True)
                    df_cleaned[col] = df_cleaned[col].fillna(mode_values.iloc[0])
                else:
                    # df_cleaned[col].fillna(pd.NA, inplace=True)
                    df_cleaned[col] = df_cleaned[col].fillna(pd.NA)
                    print(f"Warning: Mode for column '{col}' is empty. Filled with pd.NA.")
            elif strategy == 'drop_rows':
                # This will drop rows if *any* of the target_columns have NA in that row.
                original_shape = df_cleaned.shape
                df_cleaned.dropna(subset=target_columns, inplace=True)
                if df_cleaned.shape[0] < original_shape[0]:
                    print(f"Dropped {original_shape[0] - df_cleaned.shape[0]} rows with NA in columns: {target_columns}. New shape: {df_cleaned.shape}")
                return df_cleaned # Return early as shape may have changed significantly
            else:
                print(f"Warning: Unknown strategy '{strategy}'. No changes made for column '{col}'.")
        else:
            # print(f"No missing values in column '{col}'.") # Optional: for verbosity
            pass

    return df_cleaned

def correct_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Corrects data types of specified columns.
    Note: The data_loader already attempts pd.to_numeric. This is for more specific corrections if needed.

    Args:
        df: The input DataFrame.
        type_mapping: A dictionary where keys are column names and values are the target data types (e.g., 'int64', 'float64', 'str').

    Returns:
        DataFrame with corrected data types.
    """
    df_corrected = df.copy()
    for col, target_type in type_mapping.items():
        if col in df_corrected.columns:
            try:
                # For integer types, use pandas nullable integer type if NaNs are present
                if df_corrected[col].isnull().any() and target_type.lower().startswith('int'):
                    # Ensure target_type is in a format like 'Int64' (capital I)
                    nullable_int_type = target_type.capitalize() + 'Dtype'
                    if not nullable_int_type.endswith('DtypeDtype'): # Avoid double Dtype
                        if not nullable_int_type.startswith('Int'): nullable_int_type = 'Int' + nullable_int_type.replace('dtype','Dtype')
                        if not nullable_int_type.endswith('Dtype'): nullable_int_type=nullable_int_type.replace('dtype','Dtype')
                        if 'Dtype' not in nullable_int_type : nullable_int_type +='Dtype'
                        # a bit of hackery to ensure it's Int64Dtype() etc.
                        if '64' in nullable_int_type : actual_type = 'Int64'
                        elif '32' in nullable_int_type : actual_type = 'Int32'
                        elif '16' in nullable_int_type : actual_type = 'Int16'
                        elif '8' in nullable_int_type : actual_type = 'Int8'
                        else : actual_type = 'Int64' # Default if no size found
                        df_corrected[col] = df_corrected[col].astype(actual_type)
                else:
                    df_corrected[col] = df_corrected[col].astype(target_type)
                print(f"Corrected dtype for column '{col}' to {df_corrected[col].dtype}.")
            except Exception as e:
                print(f"Error correcting dtype for column '{col}' to {target_type}: {e}")
        else:
            print(f"Warning: Column '{col}' not found for dtype correction.")
    return df_corrected

def apply_rolling_mean(df: pd.DataFrame, columns: List[str] = None, window_size: int = 3) -> pd.DataFrame:
    """
    Applies a rolling mean (moving average) to specified numeric columns for noise reduction.

    Args:
        df: The input DataFrame.
        columns: List of columns to apply noise reduction. If None, applies to all numeric columns.
        window_size: The size of the moving window.

    Returns:
        DataFrame with rolling mean applied to specified columns.
    """
    df_smoothed = df.copy()

    if columns is None:
        target_columns = df_smoothed.select_dtypes(include=['number']).columns.tolist()
    else:
        # Ensure columns exist and are numeric
        target_columns = [col for col in columns if col in df_smoothed.columns and pd.api.types.is_numeric_dtype(df_smoothed[col])]

    print(f"Applying rolling mean with window size {window_size} to columns: {target_columns}")

    for col in target_columns:
        if df_smoothed[col].isnull().all():
            print(f"Column '{col}' is all NaN. Skipping rolling mean.")
            continue
        try:
            df_smoothed[col] = df_smoothed[col].rolling(window=window_size, center=True, min_periods=1).mean()
        except Exception as e:
            print(f"Error applying rolling mean to column '{col}': {e}")

    return df_smoothed

def apply_scaling(df: pd.DataFrame, columns: List[str] = None, scaler_type: str = 'standard') -> pd.DataFrame:
    """
    Applies scaling to specified numeric columns in the DataFrame.

    Args:
        df: The input DataFrame.
        columns: List of columns to scale. If None, applies to all numeric columns.
        scaler_type: Type of scaler to use. Currently supports 'standard' (StandardScaler)
                        and 'minmax' (MinMaxScaler).

    Returns:
        DataFrame with specified columns scaled.
    """
    df_scaled = df.copy()

    if columns is None:
        target_columns = df_scaled.select_dtypes(include=['number']).columns.tolist()
    else:
        # Ensure columns exist and are numeric
        target_columns = [col for col in columns if col in df_scaled.columns and pd.api.types.is_numeric_dtype(df_scaled[col])]

    print(f"Applying {scaler_type} scaling to columns: {target_columns}")

    for col in target_columns:
        if df_scaled[col].isnull().all():
            print(f"Column '{col}' is all NaN. Skipping scaling.")
            continue

        # Scaler needs to operate on a 2D array (column vector)
        # Keep track of original NaNs to re-apply them after scaling, as scalers might convert them
        nan_mask = df_scaled[col].isnull()
        column_data = df_scaled[col].dropna().values.reshape(-1, 1)

        if column_data.shape[0] == 0: # All values were NaN after dropna
            print(f"Column '{col}' has no valid data points after dropping NaNs. Skipping scaling.")
            continue

        try:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                print(f"Warning: Unknown scaler_type '{scaler_type}'. Skipping column '{col}'.")
                continue

            scaled_values = scaler.fit_transform(column_data)

            # Put scaled values back into the column, respecting original NaN positions
            # Create a temporary series of the correct size, fill with np.nan (float NaN)
            # Ensure the dtype is float to be compatible with scikit-learn scalers
            temp_col = pd.Series(np.nan, index=df_scaled[col].index, dtype=float)
            # Place scaled values where original data was not NaN
            temp_col.loc[~nan_mask] = scaled_values.flatten() # Use .loc for safer assignment
            df_scaled[col] = temp_col

        except Exception as e:
            print(f"Error applying {scaler_type} scaling to column '{col}': {e}")
            # Optionally, revert to original column if scaling fails
            # df_scaled[col] = df[col]

    return df_scaled

def handle_outliers_iqr(df: pd.DataFrame, columns: List[str] = None, multiplier: float = 1.5, strategy: str = 'cap') -> pd.DataFrame:
    """
    Handles outliers in specified numeric columns using the IQR method.

    Args:
        df: The input DataFrame.
        columns: List of numeric columns to check. If None, applies to all numeric columns.
        multiplier: The IQR multiplier to define outlier bounds.
        strategy: Method to handle outliers: 'cap', 'nan', or 'remove_rows'.

    Returns:
        DataFrame with outliers handled.
    """
    df_processed = df.copy()

    if columns is None:
        target_columns = df_processed.select_dtypes(include=['number']).columns.tolist()
    else:
        target_columns = [col for col in columns if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    print(f"Handling outliers with IQR (multiplier={multiplier}, strategy='{strategy}') for columns: {target_columns}")

    rows_to_drop = pd.Index([]) # For 'remove_rows' strategy

    for col in target_columns:
        if df_processed[col].isnull().all():
            print(f"Column '{col}' is all NaN. Skipping outlier detection.")
            continue

        # Ensure column has enough non-NaN values for quantile calculation
        if df_processed[col].count() < 2: # .count() gives non-NaN values
            print(f"Column '{col}' has less than 2 non-NaN values. Skipping outlier detection.")
            continue

        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1

        # Add a check for IQR == 0 to avoid issues with constant data segments
        if IQR == 0:
            # If IQR is 0, it means at least 50% of the data points are the same value.
            # In this case, any point different from Q1 (which will be equal to Q3 and the median)
            # could be considered an outlier if we strictly follow the (Q1 - 1.5*0) rule.
            # A more robust approach for constant segments might be to only flag values
            # truly different from this constant value, or simply not apply IQR.
            # For now, let's print a warning and skip IQR for this column if IQR is 0.
            print(f"Warning: IQR for column '{col}' is 0 (data might be constant or have low variance). Skipping IQR outlier handling for this column.")
            continue # Skip to the next column

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        num_outliers = outliers.sum()

        if num_outliers > 0:
            print(f"Found {num_outliers} outliers in column '{col}' (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
            if strategy == 'cap':
                df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
                df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])
                print(f"Capped outliers in '{col}'.")
            elif strategy == 'nan':
                df_processed[col] = np.where(outliers, np.nan, df_processed[col])
                print(f"Replaced outliers with NaN in '{col}'.")
            elif strategy == 'remove_rows':
                # Collect indices of rows with outliers in this column
                rows_to_drop = rows_to_drop.union(df_processed[outliers].index)
                # Defer actual dropping until all columns are processed for 'remove_rows'
            else:
                print(f"Warning: Unknown outlier handling strategy '{strategy}' for column '{col}'.")
        # else:
            # print(f"No outliers found in column '{col}'.") # Optional for verbosity

    if strategy == 'remove_rows' and not rows_to_drop.empty:
        original_shape = df_processed.shape
        df_processed.drop(index=rows_to_drop, inplace=True)
        print(f"Dropped {original_shape[0] - df_processed.shape[0]} rows containing outliers based on columns: {target_columns}. New shape: {df_processed.shape}")

    return df_processed

if __name__ == '__main__':
    # Example Usage (requires RELEVANT_PIDS and a way to load data)
    # This will be more meaningful when integrated into a pipeline.

    # Create a sample DataFrame for demonstration
    data = {
        'ENGINE_RPM': [1000, 1100, None, 1200, 1050, 1000.0],
        'COOLANT_TEMPERATURE': [80, 82, 81, None, 79, 80.0],
        'VEHICLE_SPEED': [50, 55, 52, 58, None, 50.5],
        'FUEL_RAIL_PRESSURE': [None, None, None, None, None, None], # All missing
        'THROTTLE_POSITION': ['low', 'medium', 'low', 'high', 'medium', 'low'] # Categorical
    }
    sample_df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(sample_df)
    sample_df.info()

    print("\n--- Reporting Missing Values ---")
    report_missing_values(sample_df)

    print("\n--- Handling Missing Values (mean imputation) ---")
    # Use a copy for imputation to avoid modifying the original sample_df for subsequent tests
    df_mean_imputed = handle_missing_values(sample_df.copy(), strategy='mean')
    print(df_mean_imputed)
    report_missing_values(df_mean_imputed) # Should show 0 for imputed numeric columns

    print("\n--- Handling Missing Values (median for specific column) ---")
    df_median_imputed_specific = handle_missing_values(sample_df.copy(), strategy='median', columns=['ENGINE_RPM'])
    print(df_median_imputed_specific)

    print("\n--- Handling Missing Values (mode for throttle) ---")
    df_mode_imputed = handle_missing_values(sample_df.copy(), strategy='mode', columns=['THROTTLE_POSITION'])
    print(df_mode_imputed)

    print("\n--- Handling Missing Values (drop rows with NA in VEHICLE_SPEED) ---")
    df_dropped = handle_missing_values(sample_df.copy(), strategy='drop_rows', columns=['VEHICLE_SPEED'])
    print(df_dropped)

    print("\n--- Correcting Data Types (Example) ---")
    # Use a df that might have floats from mean imputation to test Int64 conversion
    type_map = {'ENGINE_RPM': 'Int64', 'COOLANT_TEMPERATURE': 'float32'}
    df_types_corrected = correct_data_types(df_mean_imputed.copy(), type_map)
    df_types_corrected.info()
    print(df_types_corrected)

    print("\n--- Placeholder Noise Reduction ---")
    # df_noise_reduced = placeholder_noise_reduction(df_mean_imputed.copy())
    # print(df_noise_reduced.head())

    print("\n--- Applying Rolling Mean (Noise Reduction) ---")
    # Use a df that has undergone mean imputation for a more complete example
    df_rolled = apply_rolling_mean(df_mean_imputed.copy(), window_size=3) # Apply to all numeric columns by default
    print("DataFrame after rolling mean (window=3) on all numeric columns:")
    print(df_rolled)

    df_rolled_specific = apply_rolling_mean(df_mean_imputed.copy(), columns=['ENGINE_RPM'], window_size=2)
    print("\nDataFrame after rolling mean (window=2) on ENGINE_RPM only:")
    print(df_rolled_specific)

    print("\n--- Applying StandardScaler to all numeric columns of mean-imputed data ---")
    df_standard_scaled = apply_scaling(df_mean_imputed.copy(), scaler_type='standard')
    print(df_standard_scaled)
    df_standard_scaled.info() # Check dtypes, should remain float

    print("\n--- Applying MinMaxScaler to COOLANT_TEMPERATURE of mean-imputed data ---")
    df_minmax_scaled = apply_scaling(df_mean_imputed.copy(), columns=['COOLANT_TEMPERATURE'], scaler_type='minmax')
    print(df_minmax_scaled[['COOLANT_TEMPERATURE']])

    print("\n--- Handling Outliers (IQR capping) on mean-imputed data ---")
    df_outliers_capped = handle_outliers_iqr(df_mean_imputed.copy(), strategy='cap')
    print(df_outliers_capped)

    print("\n--- Handling Outliers (IQR to NaN) on specific column of mean-imputed data ---")
    df_outliers_nan = handle_outliers_iqr(df_mean_imputed.copy(), columns=['ENGINE_RPM'], strategy='nan')
    print(df_outliers_nan)
    report_missing_values(df_outliers_nan) # Check for new NaNs

    print("\n--- Handling Outliers (IQR remove rows) on mean-imputed data ---")
    # Create a df that will definitely have outliers for removal demonstration
    temp_data_for_removal = df_mean_imputed.copy()
    if 'VEHICLE_SPEED' in temp_data_for_removal.columns and len(temp_data_for_removal) > 0:
        temp_data_for_removal.loc[temp_data_for_removal.index[0], 'VEHICLE_SPEED'] = 2000 # Introduce a clear outlier
    df_outliers_removed = handle_outliers_iqr(temp_data_for_removal, strategy='remove_rows', columns=['VEHICLE_SPEED'])
    print(df_outliers_removed)