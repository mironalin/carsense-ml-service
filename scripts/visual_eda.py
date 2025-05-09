import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # For selecting numerical dtypes
import random # For selecting random samples for time series

def plot_histograms(df: pd.DataFrame, numerical_cols: list[str], base_output_dir: str):
    """
    Generates and saves histograms for specified numerical columns into a 'histograms' subdirectory.
    """
    output_dir = os.path.join(base_output_dir, "histograms")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\n--- Generating Histograms for {len(numerical_cols)} numerical columns (in {output_dir}) ---")
    for i, col in enumerate(numerical_cols):
        plt.figure(figsize=(10, 6))
        # Using try-except for robustness, though less likely to fail with selected numericals
        try:
            sns.histplot(df[col], kde=True, bins=50)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plot_filename = os.path.join(output_dir, f"hist_{col.replace('/', '_')}.png") # Sanitize filename
            plt.savefig(plot_filename)
            plt.close() # Close the plot figure to free memory
            if (i + 1) % 10 == 0: # Print progress every 10 plots
                print(f"Generated histogram {i+1}/{len(numerical_cols)}: {plot_filename}")
        except Exception as e:
            print(f"Could not generate histogram for {col}: {e}")
            plt.close() # Ensure plot is closed even if error occurs
    print("Finished generating histograms.")

def plot_boxplots(df: pd.DataFrame, numerical_cols: list[str], base_output_dir: str):
    """
    Generates and saves box plots for specified numerical columns into a 'boxplots' subdirectory.
    """
    output_dir = os.path.join(base_output_dir, "boxplots")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\n--- Generating Box Plots for {len(numerical_cols)} numerical columns (in {output_dir}) ---")
    for i, col in enumerate(numerical_cols):
        plt.figure(figsize=(8, 10)) # Often better with vertical orientation for many features
        try:
            # Check if column has any non-NaN values and some variance
            if df[col].notna().sum() == 0:
                print(f"Skipping box plot for {col}: all values are NaN.")
                plt.close()
                continue
            if df[col].nunique(dropna=True) < 2:
                print(f"Skipping box plot for {col}: not enough unique values to draw a meaningful box plot.")
                plt.close()
                continue

            sns.boxplot(y=df[col])
            plt.title(f"Box Plot of {col}")
            plt.ylabel(col)
            plot_filename = os.path.join(output_dir, f"boxplot_{col.replace('/', '_')}.png")
            plt.savefig(plot_filename)
            plt.close()
            if (i + 1) % 10 == 0:
                 print(f"Generated box plot {i+1}/{len(numerical_cols)}: {plot_filename}")
        except Exception as e:
            print(f"Could not generate box plot for {col}: {e}")
            plt.close()
    print("Finished generating box plots.")

def plot_countplots(df: pd.DataFrame, categorical_cols: list[str], base_output_dir: str):
    """
    Generates and saves count plots into a 'countplots' subdirectory.
    """
    output_dir = os.path.join(base_output_dir, "countplots")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\n--- Generating Count Plots for {len(categorical_cols)} categorical columns (in {output_dir}) ---")
    for i, col in enumerate(categorical_cols):
        if df[col].nunique(dropna=False) > 50: # Heuristic: don't plot if too many unique values (e.g. high cardinality identifiers)
            print(f"Skipping count plot for {col}: more than 50 unique values ({df[col].nunique(dropna=False)}). May be an ID-like column.")
            continue

        plt.figure(figsize=(12, 7))
        try:
            sns.countplot(y=df[col], order=df[col].value_counts(dropna=False).index) # Show NaNs if any, order by frequency
            plt.title(f"Count Plot of {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            plot_filename = os.path.join(output_dir, f"countplot_{col.replace('/', '_')}.png")
            plt.savefig(plot_filename)
            plt.close()
            if (i + 1) % 5 == 0 or (i+1) == len(categorical_cols): # Print progress more frequently for fewer categoricals
                print(f"Generated count plot {i+1}/{len(categorical_cols)}: {plot_filename}")
        except Exception as e:
            print(f"Could not generate count plot for {col}: {e}")
            plt.close()
    print("Finished generating count plots.")

def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: list[str], base_output_dir: str, file_prefix: str = "correlation_heatmap"):
    """
    Generates and saves a correlation heatmap into a 'correlations' subdirectory.
    """
    output_dir = os.path.join(base_output_dir, "correlations")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\n--- Generating Correlation Heatmap for {len(numerical_cols)} numerical columns (in {output_dir}) ---")
    if not numerical_cols:
        print("No numerical columns provided for correlation heatmap.")
        return

    plt.figure(figsize=(20, 18)) # May need to adjust based on number of features
    try:
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".1f", linewidths=.5)
        # annot=True can be very slow and cluttered for many features.
        # Consider annot=True only for smaller subsets of features if needed.
        plt.title(f"Correlation Heatmap of Numerical Features ({len(numerical_cols)} features)")
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{file_prefix}_{len(numerical_cols)}_features.png")
        plt.savefig(plot_filename, dpi=150) # Increase DPI for better resolution if needed
        plt.close()
        print(f"Generated correlation heatmap: {plot_filename}")
    except Exception as e:
        print(f"Could not generate correlation heatmap: {e}")
        plt.close()
    print("Finished generating correlation heatmap.")

def plot_time_series_snippets(
    df: pd.DataFrame,
    pids_to_plot: list[str],
    timestamp_col: str,
    group_by_col: str,
    n_samples: int,
    base_output_dir: str
):
    """
    Generates and saves time series plots for selected PIDs from a sample of groups (files/trips).

    Args:
        df (pd.DataFrame): The input DataFrame.
        pids_to_plot (list[str]): List of PID column names to plot.
        timestamp_col (str): Name of the timestamp column (e.g., 'absolute_timestamp').
        group_by_col (str): Column to group by for individual time series (e.g., 'source_file').
        n_samples (int): Number of sample groups (files) to plot.
        base_output_dir (str): Base directory to save the plots (e.g., eda_plots/romanian_driving_ds).
    """
    output_dir = os.path.join(base_output_dir, "time_series_snippets")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\n--- Generating Time Series Snippets for {n_samples} samples (in {output_dir}) ---")

    if group_by_col not in df.columns:
        print(f"Error: Group by column '{group_by_col}' not found in DataFrame.")
        return
    if timestamp_col not in df.columns:
        print(f"Error: Timestamp column '{timestamp_col}' not found in DataFrame.")
        return

    actual_pids_in_df = [pid for pid in pids_to_plot if pid in df.columns]
    if not actual_pids_in_df:
        print(f"None of the specified PIDs to plot found in DataFrame columns: {pids_to_plot}")
        return
    if len(actual_pids_in_df) != len(pids_to_plot):
        print(f"Warning: Some PIDs to plot were not found in DataFrame. Plotting available: {actual_pids_in_df}")

    unique_groups = df[group_by_col].unique()
    if len(unique_groups) == 0:
        print(f"No unique groups found for column '{group_by_col}'. Cannot generate time series snippets.")
        return

    sample_groups = random.sample(list(unique_groups), min(n_samples, len(unique_groups)))

    for i, group_name in enumerate(sample_groups):
        group_df = df[df[group_by_col] == group_name].sort_values(by=timestamp_col)
        if group_df.empty:
            print(f"Skipping group '{group_name}': No data after filtering.")
            continue

        num_pids = len(actual_pids_in_df)
        fig, axes = plt.subplots(num_pids, 1, figsize=(15, 4 * num_pids), sharex=True)
        if num_pids == 1: # If only one PID, axes is not a list
            axes = [axes]

        fig.suptitle(f"Time Series Snippet for: {group_name}", fontsize=16)

        for ax, pid in zip(axes, actual_pids_in_df):
            if pid in group_df.columns:
                ax.plot(group_df[timestamp_col], group_df[pid], label=pid)
                ax.set_ylabel(pid)
                ax.legend(loc='upper right')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'{pid}\nnot available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.grid(True)

        axes[-1].set_xlabel(timestamp_col)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

        sanitized_group_name = str(group_name).replace('/', '_').replace(':', '-') # Further sanitize
        plot_filename = os.path.join(output_dir, f"ts_snippet_{sanitized_group_name}.png")
        try:
            plt.savefig(plot_filename)
            print(f"Generated time series snippet {i+1}/{len(sample_groups)}: {plot_filename}")
        except Exception as e:
            print(f"Could not save time series snippet for {group_name}: {e}")
        plt.close(fig)
    print("Finished generating time series snippets.")

def create_visual_eda(file_path: str, base_output_plot_dir: str):
    """
    Loads an aggregated Parquet file and generates various EDA plots.

    Args:
        file_path (str): Path to the aggregated Parquet file.
        base_output_plot_dir (str): Base directory for this dataset's plots (e.g., eda_plots/romanian_driving_ds).
    """
    print(f"--- Visual EDA for: {file_path} ---")
    print(f"Plots will be saved under: {base_output_plot_dir}\n")

    # Create base output directory for the dataset if it doesn't exist
    if not os.path.exists(base_output_plot_dir):
        os.makedirs(base_output_plot_dir)
        print(f"Created base plot directory: {base_output_plot_dir}")

    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded Parquet file. Shape: {df.shape}\n")
    except Exception as e:
        print(f"Error loading Parquet file {file_path}: {e}")
        return

    # --- 1. Numerical Features Analysis ---
    # Identify numerical columns (float64, int64, int32 etc.)
    # Exclude 'TIME_SEC' as it's just a raw counter from original data.
    # Also exclude simple time components like 'hour', 'dayofweek', 'is_weekend' if they exist,
    # as their distributions are less interesting as raw numbers (categorical or cyclical better)

    # Select columns of numberic types
    potential_numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Define columns to explicitly exclude from typical numerical analysis (histograms, boxplots etc.)
    # These might be identifiers, raw time counters, or categorical-like integers.
    cols_to_exclude_from_numerical_plots = ['TIME_SEC', 'hour', 'dayofweek', 'is_weekend']

    numerical_cols_for_plotting = [
        col for col in potential_numerical_cols
        if col not in cols_to_exclude_from_numerical_plots
    ]

    if not numerical_cols_for_plotting:
        print("No numerical columns identified for general plotting after exclusions.")
    else:
        plot_histograms(df, numerical_cols_for_plotting, base_output_plot_dir)
        plot_boxplots(df, numerical_cols_for_plotting, base_output_plot_dir)
        plot_correlation_heatmap(df, numerical_cols_for_plotting, base_output_plot_dir)

    # --- 2. Categorical Features Analysis ---
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Also consider boolean columns if they exist and are not already handled
    # For now, let's stick to object/category. 'is_weekend' is int but acts categorical.

    # We might want to manually add columns like 'is_weekend' if they are integer encoded but categorical in nature
    # For now, let's rely on dtypes and add specific integer-categorical columns if needed later.
    # Example: if 'is_weekend' was not caught by select_dtypes but we want to plot it:
    # if 'is_weekend' in df.columns and 'is_weekend' not in categorical_cols:
    #     categorical_cols.append('is_weekend')

    if not categorical_cols:
        print("\nNo categorical columns (object/category dtype) identified for count plotting.")
    else:
        plot_countplots(df, categorical_cols, base_output_plot_dir)

    # --- 5. Time Series Snippets ---
    # Define some PIDs of interest that are likely to be present and informative
    # These are original (non-derived) PIDs for clearer time series interpretation
    pids_for_ts = [
        'ENGINE_RPM', 'VEHICLE_SPEED', 'COOLANT_TEMPERATURE',
        'CALCULATED_ENGINE_LOAD_VALUE', 'MASS_AIR_FLOW', 'THROTTLE_POSITION'
    ]
    # Ensure 'absolute_timestamp' and 'source_file' exist, which they should from our preprocessing
    if 'absolute_timestamp' in df.columns and 'source_file' in df.columns:
        plot_time_series_snippets(
            df,
            pids_to_plot=pids_for_ts,
            timestamp_col='absolute_timestamp',
            group_by_col='source_file',
            n_samples=5, # Plot for 5 random files
            base_output_dir=base_output_plot_dir
        )
    else:
        print("\nSkipping time series snippets: 'absolute_timestamp' or 'source_file' column missing.")

    print(f"\n--- End of Visual EDA for: {file_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual EDA plots for an aggregated Parquet file.")
    parser.add_argument("input_file", type=str,
                        help="Path to the aggregated Parquet file (e.g., data/features/romanian_driving_ds_aggregated.parquet).")
    parser.add_argument("--plot_dir", type=str, default="eda_plots",
                        help="Directory to save the generated plots (default: eda_plots).")

    args = parser.parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.input_file))[0].replace('_aggregated', '')
    # base_output_plot_dir is now the dataset-specific directory like eda_plots/romanian_driving_ds
    base_output_plot_dir_for_dataset = os.path.join(args.plot_dir, dataset_name)

    create_visual_eda(args.input_file, base_output_plot_dir_for_dataset)