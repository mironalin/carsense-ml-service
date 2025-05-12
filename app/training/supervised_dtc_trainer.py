#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to train a supervised model to predict potential DTCs based on sensor data,
using heuristically generated labels from the anomaly detection step.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example classifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer # Import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier # Import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV # Import RandomizedSearchCV
from scipy.stats import randint, uniform # Import distributions for param grid
from joblib import dump, load # Import load as well for potentially loading later
import argparse
import os
import logging

try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except ImportError:
    LGBMClassifier = None
    lightgbm_available = False
    logging.warning("LightGBM not installed. LGBMClassifier will not be available.")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define feature lists (mirroring anomaly_detection.py for consistency)
CORE_PIDS_FOR_SUPERVISED = [
    "ENGINE_RPM",
    "COOLANT_TEMPERATURE",
    "INTAKE_MANIFOLD_ABSOLUTE_PRESSURE",
    "INTAKE_AIR_TEMPERATURE",
    "MASS_AIR_FLOW",
    "THROTTLE_POSITION",
    "CONTROL_MODULE_VOLTAGE",
    "ENGINE_FUEL_RATE",
    "VEHICLE_SPEED",
    "AMBIENT_AIR_TEMPERATURE",
    "BAROMETRIC_PRESSURE",
    "CALCULATED_ENGINE_LOAD_VALUE",
]

DERIVED_FEATURES_FOR_SUPERVISED = [
    # Rolling Means (Window 10)
    'ENGINE_RPM_rol_10_mean',
    'VEHICLE_SPEED_rol_10_mean',
    'CALCULATED_ENGINE_LOAD_VALUE_rol_10_mean',
    # Rolling Std Devs (Window 10)
    'ENGINE_RPM_rol_10_std',
    'VEHICLE_SPEED_rol_10_std',
    'CALCULATED_ENGINE_LOAD_VALUE_rol_10_std',
    # Differences (Lag 1)
    'ENGINE_RPM_diff_1',
    'VEHICLE_SPEED_diff_1',
    'CALCULATED_ENGINE_LOAD_VALUE_diff_1'
]

# Potentially reuse feature lists from anomaly detection or define specific ones here
# from app.preprocessing.anomaly_detection import CORE_PIDS_FOR_ANOMALY, DERIVED_FEATURES_FOR_ANOMALY


def load_and_prepare_data(features_path: str, anomalies_path: str) -> pd.DataFrame:
    """
    Loads the main features dataset and the anomaly analysis results,
    merges them, and prepares the multi-label target variable using MultiLabelBinarizer.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MultiLabelBinarizer, list]: X, y, fitted_binarizer, labels
    """
    logging.info(f"Loading features from: {features_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    df_features = pd.read_parquet(features_path)
    logging.info(f"Feature data shape: {df_features.shape}")

    logging.info(f"Loading anomaly results from: {anomalies_path}")
    if not os.path.exists(anomalies_path):
        raise FileNotFoundError(f"Anomaly results file not found: {anomalies_path}")
    df_anomalies = pd.read_csv(anomalies_path, index_col=0) # Assuming index is timestamp/identifier
    # Parse the 'potential_dtcs' column back into lists (it gets saved as string)
    # Safely evaluate the string representation of the list
    try:
        df_anomalies['potential_dtcs'] = df_anomalies['potential_dtcs'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
    except Exception as e:
        logging.error(f"Error parsing 'potential_dtcs' column: {e}. Ensure it contains valid list strings.")
        # Handle error - maybe fill with empty lists or raise
        df_anomalies['potential_dtcs'] = [[] for _ in range(len(df_anomalies))]

    logging.info(f"Anomaly data shape: {df_anomalies.shape}")

    # Extract only the 'potential_dtcs' column to merge
    df_dtcs = df_anomalies[['potential_dtcs']]

    # Merge DTCs into the main feature DataFrame based on index
    # Use a left merge to keep all original feature data points
    df_merged = df_features.merge(df_dtcs, left_index=True, right_index=True, how='left')

    # Fill NaNs created by the merge (for data points not flagged as anomalies) with empty lists
    # Important: Access the column using .loc for reliable assignment
    dtc_col_mask = df_merged['potential_dtcs'].isna()
    df_merged.loc[dtc_col_mask, 'potential_dtcs'] = df_merged.loc[dtc_col_mask, 'potential_dtcs'].apply(lambda x: [])


    logging.info(f"Merged data shape: {df_merged.shape}")
    if df_merged.shape[0] != df_features.shape[0]:
        logging.warning("Row count changed after merging anomalies. Check index alignment.")

    # --- Define Multi-Label Target Variable --- 
    logging.info("Preparing multi-label target variable...")
    mlb = MultiLabelBinarizer()

    # Fit and transform the list of potential DTCs
    y = mlb.fit_transform(df_merged['potential_dtcs'])
    labels = mlb.classes_.tolist() # Get the DTC labels
    logging.info(f"Created multi-label target matrix with shape: {y.shape}")
    logging.info(f"Detected DTC Labels (Columns): {labels}")

    # Convert y back to a DataFrame for easier handling later if needed (optional)
    y_df = pd.DataFrame(y, columns=labels, index=df_merged.index)

    # Calculate and log label distribution
    label_counts = y_df.sum().sort_values(ascending=False)
    logging.info(f"Label distribution (count per DTC):\n{label_counts.to_string()}")

    # Define specific features to use for training
    features_to_use = CORE_PIDS_FOR_SUPERVISED + DERIVED_FEATURES_FOR_SUPERVISED
    # features_to_use = [] # Placeholder

    # Ensure selected features exist in the dataframe
    available_features = [f for f in features_to_use if f in df_merged.columns]
    if len(available_features) != len(features_to_use):
        missing = set(features_to_use) - set(available_features)
        logging.warning(f"Some specified features are missing: {missing}")
    if not available_features:
         raise ValueError("No features selected or available for training.")
    logging.info(f"Using {len(available_features)} features for training: {available_features}")

    # Prepare X
    X = df_merged[available_features]
    # y is now y_df (DataFrame)

    # Note: Imputation will be done *after* train/test split to avoid data leakage

    return X, y_df, mlb, labels


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, model_save_path: str, model_type: str = 'rf', n_iter=10, cv=3):
    """Trains the specified classifier wrapped in OneVsRestClassifier, potentially using RandomizedSearchCV, and saves it."""
    # Note: Imputer is now handled in main and saved separately
    logging.info(f"Training model ({model_type} wrapped in OneVsRestClassifier)...")

    # Define the base estimator
    if model_type == 'rf':
        base_estimator = RandomForestClassifier(random_state=42, class_weight='balanced')
        # Parameters for RandomForest (prefix with 'estimator__')
        param_distributions = {
            'estimator__n_estimators': randint(50, 200),
            'estimator__max_depth': [None] + list(randint(10, 50).rvs(5)), # None + 5 random depths
            'estimator__min_samples_split': randint(2, 11),
            'estimator__min_samples_leaf': randint(1, 11)
        }
        scoring = 'f1_weighted' # Example scoring

    elif model_type == 'lgbm' and lightgbm_available:
        base_estimator = LGBMClassifier(random_state=42, class_weight='balanced')
        # Parameters for LightGBM (prefix with 'estimator__')
        param_distributions = {
            'estimator__n_estimators': randint(50, 300),
            'estimator__num_leaves': randint(20, 60),
            'estimator__learning_rate': uniform(0.01, 0.2), # Distribution from 0.01 to 0.21
            'estimator__colsample_bytree': uniform(0.6, 0.4), # Distribution from 0.6 to 1.0
            'estimator__reg_alpha': uniform(0, 1),
            'estimator__reg_lambda': uniform(0, 1),
        }
        scoring = 'f1_weighted' # Can customize scoring

    elif model_type == 'lgbm' and not lightgbm_available:
         raise ValueError("LightGBM selected but not installed. Please install lightgbm.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Wrap the base estimator in OneVsRestClassifier
    ovr_classifier = OneVsRestClassifier(base_estimator, n_jobs=1) # n_jobs=-1 here conflicts with RandomizedSearchCV

    # Setup RandomizedSearchCV
    # n_jobs=-1 uses all available cores for CV folds
    logging.info(f"Starting RandomizedSearchCV (n_iter={n_iter}, cv={cv}, scoring='{scoring}')...")
    random_search = RandomizedSearchCV(
        ovr_classifier,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1, # Parallelize CV folds
        random_state=42,
        verbose=1 # Log search progress
    )

    # Fit the RandomizedSearchCV object (this performs the search and fits the best model)
    random_search.fit(X_train, y_train)

    logging.info(f"RandomizedSearchCV complete. Best score ({scoring}): {random_search.best_score_:.4f}")
    logging.info(f"Best parameters found: {random_search.best_params_}")

    # The best model found by the search
    best_model = random_search.best_estimator_

    # Save the best model
    if model_save_path:
        logging.info(f"Saving best model found by search to: {model_save_path}")
        output_dir = os.path.dirname(model_save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        dump(best_model, model_save_path)

    return best_model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, labels=None):
    """Evaluates the model on the test set."""
    # Added labels argument for classification report
    logging.info("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    # Confusion matrix is not straightforward for multi-label, removing for now
    # logging.info("Confusion Matrix:")
    # cm_str = np.array_str(confusion_matrix(y_test, y_pred)) # Doesn't work directly
    # logging.info("\n" + cm_str)

    logging.info("\nClassification Report:")
    # classification_report handles multi-label format if y_test/y_pred are binary indicator format
    # Provide target_names if available
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    logging.info("\n" + report)

    # --- Feature Importances ---
    if hasattr(model, 'feature_importances_') and hasattr(X_test, 'columns'):
        logging.info("\n--- Feature Importances ---")
        importances = model.feature_importances_
        feature_names = X_test.columns
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        # Log the feature importances
        logging.info("\n" + feature_importance_df.to_string(index=False))
    else:
        logging.warning("Could not extract feature importances (model type might not support it or X_test is not a DataFrame).")


def main(args):
    """Main execution function."""
    logging.info("Starting supervised DTC training process (Multi-Label)...")

    try:
        # Load data (before imputation), also get binarizer and labels
        X, y, mlb, labels = load_and_prepare_data(args.features_path, args.anomalies_path)

        # Split data
        logging.info(f"Splitting data (test_size={args.test_size}, random_state=42)...")
        # Stratification is tricky for multi-label, often omitted or requires advanced techniques
        # We will omit stratify=y for now
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
        logging.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

        # --- Imputation (Fit on Train, Transform Train & Test) ---
        logging.info(f"Applying SimpleImputer (strategy='{args.imputer_strategy}')...")
        imputer = SimpleImputer(strategy=args.imputer_strategy)

        # Get feature names before imputation (imputer returns numpy array)
        feature_names = X_train.columns.tolist()

        # Fit on the training data ONLY
        X_train_imputed = imputer.fit_transform(X_train)
        # Transform the test data using the *fitted* imputer
        X_test_imputed = imputer.transform(X_test)

        # Convert back to DataFrame to keep feature names (optional but good practice)
        X_train = pd.DataFrame(X_train_imputed, columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=feature_names, index=X_test.index)
        logging.info("Imputation complete.")

        # Save the fitted imputer
        if args.imputer_output_path:
            logging.info(f"Saving imputer to: {args.imputer_output_path}")
            output_dir = os.path.dirname(args.imputer_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            dump(imputer, args.imputer_output_path)

        # --- Save the Label Binarizer ---
        if args.label_binarizer_path:
            logging.info(f"Saving label binarizer to: {args.label_binarizer_path}")
            output_dir = os.path.dirname(args.label_binarizer_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            dump(mlb, args.label_binarizer_path)
        # --- End Imputation & Binarizer Saving ---

        # Train model using imputed data (now potentially with hyperparameter search)
        model = train_model(
            X_train, y_train,
            args.model_output_path,
            model_type=args.model_type,
            n_iter=args.search_iterations, # Pass search args
            cv=args.search_cv_folds      # Pass search args
        )

        # Evaluate model using imputed data
        evaluate_model(model, X_test, y_test, labels=labels) # Pass labels

        logging.info("Supervised DTC training script finished successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
    except ValueError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a supervised multi-label model with optional hyperparameter tuning.")
    parser.add_argument("--features-path", required=True, help="Path to the main Parquet file with features.")
    parser.add_argument("--anomalies-path", required=True, help="Path to the CSV file containing anomaly detection results (including 'potential_dtcs').")
    parser.add_argument("--model-output-path", required=True, help="Path to save the trained model (e.g., .joblib file).")
    parser.add_argument("--imputer-output-path", required=True, help="Path to save the fitted imputer (e.g., .joblib file).")
    parser.add_argument("--label-binarizer-path", required=True, help="Path to save the fitted MultiLabelBinarizer (e.g., .joblib file).")
    parser.add_argument("--imputer-strategy", type=str, default='mean', choices=['mean', 'median', 'most_frequent'], help="Strategy for SimpleImputer.")
    parser.add_argument("--model-type", type=str, default='rf', choices=['rf', 'lgbm'], help="Type of model to train ('rf' for RandomForest, 'lgbm' for LightGBM).")
    parser.add_argument("--search-iterations", type=int, default=10, help="Number of parameter settings sampled by RandomizedSearchCV. Set to 0 to disable search and use defaults.")
    parser.add_argument("--search-cv-folds", type=int, default=3, help="Number of cross-validation folds for RandomizedSearchCV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for the test set.")

    args = parser.parse_args()
    main(args)