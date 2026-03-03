import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Argument Parsing
# ----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train baseline ML model with LOPO")
    parser.add_argument("--dataset_path", required=True, help="Path to Dataset folder")
    return parser.parse_args()


# ----------------------------
# Load and Combine Datasets
# ----------------------------

def load_datasets(dataset_path):
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found in dataset folder")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info(f"Loaded dataset with shape: {combined_df.shape}")

    # Drop any rows that contain NaN values (e.g. skew=NaN for constant windows)
    nan_rows = combined_df.isnull().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning(f"Dropping {nan_rows} rows containing NaN values")
        combined_df = combined_df.dropna().reset_index(drop=True)
        logger.info(f"Shape after NaN drop: {combined_df.shape}")
    else:
        logger.info("No NaN values found — dataset is clean")

    return combined_df


# ----------------------------
# Training Pipeline
# ----------------------------

def train_lopo(df):

    X = df.drop(columns=["participant_id", "label"]).values
    y = df["label"].values
    groups = df["participant_id"].values

    logo = LeaveOneGroupOut()

    all_reports = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        logger.info(f"Starting fold {fold+1}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaling (fit ONLY on train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic Regression
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        all_reports.append(report)

        logger.info("\n" + classification_report(y_test, y_pred))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    return all_reports


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_arguments()
    df = load_datasets(args.dataset_path)
    train_lopo(df)


if __name__ == "__main__":
    main()