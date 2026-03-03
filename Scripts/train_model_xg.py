import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Argument Parsing
# ----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train XGBoost model with LOPO cross-validation")
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

        # XGBoost — tree-based, no feature scaling needed
        # scale_pos_weight handles class imbalance (normal vs. abnormal)
        neg = int(np.sum(y_train == 0))
        pos = int(np.sum(y_train == 1))
        scale_pos_weight = neg / pos if pos > 0 else 1

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
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