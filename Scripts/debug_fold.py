import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


DATASET_PATH = "Dataset_binary/"
TARGET_PARTICIPANT = "AP05"   # <-- Change this


def load_dataset(dataset_path):
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def run_single_fold(df, target_participant):
    print(f"\n=== Debugging Fold: Test = {target_participant} ===")

    train_df = df[df["participant_id"] != target_participant]
    test_df = df[df["participant_id"] == target_participant]

    print("\nTrain distribution:")
    print(train_df["label"].value_counts())

    print("\nTest distribution:")
    print(test_df["label"].value_counts())

    X_train = train_df.drop(columns=["participant_id", "label"]).values
    y_train = train_df["label"].values

    X_test = test_df.drop(columns=["participant_id", "label"]).values
    y_test = test_df["label"].values

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Extra diagnostic: predicted class distribution
    print("\nPredicted distribution:")
    print(pd.Series(y_pred).value_counts())


def main():
    df = load_dataset(DATASET_PATH)

    print("Overall dataset distribution:")
    print(df["label"].value_counts())

    run_single_fold(df, TARGET_PARTICIPANT)


if __name__ == "__main__":
    main()