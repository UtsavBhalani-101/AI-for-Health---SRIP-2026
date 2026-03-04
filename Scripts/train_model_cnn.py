import os
import sys
import glob
import argparse

# Add project root to path so 'models' package can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from models.cnn_model import SimpleCNN


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train CNN with LOPO")
    parser.add_argument("--dataset_path", required=True)
    return parser.parse_args()


def load_dataset(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def train_lopo(df):

    X = df.drop(columns=["participant_id", "label"]).values
    y = df["label"].values
    groups = df["participant_id"].values

    # reshape for CNN: (samples, channels, sequence)
    X = X.reshape(-1, 1, X.shape[1])

    logo = LeaveOneGroupOut()

    device = torch.device("cpu")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        print(f"\n=== Fold {fold+1} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=64,
            shuffle=True
        )

        model = SimpleCNN().to(device)

        neg = (y_train == 0).sum().item()
        pos = (y_train == 1).sum().item()
        pos_weight = torch.tensor([neg / pos]).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(15):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = model(xb).squeeze()
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

        print(classification_report(y_test.cpu(), preds.cpu()))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test.cpu(), preds.cpu()))


def main():
    args = parse_arguments()
    df = load_dataset(args.dataset_path)
    train_lopo(df)


if __name__ == "__main__":
    main()