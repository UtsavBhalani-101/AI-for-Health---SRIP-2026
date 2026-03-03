import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# * ----- Feature Extraction -----

FEATURE_NAMES = ["mean", "std", "min", "max", "ptp", "energy", "median", "skew"]

def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Given a 1D array of raw signal samples for one window,
    returns an 8-element feature vector.
    """
    return np.array([
        np.mean(signal),          # mean
        np.std(signal),           # standard deviation
        np.min(signal),           # minimum
        np.max(signal),           # maximum
        np.ptp(signal),           # peak-to-peak (max - min)
        np.sum(signal ** 2),      # signal energy
        np.median(signal),        # median
        0.0 if np.std(signal) == 0 else float(skew(signal)),  # skewness (0 for constant signal)
    ])


# * ----- Core Pipeline -----

def get_csv_files(dataset_path: str) -> list:
    """Return sorted list of participant CSV paths from the dataset folder."""
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {dataset_path}")
    return csv_files


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every row in df, extract statistical features from the raw signal
    columns (everything except 'participant_id' and 'label').

    Returns a compact DataFrame with columns:
        participant_id | label | mean | std | min | max | ptp | energy | median | skew
    """
    meta_cols = ["participant_id", "label"]
    signal_cols = [c for c in df.columns if c not in meta_cols]

    logger.info(f"Signal columns per row  : {len(signal_cols)}")
    logger.info(f"Output features per row : {len(FEATURE_NAMES)}  {FEATURE_NAMES}")
    logger.info("Extracting features...")

    raw_signals = df[signal_cols].values  # shape: (n_rows, n_samples)

    feature_matrix = np.apply_along_axis(extract_features, axis=1, arr=raw_signals)
    # feature_matrix shape: (n_rows, 8)

    feature_df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)

    # Prepend meta columns
    result = pd.concat([df[meta_cols].reset_index(drop=True), feature_df], axis=1)

    logger.info(f"Feature extraction complete — output shape: {result.shape}")
    return result


# * ----- Validation -----

def validate_row_count(df_raw: pd.DataFrame, df_features: pd.DataFrame, filename: str) -> bool:
    """
    Checks that the number of rows in the feature DataFrame matches
    the number of rows in the raw input DataFrame.
    Logs a clear PASS or FAIL message.
    """
    raw_rows = len(df_raw)
    feat_rows = len(df_features)

    if raw_rows == feat_rows:
        logger.info(f"  [PASS] Row count match for {filename}: {raw_rows} rows")
        return True
    else:
        logger.error(
            f"  [FAIL] Row count MISMATCH for {filename}: "
            f"raw={raw_rows} rows  vs  features={feat_rows} rows"
        )
        return False


# * ----- I/O -----

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Feature engineering: compress raw windowed CSVs to statistical features"
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the Dataset folder containing raw participant CSVs"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Directory to save per-participant feature CSVs (default: <dataset_path>/features/)"
    )
    return parser.parse_args()


# * ----- Main -----

def main():
    args = parse_arguments()

    dataset_path = os.path.abspath(args.dataset_path)

    # Output directory (default: Dataset/features/)
    if args.output_path is not None:
        output_dir = os.path.abspath(args.output_path)
    else:
        output_dir = os.path.join(dataset_path, "features")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Dataset path : {dataset_path}")
    logger.info(f"Output dir   : {output_dir}")

    csv_files = get_csv_files(dataset_path)
    logger.info(f"Found {len(csv_files)} participant file(s)")

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)          # e.g. AP01.csv
        participant = os.path.splitext(filename)[0]     # e.g. AP01

        df_raw = pd.read_csv(csv_path)
        logger.info(f"Processing {filename} — {len(df_raw)} rows")

        df_features = engineer_features(df_raw)

        # Sanity check: row counts must match
        validate_row_count(df_raw, df_features, filename)

        # Label distribution per participant
        label_counts = df_features["label"].value_counts().sort_index()
        for label, count in label_counts.items():
            logger.info(f"  Label {label}: {count} windows ({100*count/len(df_features):.1f}%)")

        out_path = os.path.join(output_dir, filename)   # same name, different folder
        df_features.to_csv(out_path, index=False)
        logger.info(f"  Saved -> {out_path}")

    logger.info("All participants processed.")


if __name__ == "__main__":
    main()
