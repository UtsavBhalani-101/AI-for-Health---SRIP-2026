from collections import Counter
import numpy as np
import pandas as pd
import argparse
import logging
import os
import glob
from utils.cleaning import clean_data
from utils.io import parse_arguments, validate_input_path, load_data
from utils.signal_processing import bandpass_filter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# * ----- core functions ------

# ^ creating 30 sec windows with 15 sec overlap for the given data
def create_windows(nasal):
    logger.info("Window creating started...")
    signal = nasal['Value'].values
    time_index = nasal.index
    
    sampling_interval = (time_index[1] - time_index[0]).total_seconds()
    samples_per_30sec = int(30 / sampling_interval)
    samples_per_15sec = int(15 / sampling_interval)
    
    windows = []
    window_times = []

    for start in range(0, len(signal) - samples_per_30sec + 1, samples_per_15sec):

        end = start + samples_per_30sec

        window_signal = signal[start:end]

        window_start_time = time_index[start]
        window_end_time = time_index[end - 1]

        windows.append(window_signal)
        window_times.append((window_start_time, window_end_time))

    X = np.array(windows)
    
    logger.info("Windows created")
    return window_times, X
    
# ^ labelling each window with one of the labels of flow events
def labeling_windows(window_times, flow_events):
    labels = []
    logger.info("Labeling started...")
    
    for (window_start, window_end) in window_times:

        overlapping = flow_events[
            (flow_events['start_time'] <= window_end) &
            (flow_events['end_time'] >= window_start)
        ]

        max_overlap = 0
        assigned_label = "Normal"

        for _, row in overlapping.iterrows():

            overlap_start = max(window_start, row['start_time'])
            overlap_end = min(window_end, row['end_time'])

            overlap_duration = (overlap_end - overlap_start).total_seconds()

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                assigned_label = row['Disorder']

        if max_overlap < 15:
            assigned_label = "Normal"

        labels.append(assigned_label)
        
    

    unknown = set(l for l in labels if l != "Normal")
    if unknown:
        logger.info(f"Abnormal event types found: {sorted(unknown)}")

    y = np.array([0 if l == "Normal" else 1 for l in labels])
    
    logger.info("Labeling completed")
    return y

# ^ combining X, y and participant id into one df and save the output
def save_output(input_path, output_path, X, y):
    participant_id = os.path.basename(input_path)

    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)

    df.insert(0, "label", y)
    df.insert(0, "participant_id", participant_id)

    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"{participant_id}.csv")
    df.to_csv(output_file, index=False)

    class_counts = Counter(y)
    logger.info(f"Saved dataset to: {output_file}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    # output_path = os.path.join(output_path, f"{participant_id}.csv")
    # df.to_csv(output_path, index=False)

# * ------ Wrapper function ------------    
 
def initialize_paths():
    parser = argparse.ArgumentParser(description="Create breathing dataset")
    
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=False, default="Dataset/")
    parser.add_argument("--filter", action="store_true",
                        help="Apply bandpass filtering to nasal signal")

    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    logger.info(f"Arguments parsed — input: {input_path} | output: {output_path} | filter: {args.filter}")
    
    return input_path, output_path, args.filter

def path_validation(input_path):
    flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path = validate_input_path(input_path)
    logger.info("Input path validated")
    return flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path

def get_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path):
    logger.info("Loading data from the given input path...")
    flow_events, thorac, spo2, sleep_profile, nasal = load_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path)
    logger.info("Data loaded")
    return flow_events, thorac, spo2, sleep_profile, nasal

def preprocessing_data(flow_events, thorac, spo2, sleep_profile, nasal):
    logger.info("Data preprocessing started...")
    flow_events, thorac, spo2, sleep_profile, nasal = clean_data(flow_events, thorac, spo2, sleep_profile, nasal)
    logger.info("Data preprocessing completed")
    return flow_events, thorac, spo2, sleep_profile, nasal
    

def main():
    logger.info("Starting dataset creation pipeline")
    try:
        input_path, output_path, apply_filter = initialize_paths()

        flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path = path_validation(input_path)

        flow_events, thorac, spo2, sleep_profile, nasal = get_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path)

        flow_events, thorac, spo2, sleep_profile, nasal = preprocessing_data(flow_events, thorac, spo2, sleep_profile, nasal)

        if apply_filter:
            logger.info("Applying bandpass filter to nasal signal")
            nasal = bandpass_filter(nasal)

        window_times, X = create_windows(nasal)

        y = labeling_windows(window_times, flow_events)

        save_output(input_path, output_path, X, y)

        logger.info("Pipeline completed successfully")

    except Exception:
        logger.exception(f"Error, Pipeline Failed")

if __name__ == "__main__":
    main()