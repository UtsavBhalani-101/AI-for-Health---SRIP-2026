import numpy as np
import pandas as pd
import argparse
import logging
import os
import glob
from utils.cleaning import *
from utils.io import *

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
        
    
    # 3-class mapping:
    #   0 = Normal
    #   1 = Any apnea (Hypopnea, Obstructive Apnea, Mixed Apnea, or any future apnea type)
    #   2 = Body event
    # Using conditions instead of a hardcoded dict — no KeyError on unseen labels.
    unique_events = set(l for l in labels if l != "Normal")
    if unique_events:
        logger.info(f"Non-normal event types found: {sorted(unique_events)}")

    def map_label(l):
        if l == "Normal":
            return 0
        elif l == "Body event":
            return 2
        else:
            return 1   # any apnea type

    y = np.array([map_label(l) for l in labels])
    
    logger.info("Labeling completed")
    return y

# ^ combining X, y and participant id into one df and save the output
def save_output(input_path, output_path, X, y):
    participant_id = os.path.basename(input_path)
    
    df = pd.DataFrame(X)
    df["label"] = y
    df['participant_id'] = participant_id
    
    cols = ["participant_id", "label"] + [c for c in df.columns if c not in ("participant_id", "label")]
    df = df[cols]
    print(df)
    
    from collections import Counter
    print(Counter(y))
    
    output_path = os.path.join(output_path, f"{participant_id}.csv")
    df.to_csv(output_path, index=False)

# * ------ Wrapper (helper) function ------------    
 
def initialize_paths():
    input_path, output_path = parse_arguments(default_output_path=r"Dataset_3labels/")
    logger.info(f"Arguments parsed — input: {input_path} | output: {output_path}")
    return input_path, output_path

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
        input_path, output_path = initialize_paths()

        flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path = path_validation(input_path)

        flow_events, thorac, spo2, sleep_profile, nasal = get_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path)

        flow_events, thorac, spo2, sleep_profile, nasal = preprocessing_data(flow_events, thorac, spo2, sleep_profile, nasal)

        window_times, X = create_windows(nasal)

        y = labeling_windows(window_times, flow_events)

        save_output(input_path, output_path, X, y)

        logger.info("Pipeline completed successfully")

    except Exception:
        logger.exception(f"Error, Pipeline Failed")

if __name__ == "__main__":
    main()