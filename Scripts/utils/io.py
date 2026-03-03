import numpy as np
import pandas as pd
import argparse
import os
import glob

# ^ enforcing strict user file input path for data
def parse_arguments(default_output_path):
    parser = argparse.ArgumentParser(
        description="Generate visualization PDF for a participant folder")
    
    parser.add_argument("--input",required=True,type=str,help="Path to participant folder (e.g., Data/AP01)")
    parser.add_argument("--output", required=False, type=str, default=default_output_path, help="Path where visulization saved")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    return input_path, output_path


# ^ validating given input path
def validate_input_path(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Expected a directory: {input_path}")

    flow_event_path = glob.glob(os.path.join(input_path, "Flow Events*.txt"))
    thorac_path = glob.glob(os.path.join(input_path, "Thorac*.txt"))
    spo2_path = glob.glob(os.path.join(input_path, "SPO2*.txt"))
    sleep_profile_path = glob.glob(os.path.join(input_path, "Sleep profile*.txt"))
    nasal_path = [
        f for f in glob.glob(os.path.join(input_path, "Flow*.txt"))
        if "events" not in os.path.basename(f).lower()
    ]

    if len(nasal_path) != 1:
        raise ValueError(f"Expected exactly 1 flow signal file, found: {nasal_path}")

    return flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path

    
# ^ loading data 
def load_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path):
    nasal = pd.read_csv(nasal_path[0], delimiter='\t')
    flow_events = pd.read_csv(flow_event_path[0], delimiter='\t')
    thorac = pd.read_csv(thorac_path[0], delimiter='\t')
    spo2 = pd.read_csv(spo2_path[0], delimiter='\t')
    sleep_profile = pd.read_csv(sleep_profile_path[0], delimiter='\t')

    return flow_events, thorac, spo2, sleep_profile, nasal 