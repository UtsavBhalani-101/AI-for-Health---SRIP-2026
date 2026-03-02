import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import logging
import os
import glob

# & helper func - clean signal files metadata
def clean_signal_files(df, value_type='integer'):
    df = df.iloc[5: , :].copy()
    
    col_name = df.columns[0]
    
    df['Time'] = df[col_name].str.split(';').str[0]
    df['Value'] = df[col_name].str.split(';').str[1]
    
    df = df.drop([col_name], axis=1)
    
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S,%f")

    if value_type == 'integer':
        df['Value'] = pd.to_numeric(df['Value'].astype(str).str.strip(), errors='coerce')
    elif value_type == 'string':
        df['Value'] = df['Value'].astype(str).str.strip()
        
    
    return df
    

# & helper func - clean event file metadata
def clean_event_file(flow_events):
    flow_events = flow_events.iloc[3: , :].copy()
    
    col_name = flow_events.columns[0]
    
    flow_events['Time'] = flow_events[col_name].str.split(';').str[0]
    flow_events['Value'] = flow_events[col_name].str.split(';').str[1]
    flow_events['Disorder'] = flow_events[col_name].str.split(';').str[2]
    flow_events['Stage'] = flow_events[col_name].str.split(';').str[3]


    flow_events[['start_str', 'end_str']] = flow_events['Time'].str.split('-', expand=True)
    flow_events['date_part'] = flow_events['start_str'].str.split(' ').str[0]
    flow_events['end_str'] = flow_events['date_part'] + ' ' + flow_events['end_str']
    flow_events['start_time'] = pd.to_datetime(flow_events['start_str'], format="%d.%m.%Y %H:%M:%S,%f")
    flow_events['end_time'] = pd.to_datetime(flow_events['end_str'], format="%d.%m.%Y %H:%M:%S,%f")
    flow_events.drop([col_name, 'Time', 'start_str', 'end_str', 'date_part'], axis=1, inplace=True)
        
    
    return flow_events
 

# ^ enforcing strict user file input path for data
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate visualization PDF for a participant folder")
    
    parser.add_argument("--input",required=True,type=str,help="Path to participant folder (e.g., Data/AP01)")
    parser.add_argument("--output", required=False, type=str, default=r"Visualizations/", help="Path where visulization saved")
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

    return nasal, flow_events, thorac, spo2, sleep_profile 


# ^ clean data and make it ready for visulization
def clean_data(nasal, thorac, spo2, flow_events, sleep_profile):
    nasal = clean_signal_files(nasal)
    thorac = clean_signal_files(thorac)
    spo2 = clean_signal_files(spo2)
    sleep_profile = clean_signal_files(sleep_profile, value_type='string')
    flow_events = clean_event_file(flow_events)
    
    return nasal, thorac, spo2, flow_events, sleep_profile

# ^ generating visualization and saving it as pdf
def generate_visualization(nasal, thorac, spo2, flow_events, output_path, input_path):
    window = pd.Timedelta(minutes=5)

    start_time = max(nasal.index[0], thorac.index[0], spo2.index[0])
    end_time = min(nasal.index[-1], thorac.index[-1], spo2.index[-1])

    current_time = start_time

    with PdfPages(output_path) as pdf:

        while current_time < end_time:

            window_end = current_time + window

            seg_flow = nasal.loc[current_time:window_end]
            seg_thorac = thorac.loc[current_time:window_end]
            seg_spo2 = spo2.loc[current_time:window_end]

            fig, axs = plt.subplots(3, 1, figsize=(18, 9), sharex=True)

            # --- Plot signals ---
            axs[0].plot(seg_flow.index, seg_flow['Value'], color='tab:blue', linewidth=0.8)
            axs[0].set_ylabel("Flow")

            axs[1].plot(seg_thorac.index, seg_thorac['Value'], color='tab:orange', linewidth=0.8)
            axs[1].set_ylabel("Thorac")

            axs[2].plot(seg_spo2.index, seg_spo2['Value'], color='tab:green', linewidth=1.0)
            axs[2].set_ylabel("SpO₂")
            axs[2].set_xlabel("Time")

            # --- Add grid ---
            for ax in axs:
                ax.grid(True, alpha=0.3)
                ax.set_xlim(current_time, window_end)

            # --- Highlight Events ---
            events_in_window = flow_events[
                (flow_events['start_time'] <= window_end) &
                (flow_events['end_time'] >= current_time)
            ]

            for _, row in events_in_window.iterrows():

                event_start = max(row['start_time'], current_time)
                event_end = min(row['end_time'], window_end)

                if row['Disorder'] == "Obstructive Apnea":
                    color = 'red'
                elif row['Disorder'] == "Hypopnea":
                    color = 'yellow'
                else:
                    color = 'purple'

                for ax in axs:
                    ax.axvspan(event_start, event_end, color=color, alpha=0.3)

            # --- Improve time ticks ---
            locator = mdates.MinuteLocator(interval=1)   # tick every 1 minute
            formatter = mdates.DateFormatter('%H:%M:%S')

            axs[-1].xaxis.set_major_locator(locator)
            axs[-1].xaxis.set_major_formatter(formatter)

            # --- Title ---
            participant = os.path.basename(input_path)
            fig.suptitle(
                f"{participant} | {current_time.strftime('%H:%M')} - {window_end.strftime('%H:%M')}",
                fontsize=14,
                fontweight='bold'
            )

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            pdf.savefig(fig)
            plt.close(fig)

            current_time = window_end

def ensure_output_directory(output_path):
    pass

def main():
    input_path, output_path = parse_arguments()

    flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path = validate_input_path(input_path)

    nasal, flow_events, thorac, spo2, sleep_profile = load_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path)

    nasal, thorac, spo2, flow_events, sleep_profile = clean_data(nasal, thorac, spo2, flow_events, sleep_profile)

    participant = os.path.basename(input_path)
    output_pdf = os.path.join(output_path, f"{participant}_visualization.pdf")
    generate_visualization(nasal, thorac, spo2, flow_events, output_pdf, input_path)


if __name__ == "__main__":
    main()