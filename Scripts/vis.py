import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import logging
import os
import glob

from utils.cleaning import clean_data
from utils.io import parse_arguments, validate_input_path, load_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
 

# ^ generating visualization and saving it as pdf
def generate_visualization(flow_events, thorac, spo2, sleep_profile, nasal, output_path, input_path):
    window = pd.Timedelta(minutes=5)

    start_time = max(nasal.index[0], thorac.index[0], spo2.index[0])
    end_time = min(nasal.index[-1], thorac.index[-1], spo2.index[-1])

    current_time = start_time

    with PdfPages(output_path) as pdf:
        
        # ======================================
        #  MACRO FULL TIMELINE PAGE (FIRST PAGE)
        # ======================================

        fig_macro, axs_macro = plt.subplots(
            3, 1,
            figsize=(20, 8),
            sharex=True
        )

        ax_flow_m, ax_thorac_m, ax_spo2_m = axs_macro

        # Remove internal padding completely
        fig_macro.subplots_adjust(
            left=0.05,
            right=0.995,
            top=0.92,
            bottom=0.08,
            hspace=0.15
        )

        # Downsample to 1-second resolution for the macro overview
        # (reduces point count from ~millions to ~hours×3600, much faster & cleaner)
        nasal_macro  = nasal.resample("2s").mean()
        thorac_macro = thorac.resample("2s").mean()
        spo2_macro   = spo2.resample("2s").mean()
        sleep_macro = sleep_profile.resample("30s").ffill()
        
        fig_macro, axs_macro = plt.subplots(
            4, 1,
            figsize=(32, 12),
            sharex=True
        )

        ax_flow_m, ax_thorac_m, ax_spo2_m, ax_sleep_m = axs_macro
        
        # Convert stage labels to numeric
        stage_map = {
            "Wake": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "REM": 4
        }

        sleep_macro['StageNum'] = sleep_macro['Value'].map(stage_map)

        ax_sleep_m.step(
            sleep_macro.index,
            sleep_macro['StageNum'],
            where='post',
            linewidth=2,
            color='black'
        )

        ax_sleep_m.set_ylabel("Sleep Stage")
        ax_sleep_m.set_yticks(list(stage_map.values()))
        ax_sleep_m.set_yticklabels(list(stage_map.keys()))

        # Plot signals (resampled)
        ax_flow_m.plot(nasal_macro.index, nasal_macro['Value'],
                       color='tab:blue',
                       linewidth=1.2,
                       label="Nasal Flow")

        ax_thorac_m.plot(thorac_macro.index, thorac_macro['Value'],
                         color='tab:orange',
                         linewidth=1.2,
                         label="Thoracic/Abdominal Resp.")

        ax_spo2_m.plot(spo2_macro.index, spo2_macro['Value'],
                       color='tab:green',
                       linewidth=1.2,
                       label="SpO₂")

        # Remove x margins (important!)
        for ax in axs_macro:
            ax.margins(x=0)
            ax.set_xlim(nasal.index[0], nasal.index[-1])
            ax.grid(True, alpha=0.15)

        # Labels
        ax_flow_m.set_ylabel("Nasal Flow (L/min)")
        ax_thorac_m.set_ylabel("Resp. Amplitude")
        ax_spo2_m.set_ylabel("SpO₂ (%)")
        ax_spo2_m.set_xlabel("Time")

        ax_flow_m.legend(loc="upper right")
        ax_thorac_m.legend(loc="upper right")
        ax_spo2_m.legend(loc="upper right")

        # Highlight abnormalities across all 3 signal subplots
        for _, row in flow_events.iterrows():

            if row['Disorder'] == "Obstructive Apnea":
                color = 'red'
            elif row['Disorder'] == "Hypopnea":
                color = 'yellow'
            else:
                color = 'purple'

            for ax_m in (ax_flow_m, ax_thorac_m, ax_spo2_m):
                ax_m.axvspan(
                    row['start_time'],
                    row['end_time'],
                    color=color,
                    alpha=0.3
                )

        # Clean hour-based ticks
        locator = mdates.HourLocator(interval=1)
        formatter = mdates.DateFormatter('%H:%M')

        ax_spo2_m.xaxis.set_major_locator(locator)
        ax_spo2_m.xaxis.set_major_formatter(formatter)

        participant = os.path.basename(input_path)

        fig_macro.suptitle(
            f"{participant} - Full Night Overview",
            fontsize=18,
            fontweight='bold'
        )

        pdf.savefig(fig_macro)
        plt.close(fig_macro)

        while current_time < end_time:

            window_end = current_time + window

            seg_flow = nasal.loc[current_time:window_end]
            seg_thorac = thorac.loc[current_time:window_end]
            seg_spo2 = spo2.loc[current_time:window_end]

            fig, axs = plt.subplots(3, 1, figsize=(18, 9), sharex=True)

            ax_flow, ax_thorac, ax_spo2 = axs

            # =========================
            # 1️⃣ BOLD SIGNAL LINES
            # =========================

            ax_flow.plot(seg_flow.index, seg_flow['Value'],
                         color='tab:blue',
                         linewidth=1.8,
                         label="Nasal Flow")

            ax_thorac.plot(seg_thorac.index, seg_thorac['Value'],
                           color='tab:orange',
                           linewidth=1.8,
                           label="Thoracic/Abdominal Resp.")

            ax_spo2.plot(seg_spo2.index, seg_spo2['Value'],
                         color='tab:green',
                         linewidth=1.8,
                         label="SpO₂")

            # =========================
            # 2️⃣ Proper Y Labels + Legends
            # =========================

            ax_flow.set_ylabel("Nasal Flow (L/min)")
            ax_flow.legend(loc="upper right")

            ax_thorac.set_ylabel("Resp. Amplitude")
            ax_thorac.legend(loc="upper right")

            ax_spo2.set_ylabel("SpO₂ (%)")
            ax_spo2.legend(loc="upper right")
            ax_spo2.set_xlabel("Time")

            # =========================
            # 3️⃣ Grid + Limits
            # =========================

            for ax in axs:
                ax.grid(True, alpha=0.2)
                ax.set_xlim(current_time, window_end)

            # =========================
            # 4️⃣ Highlight ONLY Nasal Plot
            # =========================

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

                # Highlight all 3 signal subplots
                for ax in (ax_flow, ax_thorac, ax_spo2):
                    ax.axvspan(event_start, event_end,
                               color=color,
                               alpha=0.3)

                # =========================
                # 5️⃣ Event Name On Top (nasal only)
                # =========================

                ax_flow.text(
                    event_start + (event_end - event_start) / 2,
                    ax_flow.get_ylim()[1] * 0.9,
                    row['Disorder'],
                    ha='center',
                    fontsize=8,
                    fontweight='bold'
                )

            # =========================
            # 6️⃣ Clean Time Axis (10s spacing)
            # =========================

            # Major ticks every 10 seconds
            major_locator = mdates.SecondLocator(interval=10)
            major_formatter = mdates.DateFormatter('%H:%M:%S')

            # Minor ticks every 5 seconds (optional but cleaner)
            minor_locator = mdates.SecondLocator(interval=5)

            ax_spo2.xaxis.set_major_locator(major_locator)
            ax_spo2.xaxis.set_major_formatter(major_formatter)
            ax_spo2.xaxis.set_minor_locator(minor_locator)

            # Darker vertical lines for major ticks
            ax_spo2.grid(True, which='major', axis='x', linewidth=0.9, alpha=0.6)

            # Lighter minor vertical lines
            ax_spo2.grid(True, which='minor', axis='x', linewidth=0.5, alpha=0.3)

            # Optional: rotate labels slightly if crowded
            plt.setp(ax_spo2.xaxis.get_majorticklabels(), rotation=45)

            # =========================
            # 7️⃣ Proper Title Format
            # =========================

            participant = os.path.basename(input_path)

            fig.suptitle(
                f"{participant} - {current_time.strftime('%Y-%m-%d %H:%M')} "
                f"to {window_end.strftime('%H:%M')}",
                fontsize=14,
                fontweight='bold'
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            pdf.savefig(fig)
            plt.close(fig)

            current_time = window_end

def ensure_output_directory(output_path):
    pass

# * ---------- Wrapper function -------------
# ^ enforcing strict user file input path for data
def initialize_paths():
    input_path, output_path = parse_arguments(default_output_path=r"Visualizations/")
    logger.info(f"Arguments parsed — input: {input_path} | output: {output_path}")
    return input_path, output_path


# ^ validating given input path
def path_validation(input_path):
    flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path = validate_input_path(input_path)
    logger.info("Input path validated")
    return flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path

    
# ^ loading data 
def get_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path):
    logger.info("Loading data from the given input path...")
    flow_events, thorac, spo2, sleep_profile, nasal = load_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path)
    logger.info("Data loaded")
    return flow_events, thorac, spo2, sleep_profile, nasal 


# ^ clean data and make it ready for visulization
def preprocessing_data(flow_events, thorac, spo2, sleep_profile, nasal):
    logger.info("Data preprocessing started...")
    flow_events, thorac, spo2, sleep_profile, nasal = clean_data(flow_events, thorac, spo2, sleep_profile, nasal)
    logger.info("Data preprocessing completed")
    return flow_events, thorac, spo2, sleep_profile, nasal

def main():
    
    try:
        logger.info("Pipeline started...")
        input_path, output_path = initialize_paths()
        flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path = path_validation(input_path)
        flow_events, thorac, spo2, sleep_profile, nasal = get_data(flow_event_path, thorac_path, spo2_path, sleep_profile_path, nasal_path)
        flow_events, thorac, spo2, sleep_profile, nasal = preprocessing_data(flow_events, thorac, spo2, sleep_profile, nasal)

        participant = os.path.basename(input_path)
        output_pdf = os.path.join(output_path, f"{participant}_visualization.pdf")
        generate_visualization(flow_events, thorac, spo2, sleep_profile, nasal, output_pdf, input_path)
        
        logger.info("Pipeline completed successfully")
        
    except Exception:
        logger.exception("Pipeline failed")
        

if __name__ == "__main__":
    main()