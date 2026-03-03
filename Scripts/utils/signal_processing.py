import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(df, lowcut=0.1, highcut=0.5, order=4):
    """
    Apply Butterworth bandpass filter to a time-series dataframe.
    Assumes:
        - df.index is datetime
        - df has column 'Value'
    """

    values = df["Value"].values
    time_index = df.index

    sampling_interval = (time_index[1] - time_index[0]).total_seconds()
    sampling_rate = 1 / sampling_interval
    nyquist = 0.5 * sampling_rate

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, values)

    df["Value"] = filtered
    return df