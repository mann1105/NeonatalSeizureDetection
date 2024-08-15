import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# sampling frequency and filter parameters
fs = 32
f_high = 0.5  # highpass cutoff frequency
nyquist_freq = 0.5 * fs
cut_off = f_high / nyquist_freq  # normalized cutoff frequency
order = 4

def filtering(signal):
    """
    Applies a highpass filter and min-max scaling to the input signal.

    Args:
        signal (np.ndarray): The input signal.

    Returns:
        np.ndarray: The filtered and scaled signal.
    """

    b, a = butter(order, cut_off, btype='high')
    filtered_signal = filtfilt(b, a, signal)

    scaled_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))

    return scaled_signal

def process_row(row):
    """
    Processes a single row of the DataFrame, applying filtering and scaling to each channel.

    Args:
        row (pd.Series): A row of the DataFrame containing EEG data and annotation.

    Returns:
        np.ndarray: The processed row with filtered and scaled EEG data, and the original annotation.
    """

    annotation = row[-1]
    eeg_data = np.array(row[:-1]).reshape(21, 32)  
    filtered_eeg_data = np.array([filtering(channel) for channel in eeg_data])

    return np.append(filtered_eeg_data.flatten(), annotation)

df = pd.read_csv('downsampled8.csv', index_col=0)
df.reset_index(drop=True, inplace=True)

df_new = df.apply(process_row, axis=1)
filtered_df = pd.DataFrame(df_new.tolist())  

filtered_df.to_csv('filtered_df8sec.csv', index=False)
