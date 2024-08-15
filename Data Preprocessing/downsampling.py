import os
import pandas as pd
import numpy as np
from scipy.signal import resample

# Hyperparameters
oldFS = 256
newFS = 32
length = 8 * newFS 
stride = length
p = 0.5

def downsample(data, oldFS, newFS):
    """
    Downsamples the EEG data and adjusts annotations accordingly.

    Args:
        data (pd.DataFrame): The input EEG data with annotations.
        oldFS (int): The original sampling frequency.
        newFS (int): The desired sampling frequency.

    Returns:
        pd.DataFrame: The downsampled EEG data with adjusted annotations.
    """
    
    newNumSamples = int((data.shape[0] / oldFS) * newFS)
    newData = pd.DataFrame(resample(data[data.columns[:-1]], newNumSamples))

    annotation_indices = list(range(0, len(data), 8)) 
    annotation = data.annotation.loc[annotation_indices].reset_index(drop=True)
    newData['annotation'] = annotation

    return newData

path = 'eeg_csv/'

babydfs = list()
for file in sorted(os.listdir(path)):
    # print(file)
    df = downsample(pd.read_csv(path + file), oldFS, newFS)

    finaldfs = list()
    for i in range(0, len(df), stride):
        annotation = 0

        # Determine annotation based on majority within the window
        try:
            if df[df.columns[-1]].iloc[i:i+length].value_counts()[1] > int(p * length):
                annotation = 1
        except:
            annotation = 0

        int_dfs = list()
        for j in range(21):  # 21 EEG channels
            window = df[df.columns[j]].iloc[i:i+length]
            int_dfs.append(window)

        int_df = pd.DataFrame(pd.concat(int_dfs, axis=0, ignore_index=True)).T
        int_df['annotation'] = annotation
        finaldfs.append(int_df)

    finaldf = pd.concat(finaldfs, axis=0)
    babydfs.append(finaldf)

babydf = pd.concat(babydfs, axis=0).reset_index(drop=True).dropna(how='any')
babydf.to_csv('downsampled8.csv')
