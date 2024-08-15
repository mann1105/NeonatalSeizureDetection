import pandas as pd
import numpy as np
import librosa

data = pd.read_csv("filtered_df8sec.csv")
target = data.iloc[:, -1].values

n_mels = 128  # number of mel bands

mel_spectrogram_features = []
for i in range(len(data)):
    signal = data.iloc[i, :-1].values  
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=32, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    mel_spectrogram_features.append(mel_spectrogram_db)

max_frames = max(mel.shape[1] for mel in mel_spectrogram_features)

X = np.zeros((len(mel_spectrogram_features), n_mels * max_frames))

for i, mel in enumerate(mel_spectrogram_features):
    mel = mel[:, :max_frames] 
    X[i, :] = mel.flatten()

features_df = pd.DataFrame(X)
features_df['target'] = target

features_df.to_csv('finalfeatures.csv', index=False) 