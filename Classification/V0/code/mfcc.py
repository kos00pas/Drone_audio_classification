import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd

# Function to create MFCC and plot the image
def create_and_plot_mfcc(signal, sr):
    audio = signal.astype(np.float32)

    frame_duration = 0.02
    frame_length = int(frame_duration * sr)
    normalized_frames = []

    # Normalize each frame by its maximum amplitude
    for i in range(0, len(audio), frame_length):
        frame = audio[i:i + frame_length]
        max_amp = np.max(np.abs(frame))
        normalized_frame = frame / max_amp if max_amp > 0 else frame
        normalized_frames.append(normalized_frame)

    normalized_audio = np.concatenate(normalized_frames)
    n_fft = 2048
    if len(normalized_audio) < n_fft:
        normalized_audio = np.pad(normalized_audio, (0, n_fft - len(normalized_audio)), mode='constant')

    mfcc = librosa.feature.mfcc(y=normalized_audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512, fmax=8000)

    # Plot MFCC
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


# Load the signal data
signal_data = pd.read_csv('signal.csv', header=None).values.flatten()

# Create and plot MFCC
create_and_plot_mfcc(signal_data, sr=16000)
