import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt




def make_mfcc(file_path, save_path):
    """
    Scope:
        1. Get corresponding data and their details
        2. Normalize data to a reference level
        3. Make the MFCC through librosa library
        4. Save MFCC to a CSV file
    """
    print(f'Making MFCC for {file_path}.....wait')
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Convert the data to a numpy array
    audio_data = data.values.flatten().astype(np.float32)

    # Parameters
    sr = 16000  # Assuming a sample rate of 16000 Hz, update if different

    # Normalize the entire audio to a reference level (e.g., -20 dB)
    audio_data /= np.max(np.abs(audio_data))
    audio_data *= 10 ** (-20 / 20)  # Reference level at -20 dB

    # Compute MFCC
    n_fft = 2048
    hop_length = 512
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length, fmax=8000)

    print(f"Done MFCC for {file_path}")

    # Save MFCC to CSV
    mfcc_df = pd.DataFrame(mfcc)
    mfcc_df.to_csv(save_path, index=False)

    #
    # # Plot the MFCC
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('MFCC')
    # plt.ylabel('MFCC Coefficients')
    # plt.xlabel('Time')
    # plt.tight_layout()
    # plt.show()


def process_directories(root_dir):
    """
    Process each subdirectory in the root directory to find signal.csv
    and create mfcc.csv in the same directory.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'signal.csv':
                file_path = os.path.join(subdir, file)
                save_path = os.path.join(subdir, 'mfcc.csv')
                make_mfcc(file_path, save_path)


# Set the root directory
all_data = './all_data'

# Example usage
process_directories(all_data)