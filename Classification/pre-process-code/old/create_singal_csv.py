import os
import wave
import numpy as np
import shutil

# Define the input and output directories
splitted_dir = './splitted_dir'
all_data = './all_data'

# Function to create directories and save first channel signal data to signal.csv
def create_directories_and_save_signal(src_root, dest_root):
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.wav'):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_root)
                dest_file_path = os.path.join(dest_root, relative_path)

                # Create the directory if it does not exist
                dest_dir_path = os.path.dirname(dest_file_path)
                os.makedirs(dest_dir_path, exist_ok=True)
                print(f'Created directory {dest_dir_path}')

                # Read the .wav file and extract the first channel data
                with wave.open(src_file_path, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    # Read the frames and convert to numpy array
                    frames = wav_file.readframes(n_frames)
                    data = np.frombuffer(frames, dtype=np.int16)
                    data = data.reshape(-1, n_channels)

                    # Extract the first channel
                    c0 = data[:, 0]

                # Check if the signal data is smaller than (16000, 1)
                if c0.shape[0] < 16000:
                    print(f'Signal data is smaller than (16000, 1) for {src_file_path}')
                    # Delete the created directory
                    shutil.rmtree(dest_dir_path)
                    print(f'Deleted directory {dest_dir_path}')
                else:
                    # Save the first channel data to signal.csv
                    signal_csv_path = os.path.join(dest_dir_path, 'signal.csv')
                    np.savetxt(signal_csv_path, c0, delimiter=',', fmt='%d')
                    print(f'Created signal.csv at {signal_csv_path}')

# Call the function
create_directories_and_save_signal(splitted_dir, all_data)
