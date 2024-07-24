import os
import librosa
import soundfile as sf

# Define the directories
data_dir = './converted_war'
resampled_dir = './resampled_data'

# Create the resampled directory if it doesn't exist
os.makedirs(resampled_dir, exist_ok=True)


# Function to resample audio files
def resample_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_resampled, target_sr


# Iterate over all files in the data directory
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            resampled_audio, sr = resample_audio(file_path)

            # Create the target subdirectory structure
            relative_path = os.path.relpath(root, data_dir)
            target_dir = os.path.join(resampled_dir, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            # Save the resampled audio with _resampled suffix
            resampled_file_path = os.path.join(target_dir, file.replace('.wav', '_resampled.wav'))
            sf.write(resampled_file_path, resampled_audio, sr)

print("Resampling complete.")
