import os
from pydub import AudioSegment


def convert_all_mp3_to_wav(input_folder_path, output_folder_name="converted"):
    # Create the output folder if it does not exist
    output_folder_path = os.path.join(input_folder_path, output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if filename.endswith(".mp3"):
            mp3_file_path = os.path.join(input_folder_path, filename)
            wav_file_path = os.path.join(output_folder_path, filename.replace(".mp3", ".wav"))

            # Load the MP3 file
            audio = AudioSegment.from_mp3(mp3_file_path)

            # Export as WAV file
            audio.export(wav_file_path, format="wav")
            print(f"Converted {filename} to WAV in {output_folder_path}")


# Example usage
convert_all_mp3_to_wav("data")
