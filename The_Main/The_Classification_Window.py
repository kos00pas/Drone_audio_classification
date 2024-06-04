import os
import threading
import time
import numpy as np
import librosa
import librosa.display

class Classification_Window:
    def __init__(self, main_window, database):
        self.main_window = main_window
        self.DATA = database
        self.close_mfcc_event = threading.Event()  # Initialize the event

    def make_mfcc(self):
        print('making mfcc.....wait')
        time_start = time.time()
        last_second_data = np.concatenate(self.DATA.last_second_c0)
        audio = last_second_data.astype(np.float32)
        frame_duration = 0.02
        sr = self.DATA.resp4.RESPEAKER_RATE
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
        self.main_window.axis_mfcc.clear()
        librosa.display.specshow(mfcc, x_axis='time', ax=self.main_window.axis_mfcc)
        self.main_window.axis_mfcc.set_ylabel('MFCC')

        # Create the directory based on the current time
        the_now_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "..", "MFCC_SIGNAL_saves", the_now_time)
        os.makedirs(save_dir, exist_ok=True)

        # Save the MFCC and signal to CSV
        self.Save_mfcc_csv(mfcc, save_dir)
        self.Save_signal_csv(normalized_audio, save_dir)

    def Save_signal_csv(self, normalized_audio, save_dir):
        full_path = os.path.join(save_dir, "signal.csv")
        np.savetxt(full_path, normalized_audio, delimiter=',')
        print(f'Saved: {full_path}')

    def Save_mfcc_csv(self, mfcc, save_dir):
        full_path = os.path.join(save_dir, "mfcc.csv")
        np.savetxt(full_path, mfcc, delimiter=',')
        print(f'Saved: {full_path}')
