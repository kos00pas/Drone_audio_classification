import threading
import numpy as np
import pyaudio
import time
import keyboard

class Signal:
    def __init__(self, main_window, database):
        self.DATA = database  # for access to database
        self.main_window = main_window  # for access to main window

        # some initializations
        self.c0_stop_real_time_var = False
        (self.loaded_stream, self.p_load,
         self.start_time, self.playback_thread, self.update_loaded_signal_thread, self.p_recording,
         self.p_c0, self.stream_in, self.stream_out, self.c0_playback_thread,
         self.first_time, self.control_frame, self.classification_button, self.parameters_button,
         self.stop_button, self.start_button, self.time_of_Start, self.recording_thread,
         self.update_signal_thread, self.tdoa_thread,
         self.canvas, self.fig) = (None,) * 22
        self.stop_event = threading.Event()

    def start_recording(self):
        if not self.DATA.recording:
            self.DATA.recording = True

            print("Start rec, wait")
            self.time_of_Start = time.time()

            self.main_window.listening_button.config(state='normal')

            self.main_window.axis_signal.clear()
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.update_signal_thread = threading.Thread(target=self.update_signal_plot)
            self.recording_thread.start()
            self.update_signal_thread.start()

    def stop_recording(self):
        if self.DATA.recording:
            self.DATA.recording = False

            self.main_window.listening_button.config(state='disabled', text="Start Listen", bg="light yellow")
            self.main_window.make_mfcc_button.config(state='disabled', bg="light yellow")

            self.DATA.The_Saves_after_recording()
            self.stop_event.set()
            self.update_signal_thread.join(timeout=1)
            if self.update_signal_thread.is_alive():
                print("Warning: update_signal_thread did not terminate. Retrying...")
                self.update_signal_thread.join(timeout=3)
            self.c0_stop_real_time()

    def record_audio(self):
        self.p_recording = pyaudio.PyAudio()
        stream = self.p_recording.open(
            rate=self.DATA.resp4.RESPEAKER_RATE,  # 16000
            channels=self.DATA.resp4.RESPEAKER_CHANNELS,  # 6
            format=pyaudio.paInt16,
            input=True,
            input_device_index=self.DATA.resp4.RESPEAKER_INDEX,
            frames_per_buffer=self.DATA.resp4.CHUNK
        )
        self.stream_in = stream  # Assign the input stream to self.stream_in
        try:
            self.DATA.c0_buff = []
            self.DATA.last_second_c0 = []  # Initialize the buffer for last 1 second of c0 data

            self.first_time = True
            self.DATA.frames = []
            while self.DATA.recording:
                data = stream.read(self.DATA.resp4.CHUNK, exception_on_overflow=False)
                self.DATA.frames.append(data)

                channel_data_chunk = np.frombuffer(data, dtype=np.int16)
                channel_data_chunk = channel_data_chunk.reshape(-1, self.DATA.resp4.RESPEAKER_CHANNELS).T
                c0 = channel_data_chunk[0]

                # C0 LAST second data -> mel spectrogram, classification
                self.DATA.last_second_c0.append(c0)
                if len(self.DATA.last_second_c0) > self.DATA.chunks_per_second:
                    self.main_window.make_mfcc_button.config(state='normal', bg="cyan")
                    self.DATA.last_second_c0.pop(0)

                # C0 Signal plotting
                self.DATA.c0_buff.append(c0)
                if len(self.DATA.c0_buff) > self.DATA.seconds_signal_plotting * self.DATA.resp4.RESPEAKER_RATE / self.DATA.resp4.CHUNK:
                    self.DATA.c0_buff.pop(0)

        except Exception as e:
            print(f"An error occurred during recording: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            self.p_recording.terminate()
            print("Recording stopped")

    def update_signal_plot(self):
        if self.DATA.recording and len(self.DATA.c0_buff) > 0:
            audio_data = np.concatenate(self.DATA.c0_buff)
            time_in_seconds = np.arange(len(audio_data)) / self.DATA.resp4.RESPEAKER_RATE
            self.main_window.axis_signal.clear()
            self.main_window.axis_signal.plot(time_in_seconds, audio_data, color='b')
            self.main_window.axis_signal.relim()
            self.main_window.axis_signal.autoscale_view()
            self.main_window.canvas.draw_idle()
            elapsed = time.time() - self.time_of_Start
            elapsed_rounded = round(elapsed, 1)
            self.main_window.signal_label.config(text=f'{elapsed_rounded} seconds')

        if self.DATA.recording:
            self.main_window.after(self.DATA.signal_refresh, self.update_signal_plot)

    def c0_stop_real_time(self):
        self.c0_stop_real_time_var = False

        if self.c0_playback_thread is not None and self.c0_playback_thread.is_alive():
            self.c0_playback_thread.join(timeout=1)
            if self.c0_playback_thread.is_alive():
                print("Warning: c0_playback_thread did not terminate. Retrying...")
                self.c0_playback_thread.join(timeout=3)

        print("Stopped real-time playback.")

    def c0_play_real_time(self):
        if self.c0_playback_thread is None or not self.c0_playback_thread.is_alive():
            self.c0_playback_thread = threading.Thread(target=self.c0_play)
            self.c0_playback_thread.start()
        else:
            print("Playback thread is already running.")

    def c0_play(self):
        if not self.stream_in:  # Ensure stream_in is already open
            print("Error: Input stream not open for playback")
            return

        self.c0_stop_real_time_var = True  # Playback continues until stopped
        self.p_c0 = self.p_recording  # Use the same PyAudio instance

        try:
            # Open the playback (output) stream as mono
            if not self.stream_out:
                self.stream_out = self.p_c0.open(
                    format=pyaudio.paInt16,
                    channels=1,  # Mono output
                    rate=self.DATA.resp4.RESPEAKER_RATE,
                    output=True
                )

            print("Starting real-time playback. Press Escape, Backspace, or Space to stop.")

            while self.DATA.recording and self.c0_stop_real_time_var:
                data = self.stream_in.read(self.DATA.resp4.CHUNK, exception_on_overflow=False)

                # Convert byte data to numpy array for manipulation
                np_data = np.frombuffer(data, dtype=np.int16)
                # Assuming the channels are interleaved, reshape and select the first channel
                channel_data = np_data.reshape(-1, self.DATA.resp4.RESPEAKER_CHANNELS)[:, 0]

                # Convert the extracted channel data back to bytes
                mono_data = channel_data.tobytes()

                self.stream_out.write(mono_data)  # Play back the extracted first channel

                # Check for key presses to stop playback
                if keyboard.is_pressed('esc') or keyboard.is_pressed('backspace') or keyboard.is_pressed('space'):
                    print("Stopping real-time playback by user command.")
                    break  # Exit the loop

        except Exception as e:
            print(f"An error occurred during real-time playback: {e}")
        finally:
            # Ensure streams are closed properly
            try:
                if self.stream_out:
                    self.stream_out.stop_stream()
                    self.stream_out.close()
                    self.stream_out = None
            except Exception as e:
                print(f"Error closing streams: {e}")
            print("Real-time playback and recording stopped.")

    def for_destroy(self):
        self.stop_event.set()

        if hasattr(self, 'DATA'):
            self.DATA.recording = False

        # Close and terminate PyAudio streams
        if self.stream_in is not None:
            try:
                self.stream_in.stop_stream()
                self.stream_in.close()
            except Exception as e:
                print(f"Error closing input stream: {e}")
            self.stream_in = None

        if self.stream_out is not None:
            try:
                self.stream_out.stop_stream()
                self.stream_out.close()
            except Exception as e:
                print(f"Error closing output stream: {e}")
            self.stream_out = None

        # Terminate PyAudio instances
        pyaudio_instances = [self.p_recording, self.p_load, self.p_c0]
        for instance in pyaudio_instances:
            if instance is not None:
                instance.terminate()

        # Join threads, if they exist and are alive
        threads = [self.recording_thread, self.c0_playback_thread, self.update_signal_thread]
        for thread in threads:
            if thread is not None and thread.is_alive():
                try:
                    thread.join(timeout=1)
                    if thread.is_alive():
                        print(f"Warning: {thread.name} did not terminate. Retrying...")
                        thread.join(timeout=3)  # Retry with a longer timeout
                except Exception as e:
                    print(f"Error: Failed to join {thread.name}. Exception: {e}")

        # Set everything to None to release resources
        self.p_recording = self.p_load = self.p_c0 = None
        self.c0_playback_thread = None
