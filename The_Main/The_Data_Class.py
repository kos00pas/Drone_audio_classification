import numpy as np
import datetime
import os
import csv
import wave
import pyaudio

class Data_Class:    #   -> database

    def __init__(self,resp4):
        self.resp4 = resp4
        self.time_initial=datetime.datetime.now()

        """Refresh time for axis of gui"""
        self.doa_refresh = 0.15
        self.signal_refresh = 220
        self.loaded_signal_refresh = 220
        self.loaded_checking_time=0.45
        self.seconds_signal_plotting = 5

        """Limits"""
        self.max_radius = 15 #manimum radius that we can Localize drone
        self.tdoa_data_limit = 1024 * 100 #the amount of data that tdoa will get
        self.chunks_per_second = self.resp4.RESPEAKER_RATE // self.resp4.CHUNK #16000/1024

        """Some initializations"""
        (self.c4,self.audio_line,self.data_loaded,self.fs_loaded,self.time_initial,self.main_window,
         self.mfcc_folder_name,self.param_csv_output_filename) = (None,)*8
        self.c0_buff= []
        self.all_c4_data= []
        self.last_second_c4= []
        self.frames= []
        self.doa_log= []
        self.drone_log= []
        self.all_the_parameters= []
        self.last_second_c0= []
        """Mutex or enable variables"""
        self.recording = False
        self.listening=False
        self.gcc_doa =False ; self.cross_Cor =False
        self.loaded=False ; self.play_loaded_now=False
        self.open_param_window = True
        self.save_doa_enable=True ; self.save_recording_enable= True
        self.save_drone_enable=True ; self.save_param_enable=True ; self.save_mfcc_enable=True

        self.audio_data = np.array( [] )
        self.text_algo= " "

        """Functions calling"""
        self._2d_name()


    def save_doa_param_folder(self):
        base_directory = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        # Get the current date and time
        current_datetime = datetime.datetime.now()
        date_folder_name = current_datetime.strftime("%Y-%m-%d")  # Folder name for the current day
        self.time_folder_name = current_datetime.strftime("%H-%M-%S-%f")  # Time-specific folder name

        # Create a subdirectory for the day inside "Saved_MFcc_params" if it doesn't already exist
        daily_doa_param_directory = os.path.join(base_directory, "Saved_DOA_n_Params", date_folder_name)
        os.makedirs(daily_doa_param_directory, exist_ok=True)

        # Create a subdirectory with the time-specific name inside the day's folder
        specific_doa_param_directory = os.path.join(daily_doa_param_directory, self.time_folder_name)
        os.makedirs(specific_doa_param_directory, exist_ok=True)


        # Save parameters (assuming save_parameters is a method that handles saving some parameters to file)
        self.param_csv_output_filename = os.path.join(specific_doa_param_directory, "Parameters_" + self.time_folder_name+ ".csv")
        self.save_parameters()

        # Save parameters (assuming save_parameters is a method that handles saving some parameters to file)
        self.doa_csv_output_filename = os.path.join(specific_doa_param_directory, "DOA_" + self.time_folder_name + ".csv")
        self.save_doas()


    def _2d_name(self):
        self.text_algo="Algo-> "
        if self.gcc_doa:
            self.text_algo+= "GCC "
        if self.cross_Cor:
            self.text_algo += "cc "
        self.text_algo += ": "

    def add_drone_to_log(self,t,x,y,z,yaw):
        self.drone_log.append({'x':x,'y':y,'z':z,'yaw':yaw,'t':t})

    def add_doa_to_log(self, value):
        timestamp = datetime.datetime.now()
        self.doa_log.append({"value": value, "timestamp": timestamp})

    def The_Saves_after_recording(self):
        # Use os.path.abspath to get the absolute path of the directory where the script is running
        base_directory = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        # Get the current date and time
        current_datetime = datetime.datetime.now()
        date_folder_name = current_datetime.strftime("%Y-%m-%d")  # Folder name for the current day
        time_folder_name = current_datetime.strftime("%H-%M-%S-%f")  # Time-specific folder name

        # Create a subdirectory for the day inside "loaded_Recordings" if it doesn't already exist
        daily_recordings_directory = os.path.join(base_directory, "Saved_Recordings", date_folder_name)
        os.makedirs(daily_recordings_directory, exist_ok=True)

        # Create a subdirectory with the time-specific name inside the day's folder
        specific_recording_directory = os.path.join(daily_recordings_directory, time_folder_name)
        os.makedirs(specific_recording_directory, exist_ok=True)

        # Adjust the filenames to include "Recording_" and "DOA_" prefixes
        if self.save_recording_enable:
            wave_output_filename = os.path.join(specific_recording_directory, "Recording_" + time_folder_name + ".wav")
            self.save_recording_function(wave_output_filename , time_folder_name)
    def save_recording_function(self,wave_output_filename,time_folder_name):
        # Save the recording
        with wave.open( wave_output_filename,'wb' ) as wf:
            wf.setnchannels( self.resp4.RESPEAKER_CHANNELS )
            wf.setsampwidth( pyaudio.get_sample_size( pyaudio.paInt16 ) )
            wf.setframerate( self.resp4.RESPEAKER_RATE )
            wf.writeframes( b''.join( self.frames ) )
        self.main_window.saved_to_label.config( text="Saved to: "+time_folder_name )
        print( f"Recording saved to {time_folder_name}" )
    def save_doas(self):
        # Save the DOA log to CSV
        with open(self.doa_csv_output_filename, 'w', newline='') as csv_file:
            fieldnames = ['value','timestamp']  # Define CSV column headers
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()  # Write the header row

            for doa_entry in self.doa_log:
                # Write each DOA log entry to the CSV
                csv_writer.writerow({'value': doa_entry['value'],'timestamp': doa_entry['timestamp']})

        print(f"DOAs saved to {self.time_folder_name}")


    def save_parameters(self):
        self.append_parameters()
        the_names = ['TimeStamp']  # Include TimeStamp as the first header
        for param in self.resp4.PARAMETERS_names:
            the_names.append(param)  # Append each parameter name

        # Opening the file with 'w' mode to write headers initially
        with open(self.param_csv_output_filename, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=the_names)
            csv_writer.writeheader()  # Write the header row
        # Write all the accumulated data to the CSV file
            for entry in self.all_the_parameters:
                # Create a dictionary from each entry for csv.DictWriter
                data_dict = {name: value for name, value in zip(the_names, entry)}
                csv_writer.writerow(data_dict)


        self.all_the_parameters = []

    def append_parameters(self):
        print('append_parameters')
        now = datetime.datetime.now()
        valueees = [str(now)] + [self.resp4.read_param(param_string=f'{i}') for i in self.resp4.PARAMETERS_names]
        # print(valueees)
        self.all_the_parameters.append(valueees)

