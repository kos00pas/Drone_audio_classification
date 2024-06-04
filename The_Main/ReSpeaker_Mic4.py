import usb
import pyaudio
import struct
import numpy as np

USAGE = """Usage: python {} -h
        -p      show all parameters
        -r      read all parameters
        NAME    get the parameter with the NAME
        NAME VALUE  set the parameter with the NAME and the VALUE
"""
# parameter list
# name: (id, offset, type, max, min , r/w, info)
PARAMETERS = {
    'AECFREEZEONOFF': (18, 7, 'int', 1, 0, 'rw', 'Adaptive Echo Canceler updates inhibit.', '0 = Adaptation enabled', '1 = Freeze adaptation, filter only'),
    'AECNORM': (18, 19, 'float', 16, 0.25, 'rw', 'Limit on norm of AEC filter coefficients'),
    'AECPATHCHANGE': (18, 25, 'int', 1, 0, 'ro', 'AEC Path Change Detection.', '0 = false (no path change detected)', '1 = true (path change detected)'),
    'RT60': (18, 26, 'float', 0.9, 0.25, 'ro', 'Current RT60 estimate in seconds'),
    'HPFONOFF': (18, 27, 'int', 3, 0, 'rw', 'High-pass Filter on microphone signals.', '0 = OFF', '1 = ON - 70 Hz cut-off', '2 = ON - 125 Hz cut-off', '3 = ON - 180 Hz cut-off'),
    'RT60ONOFF': (18, 28, 'int', 1, 0, 'rw', 'RT60 Estimation for AES. 0 = OFF 1 = ON'),
    'AECSILENCELEVEL': (18, 30, 'float', 1, 1e-09, 'rw', 'Threshold for signal detection in AEC [-inf .. 0] dBov (Default: -80dBov = 10log10(1x10-8))'),
    'AECSILENCEMODE': (18, 31, 'int', 1, 0, 'ro', 'AEC far-end silence detection status. ', '0 = false (signal detected) ', '1 = true (silence detected)'),
    'AGCONOFF': (19, 0, 'int', 1, 0, 'rw', 'Automatic Gain Control. ', '0 = OFF ', '1 = ON'),
    'AGCMAXGAIN': (19, 1, 'float', 1000, 1, 'rw', 'Maximum AGC gain factor. ', '[0 .. 60] dB (default 30dB = 20log10(31.6))'),
    'AGCDESIREDLEVEL': (19, 2, 'float', 0.99, 1e-08, 'rw', 'Target power level of the output signal. ', '[-inf .. 0] dBov (default: -23dBov = 10log10(0.005))'),
    'AGCGAIN': (19, 3, 'float', 1000, 1, 'rw', 'Current AGC gain factor. ', '[0 .. 60] dB (default: 0.0dB = 20log10(1.0))'),
    'AGCTIME': (19, 4, 'float', 1, 0.1, 'rw', 'Ramps-up / down time-constant in seconds.'),
    'CNIONOFF': (19, 5, 'int', 1, 0, 'rw', 'Comfort Noise Insertion.', '0 = OFF', '1 = ON'),
    'FREEZEONOFF': (19, 6, 'int', 1, 0, 'rw', 'Adaptive beamformer updates.', '0 = Adaptation enabled', '1 = Freeze adaptation, filter only'),
    'STATNOISEONOFF': (19, 8, 'int', 1, 0, 'rw', 'Stationary noise suppression.', '0 = OFF', '1 = ON'),
    'GAMMA_NS': (19, 9, 'float', 3, 0, 'rw', 'Over-subtraction factor of stationary noise. min .. max attenuation'),
    'MIN_NS': (19, 10, 'float', 1, 0, 'rw', 'Gain-floor for stationary noise suppression.', '[-inf .. 0] dB (default: -16dB = 20log10(0.15))'),
    'NONSTATNOISEONOFF': (19, 11, 'int', 1, 0, 'rw', 'Non-stationary noise suppression.', '0 = OFF', '1 = ON'),
    'GAMMA_NN': (19, 12, 'float', 3, 0, 'rw', 'Over-subtraction factor of non- stationary noise. min .. max attenuation'),
    'MIN_NN': (19, 13, 'float', 1, 0, 'rw', 'Gain-floor for non-stationary noise suppression.', '[-inf .. 0] dB (default: -10dB = 20log10(0.3))'),
    'ECHOONOFF': (19, 14, 'int', 1, 0, 'rw', 'Echo suppression.', '0 = OFF', '1 = ON'),
    'GAMMA_E': (19, 15, 'float', 3, 0, 'rw', 'Over-subtraction factor of echo (direct and early components). min .. max attenuation'),
    'GAMMA_ETAIL': (19, 16, 'float', 3, 0, 'rw', 'Over-subtraction factor of echo (tail components). min .. max attenuation'),
    'GAMMA_ENL': (19, 17, 'float', 5, 0, 'rw', 'Over-subtraction factor of non-linear echo. min .. max attenuation'),
    'NLATTENONOFF': (19, 18, 'int', 1, 0, 'rw', 'Non-Linear echo attenuation.', '0 = OFF', '1 = ON'),
    'NLAEC_MODE': (19, 20, 'int', 2, 0, 'rw', 'Non-Linear AEC training mode.', '0 = OFF', '1 = ON - phase 1', '2 = ON - phase 2'),
    'SPEECHDETECTED': (19, 22, 'int', 1, 0, 'ro', 'Speech detection status.', '0 = false (no speech detected)', '1 = true (speech detected)'),
    'FSBUPDATED': (19, 23, 'int', 1, 0, 'ro', 'FSB Update Decision.', '0 = false (FSB was not updated)', '1 = true (FSB was updated)'),
    'FSBPATHCHANGE': (19, 24, 'int', 1, 0, 'ro', 'FSB Path Change Detection.', '0 = false (no path change detected)', '1 = true (path change detected)'),
    'TRANSIENTONOFF': (19, 29, 'int', 1, 0, 'rw', 'Transient echo suppression.', '0 = OFF', '1 = ON'),
    'VOICEACTIVITY': (19, 32, 'int', 1, 0, 'ro', 'VAD voice activity status.', '0 = false (no voice activity)', '1 = true (voice activity)'),
    'STATNOISEONOFF_SR': (19, 33, 'int', 1, 0, 'rw', 'Stationary noise suppression for ASR.', '0 = OFF', '1 = ON'),
    'NONSTATNOISEONOFF_SR': (19, 34, 'int', 1, 0, 'rw', 'Non-stationary noise suppression for ASR.', '0 = OFF', '1 = ON'),
    'GAMMA_NS_SR': (19, 35, 'float', 3, 0, 'rw', 'Over-subtraction factor of stationary noise for ASR. ', '[0.0 .. 3.0] (default: 1.0)'),
    'GAMMA_NN_SR': (19, 36, 'float', 3, 0, 'rw', 'Over-subtraction factor of non-stationary noise for ASR. ', '[0.0 .. 3.0] (default: 1.1)'),
    'MIN_NS_SR': (19, 37, 'float', 1, 0, 'rw', 'Gain-floor for stationary noise suppression for ASR.', '[-inf .. 0] dB (default: -16dB = 20log10(0.15))'),
    'MIN_NN_SR': (19, 38, 'float', 1, 0, 'rw', 'Gain-floor for non-stationary noise suppression for ASR.', '[-inf .. 0] dB (default: -10dB = 20log10(0.3))'),
    'GAMMAVAD_SR': (19, 39, 'float', 1000, 0, 'rw', 'Set the threshold for voice activity detection.', '[-inf .. 60] dB (default: 3.5dB 20log10(1.5))'),
    # 'KEYWORDDETECT': (20, 0, 'int', 1, 0, 'ro', 'Keyword detected. Current value so needs polling.'),
    'DOAANGLE': (21, 0, 'int', 359, 0, 'ro', 'DOA angle. Current value. Orientation depends on build configuration.')
}
"""
    ASR Automatic Speech Recognition
    AES Audio Engineering Society
    AEC Acoustic Echo Cancellation
    AGC Automatic Gain Control
    RT60 Estimation for AES refers to the measurement and calculation of the Reverberation Time (RT60) using a method or standard defined by the Audio Engineering Society (AES). 
    FSB Frame Sync Boundary ,ensuring that the data from all microphones is properly aligned and synchronized.

"""

"""
    AECNORM (Limit on norm of AEC filter coefficients):
            --->control the stability of the AEC filter and prevent it from becoming too aggressive
    
    AECSILENCELEVEL (Threshold for signal detection in AEC):
            ---> threshold for signal detection in the AEC system
                    when the AEC considers incoming audio as silence or noise
                --->Lowering this threshold->  more sensitive to very low-amplitude signals -> where faint sounds need to be captured. faint= αχνός 
    
    Over-Subtraction  suppression of the echo,
        --->Trade-Offs: While over-subtraction can more effectively reduce echo, it can also negatively impact the quality of the original audio signal.
                                
            GAMMA_E (Over-subtraction factor of echo - direct and early components):
                    ---> Increasing GAMMA_E can make the AEC more aggressive in removing echo, 
                            but excessive values may lead to artifacts or distorted audio.
            
            GAMMA_ETAIL (Over-subtraction factor of echo - tail components):
                    --->fine-tuning of the AEC's behavior for different parts of the echo signal
        
            GAMMA_ENL (Over-subtraction factor of non-linear echo):
                    --->of non-linear echo components.  used to reduce distortion, impact the effectiveness of this suppression.

            NLATTENONOFF (Non-Linear echo attenuation):
                        ---> enabled or disabled
                              reduce the impact of non-linear echo.


"""

"""
    Direct Component: earliest reflection of the sound
                      This is the initial and strongest part of the echo that arrives shortly after the sound is emitted.  

    Early Components: early part of the echo.
                     These are subsequent reflections that follow the direct component but arrive relatively quickly after it. 

    Tail Component: reflections continue for a longer duration , referred to as the "tail" of the echo.
                    represents the later and often weaker reflections of the sound.
"""
class Communicate_ReSpeaker:
    """
    Communication with ReSpeaker
    :except write
    :except read
    :except close the interface
    """

    TIMEOUT = 100000

    def __init__(self, dev):
        self.dev = dev

    def write(self, name, value):
        try:
            data = PARAMETERS[name]
        except KeyError:
            return

        if data[5] == 'ro':
            raise ValueError('{} is read-only'.format(name))

        id = data[0]

        # 4 bytes offset, 4 bytes value, 4 bytes type
        if data[2] == 'int':
            payload = struct.pack(b'iii', data[1], int(value), 1)
        else:
            payload = struct.pack(b'ifi', data[1], float(value), 0)

        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0, id, payload, self.TIMEOUT)

    def read(self, name):
        try:
            data = PARAMETERS[name]
        except KeyError:
            return

        id = data[0]

        cmd = 0x80 | data[1]
        if data[2] == 'int':
            cmd |= 0x40

        length = 8

        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, id, length, self.TIMEOUT)

        response = struct.unpack(b'ii', response.tobytes())

        if data[2] == 'int':
            result = response[0]
        else:
            result = response[0] * (2.**response[1])

        return result

    def close(self):
        """
        close the interface
        """
        usb.util.dispose_resources(self.dev)

class ReSpeaker_Mic4:
    def __init__(self):

        """some initializations"""
        (self.NLAEC_MODE ,self.AECSILENCELEVEL,
        self.max_input_channels, self.dev, self.p, self.D, self.M, self.FREEZEONOFF,
        self.AGCDESIREDLEVEL, self.PARAMETERS_names, self.microphone_spacing, self.speed_of_sound,
        self.CHUNK, self.RESPEAKER_WIDTH, self.RESPEAKER_CHANNELS, self.RESPEAKER_RATE,
        self.AECFREEZEONOFF, self.AECNORM, self.AECPATHCHANGE, self.GAMMA_ENL, self.GAMMA_ETAIL,
        self.GAMMA_E, self.NLATTENONOFF, self.TRANSIENTONOFF, self.ECHOONOFF, self.HPFONOFF,
        self.GAMMA_NS, self.GAMMA_NN, self.MIN_NN, self.MIN_NS, self.NONSTATNOISEONOFF,
        self.CNIONOFF, self.AGCONOFF, self.STATNOISEONOFF, self.AGCMAXGAIN, self.AGCTIME) = (None,) * 36
        self.param_names_dict = []

        """Functions calling"""
        self.Initial_parameter_set()
        self.RESPEAKER_INDEX= self.find_device_and_channels( )
        self.set_initial_Characterestics( )
        self.set_initial_Parameters( )
        self.set_dict_names()
    def Initial_parameter_set(self):
        ##################
        #GAIN
        ##################
       
        self.AGCTIME=1 # gain set time
        """ Name:AGCTIME 
                        Max-Min:       0.1 - 1 
                        Descriptioon:  how quickly respond to changes in signal level,Ramps-up / down time-constant in seconds
                        lower:         quickly respond ,noticeable and abrupt volume fluctuations 
                        higher:        smoother adjustments may react too slowly to sudden changes in input level 
        """
        # self.AGCMAXGAIN= 32
        """ Name:AGCMAXGAIN 
                        Max-Min:1-1000 , default 30dB = 20log10(31.6)
                        Descriptioon:   Maximum AGC gain factor
                        higher:         useful in lower inputs , reduce risk of noise amplification
        """
        self.AGCDESIREDLEVEL= 0.90
        """ Name:AGCDESIREDLEVEL 
                        Max-Min:1e-08 -> 0.99
                        Descriptioon:   Target power level of the output signal.,
                        lower:          beneficial in noisy engiroments , avoid amplification of backgroud nonoise
                        higher:         more aggressice in ampilfiying input signal
        """
        self.AGCGAIN=28

        self.AGCONOFF= 0
        """ Name:AGCONOFF  
                        Max-Min:        bool
                        Descriptioon:   Automatic Gain Control.
                        0:              input gain at a fixed level ,useful in controlled enviroments , consistent   
                        1:              automatically adjust help in no controlled enviroments
        """

###################################################################################################
        ##################
        #NOISE
        ##################
        self.CNIONOFF = 0
        """ Name:
                        Max-Min:        bool
                        Descriptioon:  Controls the insertion of comfort noise,which is synthetic background noise generated to fill in the auditory gaps when noise suppression algorithms significantly reduce background noise levels.
                        lower:  make the audio feel more 'digital' or unnatural .  also mean a cleaner signal without any artificially added sounds.
                        higher: can make the audio signal feel more natural and less abruptly silent when background noises are heavily suppressed.
        """
        self.FREEZEONOFF =0
        """ Name:
                        Max-Min:        bool
                        Descriptioon: Adaptive Beamformer Freeze On/Off , . "Freezing" these coefficients can be useful in stable environments where the direction of arrival of the primary sound source is constant
                        lower:  Allows the beamformer to continuously adapt to changes in the sound scene, which is beneficial in dynamic environments where the direction of sound sources may change or where the acoustic properties of the room vary over time.
                        higher: stabilize the sound capture quality in environments where the acoustic conditions and sound source location are constant.
        """
        self.STATNOISEONOFF =0
        """ Name:
                        Max-Min:        bool
                        Descriptioon: Stationary Noise Suppression On/Off
                        lower:  in a noisier signal but can preserve more of the original sound details.
                        higher: reducing constant background noises like the hum of an air conditioner.
        """
        self.NONSTATNOISEONOFF =0
        """ Name:
                        Max-Min:       bool
                        Descriptioon: Non-Stationary Noise Suppression On/Off
                        lower:   potentially allowing more dynamic sounds through,
                        higher:  improving clarity in fluctuating noise environments., suppression of non-stationary noise, such as typing
        """
        self.GAMMA_NS = 0
        """ Name:
                        Max-Min:  0-3
                        Descriptioon: Over-Subtraction Factor of Stationary Noise
                        lower:  less aggressive, preserving more of the original signal at the cost of less effective noise reduction.
                        higher: Enhances the aggressiveness , reducing more noise but risking distortion or loss of some speech details.
        """
        self.MIN_NS = 1
        """ Name:
                        Max-Min: 0 -1 '[-inf .. 0] dB (default: -16dB = 20log10(0.15))
                        Descriptioon:  Gain-floor for stationary noise suppression.Minimum Stationary Noise Suppression Gain-Floor w such as constant hums or buzzes in the background. It sets the lowest gain level for suppressing such noises.
                        lower:   allowing the gain to be reduced more significantly. , result in an unnaturally quiet background or the loss of some ambient sounds.
                        higher:  less aggressive. This might be beneficial in environments where background noise is constant but not overly intrusive, as it helps maintain audio naturalness.
        """
        self.GAMMA_NN =0.5
        """ Name:
                        Max-Min: 0-3 
                        Descriptioon: Over-Subtraction Factor of Non-Stationary Noise. min .. max attenuation'
                        lower:   Resulting in a more natural sound but less effective noise reduction.
                        higher:  Increases the reduction of non-stationary noise ,making speech more intelligible in noisy environments but may affect the naturalness of the audio.
        """
        self.MIN_NN = 1
        """ Name:
                        Max-Min: 0-1
                        Descriptioon: Minimum Non-Stationary Noise Suppression Gain-Floor ,  This parameter defines the lowest level to which the system can reduce the gain when attempting to suppress non-stationary noise, such as sudden background noises or transient sounds.
                        lower: lowering the noise floor further, lead to a cleaner signal with less background noise, it may also risk suppressing some desired sounds (like soft speech).
                        higher:  less aggressive noise suppression because the noise floor is raised , n preserve more of the natural ambient sound at the cost of potentially allowing more background noise to pass through.
        """
###################################################################################################
        # FILTERS
        self.HPFONOFF = 3
        """ Name:
                        Descriptioon:  High-Pass Filter On/Off , 
                                         '0 = OFF', '1 = ON - 70 Hz cut-off',
                                          '2 = ON - 125 Hz cut-off', '3 = ON - 180 Hz cut-off
                        
        """
###################################################################################################

        ##################
        # ECHO
        ##################
        self.ECHOONOFF = 0
        """ Name:
                        Max-Min:        bool
                        Descriptioon: Echo Suppression On/Off,reducing or eliminating echo in the audio signal.
                        lower:  allow echo to be more prevalent in the signal.
                        higher:  reducing the presence of echo and potentially improving clarity.
        """
        self.TRANSIENTONOFF= 0
        """ Name:
                        Max-Min:       bool
                        Descriptioon: Controls the suppression of transient echoes, which are short, temporary echoes that can occur in varying acoustic environments.
                        lower:   potentially allowing short bursts of echo to pass through.
                        higher: reducing the impact of sudden echo sounds for clearer audio.
        """
        self.NLATTENONOFF = 0
        """ Name:
                        Max-Min:       bool
                        Descriptioon:  Non-Linear Echo Attenuation On/Off attenuation (reduction) of non-linear echo components, which are echoes that do not have a direct linear correlation with the original signal.
                        lower:  may result in a more natural sound but could allow some echo to remain
                        higher: reducing the presence of complex echoes for clearer communication.
        """
        self.GAMMA_E= 1.75
        """ Name:
                        Max-Min:    0-3
                        Descriptioon: Gamma Echo , Sets the over-subtraction factor for direct and early echo components, controlling how aggressively these echoes are suppressed.
                        lower:reduces the echo suppression aggressiveness, potentially allowing some echo through but preserving more of the original sound quality
                        higher: enhances echo suppression, reducing more echo but possibly affecting the naturalness of the audio.
        """
        self.GAMMA_ETAIL = 1.75
        """ Name:
                        Max-Min:     0-3
                        Descriptioon: Gamma Echo Tail , for tail components of the echo, affecting how lingering echoes are treated.
                        lower:less aggressive suppression of echo tails, which can make the audio feel more natural but may leave some echo.
                        higher: increases the suppression of echo tails, potentially offering clearer audio at the risk of unnatural sound processing.
         """
        self.GAMMA_ENL = 1.45
        """ Name:
                        Max-Min:    0-3
                        Descriptioon: Gamma Non-Linear Echo , aiming to reduce echoes that do not follow a simple path or intensity pattern.
                        lower:  less aggressive, possibly resulting in a more natural sound at the expense of clarity.
                        higher: more aggressive, which can help with clarity but might affect sound quality.
        """
###################################################################################################
        ##################
        # AEC Acoustic Echo Cancellation
        ##################
        self.AECNORM=3
        """ Name: AECNORM
                        Max-Min:     16, 0.25
                        Description: Sets the maximum norm of the AEC filter coefficients, controlling the filter's stability and adaptiveness.
                        lower: Decreasing the value makes the AEC more conservative, potentially reducing its effectiveness in rapidly changing echo conditions but improving stability.
                        higher: Increasing the value allows for more aggressive adaptation by the AEC, which can improve echo cancellation in dynamic environments at the risk of instability.
        """
        self.AECFREEZEONOFF= 0
        """ Name: AECFREEZEONOFF
                        Max-Min:   bool
                        Description: Controls whether the Adaptive Echo Canceler's filter coefficients are updated in response to changing acoustic paths.
                        '0 = Adaptation enabled', '1 = Freeze adaptation, filter only')
                        lower: 0 (Adaptation enabled) - The AEC filter coefficients are continuously updated, allowing the system to adapt to changes in the echo path.
                        higher: 1 (Freeze adaptation) - Updates to the AEC filter coefficients are halted, freezing the current filter state. Useful in stable environments where echo paths do not change.
        """
        self.NLAEC_MODE=0
        """
                       Name: NLAEC_MODE
                       Max-Min: 0, or 1 or 2  
                       Description: Non-Linear AEC training mode
                       '0 = OFF', '1 = ON - phase 1', '2 = ON - phase 2'
        """
        # self.AECSILENCELEVEL=
        """
                     Name: NLAEC_MODE
                     Max-Min: 1, 1e-09,
                     Description: Non-Linear AEC training mode.'Threshold for signal detection in AEC [-inf .. 0] dBov (Default: -80dBov = 10log10(1x10-8))'
                     lower
                     higher
        """


###################################################################################################

    def set_dict_names(self):
        # Extracting keys from self.resp4.PARAMETERS_names and appending to dict_names
        for param in self.PARAMETERS_names:
            self.param_names_dict.append(param)  # Ensure this list captures all your desired keys
    def set_initial_Parameters(self):
        # gain
        self.write_param('AGCTIME',self.AGCTIME)# respond quickly to changes in signal amplitudes, including low-amplitude sounds in dynamic outdoor environments.
        self.write_param('AGCDESIREDLEVEL',self.AGCDESIREDLEVEL) #Lowering the target power level ensures that the AGC will try to amplify signals, even if they are very faint, thus enhancing sensitivity.
        # self.write_param('AGCMAXGAIN',self.AGCMAXGAIN)#gain factor
        self.write_param('AGCONOFF',self.AGCONOFF)#gain factor
        self.write_param('AGCGAIN',self.AGCGAIN)
        #noise
        self.write_param('CNIONOFF', self.CNIONOFF)
        self.write_param('FREEZEONOFF', self.FREEZEONOFF) # never 1 , stucts the doa

        self.write_param('STATNOISEONOFF', self.STATNOISEONOFF)
        self.write_param('NONSTATNOISEONOFF', self.NONSTATNOISEONOFF)

        self.write_param('GAMMA_NS', self.GAMMA_NS)
        self.write_param('MIN_NS', self.MIN_NS)
        self.write_param('GAMMA_NN', self.GAMMA_NN)
        self.write_param('MIN_NN', self.MIN_NN)

        # filters
        self.write_param('HPFONOFF',self.HPFONOFF)#70 Hz cut-off , noises from backgroung
        # Echo
        self.write_param('ECHOONOFF', self.ECHOONOFF)
        self.write_param('TRANSIENTONOFF', self.TRANSIENTONOFF)
        self.write_param('NLATTENONOFF', self.NLATTENONOFF)

        self.write_param('GAMMA_E', self.GAMMA_E)
        self.write_param('GAMMA_ETAIL', self.GAMMA_ETAIL)
        self.write_param('GAMMA_ENL', self.GAMMA_ENL)
        #
        self.write_param('AECNORM', self.AECNORM)
        self.write_param('AECFREEZEONOFF', self.AECFREEZEONOFF)
        self.write_param('NLAEC_MODE', self.NLAEC_MODE)
        # self.write_param('AECSILENCELEVEL', self.AECSILENCELEVEL)

    def set_initial_Characterestics(self):
        self.RESPEAKER_RATE = 16000
        self.RESPEAKER_CHANNELS = 6
        self.RESPEAKER_WIDTH = 2
        self.CHUNK = 1024
        self.speed_of_sound = 343  # Speed of sound in m/s
        self.microphone_spacing=0.0457
        global PARAMETERS
        self.PARAMETERS_names=PARAMETERS
        m1 = np.array([0, 0, 0]) ; m2 = np.array([self.microphone_spacing, 0, 0]) ; m3 = np.array([self.microphone_spacing, self.microphone_spacing, 0]); m4 = np.array([0, self.microphone_spacing, 0])
        self.M=np.array([m1,m2,m3,m4])

        d12=m1-m2 ; d13=m1-m3 ; d14=m1-m4 ; d23=m2-m3 ; d24=m2-m4  ; d34=m3-m4
        self.D=np.array([d12,d13,d14,d23,d24,d34])
    def find_device_and_channels(self):
        self.p = pyaudio.PyAudio( )
        self.dev=usb.core.find( idVendor=0x2886,idProduct=0x0018 )
        target_name = "ReSpeaker 4 Mic Array (UAC1.0)"
        target_index = None

        for i in range( self.p.get_device_count( ) ):
            device_info = self.p.get_device_info_by_index( i )
            device_name = device_info.get( 'name' )
            # print(device_name)

            if target_name in device_name:
                target_index = i
                break  # Exit the loop if a match is found
        if target_index is not None:
            print( f"{target_name}: found at index {target_index}" )
            self.max_input_channels = self.p.get_device_info_by_index(target_index )['maxInputChannels']
            print("Channels: " , self.max_input_channels)
            self.p.terminate()

            return target_index
        else:
            print( f"{target_name}': not found" )
            self.p.terminate()

            return False
    def write_param(self,param,argums):
        if self.dev:
            Mic_tuning = Communicate_ReSpeaker( self.dev )
            Mic_tuning.write(param,argums)
    def read_param(self,the_Self=None,param_string=None):
        if self.dev:
            Mic_tuning = Communicate_ReSpeaker( self.dev )
            try:
                return Mic_tuning.read(param_string)
            except KeyboardInterrupt:
                return None
    @property
    def doa_from_respeaker(self):
        doa= self.read_param(param_string='DOAANGLE')
        return doa