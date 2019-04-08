"""
________________________________________________

LOUDSPEAKER SELECTION INTERFACE
University of Southampton
Institute of Sound and Vibration Research

Programmed by Charlie House - July 2018
Contact: c.house@soton.ac.uk
________________________________________________
This script is a GUI for a demo of the second year loudspeakers at the ISVR. It is designed to run on a Raspberry Pi using the small 8 output
soundcard, and allows the user to seamlessly switch between the different loudspeakers to hear the differences.

"""

##### IMPORTS #####
from guizero import App,PushButton,Slider,Text,CheckBox,Picture
import numpy as np
import pyaudio
import wave
import scipy.io as sio
import scipy.signal as sig
import pdb

##### <<<<<<<<< VARIABLES >>>>>>>>> #####

audio_gain = np.zeros((3,1))

fs = 44100
output_index = 2 #2 for DVS
output_channels = 6
frame_size = 512

# Set master woofer/tweeter balance
lpf_master = 1
hpf_master = 1

##### <<<<<<<<< LOAD AUDIO DATA >>>>>>>>> #####

# Main Audio Track
wf = wave.open("jazz.wav", 'rb')


##### <<<<<<<<< FUNCTION DEFS >>>>>>>>> #####

# MAIN CALLBACK FUNCTION
def playingCallback(in_data, frame_count, time_info, status):
    data_bytes = wf.readframes(frame_count)                 # Read WAV Data

    if len(data_bytes) == 0 :                               # loop at end of wav
        wf.rewind()

    audio_frame = np.frombuffer(data_bytes,dtype=np.int16)  # Convert Bytes to Numpy Array

    audio_frame = [audio_frame[idx::2] for idx in range(2)]	 # Un-interleave left anf right channels
    
    # pre=create output buffers - zero pads if final frame is not filled
    audio_lpf = np.zeros(frame_size,)
    audio_hpf = np.zeros(frame_size,)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< AUDIO SIGNAL PROCESSING GOES IN HERE >>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Left channel of WAV is LPF, right channel of WAV is HPF
    audio_lpf[:len(audio_frame[0])] = audio_frame[0]
    audio_hpf[:len(audio_frame[1])] = audio_frame[1]

    # Apply Gains
    spkr1_lpf = audio_lpf * audio_gain[0] * lpf_master
    spkr1_hpf = audio_hpf * audio_gain[0] * hpf_master
    spkr2_lpf = audio_lpf * audio_gain[1] * lpf_master
    spkr2_hpf = audio_hpf * audio_gain[1] * hpf_master
    spkr3_lpf = audio_lpf * audio_gain[2] * lpf_master
    spkr3_hpf = audio_hpf * audio_gain[2] * hpf_master



    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  END OF AUDIO SIGNAL PROCESSING   >>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Interleave Ouptuts 
    output = (spkr1_lpf, spkr1_hpf, spkr2_lpf, spkr2_hpf, spkr3_lpf, spkr3_hpf)
    output = np.vstack(output).reshape((-1,), order='F')

    out_data = output.astype(np.int16)
    out_data = out_data.tobytes()
    return out_data, pyaudio.paContinue



# GUI FUNCTIONS

def select_speaker(ind):
    global audio_gain

    if audio_gain[ind] == 0:
    	audio_gain = np.zeros((3,1))
    	audio_gain[ind] =  1
    else:
    	audio_gain = np.zeros((3,1))


##### <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> #####
p = pyaudio.PyAudio()

stream = p.open(format = pyaudio.paInt16, channels = output_channels, rate = fs, output = True, stream_callback=playingCallback,output_device_index=output_index,frames_per_buffer=frame_size)
stream.start_stream()


##### <<<<<<<<< CONFIGURE GUI >>>>>>>>> ##### Header Section
playback_app = App(title="Playback App",bg = (255, 255, 255),layout='grid')
annotation = Text(playback_app,text="",size=30,grid=[0,0,1,1])
isvrlogo = Picture(playback_app, image="isvrlogo.jpg",grid=[0,1,2,1])
isvrlogo.height = 60
isvrlogo.width = 170
annotation = Text(playback_app,text="Choose a Speaker",size=30,grid=[2,1,3,1],align='right')
annotation = Text(playback_app,text=" Press Twice to Turn Off ",size=20,grid=[2,2,3,1])
uoslogo = Picture(playback_app, image="uoslogo.jpg",grid=[6,1,2,1],align='right')
uoslogo.height = 60
uoslogo.width = 210

annotation = Text(playback_app,text="         ",size=30,grid=[1,2,1,1])

annotation = Text(playback_app,text="   ",size=30,grid=[0,3])
spkr1 = PushButton(playback_app,command=select_speaker,args=[0],image="spkr1.jpg",grid=[1,3])
# annotation = Text(playback_app,text=" ",size=30,grid=[2,3])
spkr2 = PushButton(playback_app,command=select_speaker,args=[1],image="spkr2.jpg",grid=[3,3])
# annotation = Text(playback_app,text=" ",size=30,grid=[4,3])
spkr3 = PushButton(playback_app,command=select_speaker,args=[2],image="spkr3.jpg",grid=[5,3,2,1])


playback_app.display()
