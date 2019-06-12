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
from guizero import App,PushButton,Slider,Text,CheckBox,Picture,Window,TextBox
import numpy as np
import pyaudio
import wave
import scipy.io as sio
import scipy.signal as sig
import pdb
import subprocess
import re
import os

##### <<<<<<<<< VARIABLES >>>>>>>>> #####

audio_gain = np.zeros((3,1))

fs = 44100
output_index = 2 #2 for DVS
output_channels = 6
frame_size = 512

admin_max = 10                # Number of Times the Admin Button Must be Pressed
admin_count = 0                 # Counter for Admin Window

# Set master woofer/tweeter balance
lpf_master = 1
hpf_master = 1
master_volume = np.load('settings.npy')     # Master Volume

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
def change_master_volume():
    global master_volume
    global audio_gain
    master_volume = slider.value/100.0

    current_speaker = np.where(audio_gain!=0)[0]
    audio_gain[current_speaker] = master_volume
    print(current_speaker)


def select_speaker(ind):
    global audio_gain

    if audio_gain[ind] == 0:
    	audio_gain = np.zeros((3,1))
    	audio_gain[ind] =  1.0*master_volume
    else:
    	audio_gain = np.zeros((3,1))


def open_admin():
    global admin_count
    if admin_count < admin_max:
        admin_count+=1
    elif admin_count==admin_max:
        admin_window.show()
        updateTemp()
        slider.value = master_volume*100
    else:
        admin_count=0
    # user_window.destroy()

def close_admin():
    global admin_count
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    np.save('settings.npy',master_volume)
    admin_window.hide()

def updateTemp():
    print('temp_check')
    proc = subprocess.Popen(['vcgencmd', 'measure_temp'], stdout=subprocess.PIPE)
    tempString = str(proc.communicate()[0])
    temp = re.findall('\D*(\d*.\d)',tempString)[0]
    tempText.value = 'CPU = ' + temp + 'deg C'

def kill_process():
    os.system('killall python3')

##### <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> #####
p = pyaudio.PyAudio()

stream = p.open(format = pyaudio.paInt16, channels = output_channels, rate = fs, output = True, stream_callback=playingCallback,output_device_index=output_index,frames_per_buffer=frame_size)
stream.start_stream()


##### <<<<<<<<< CONFIGURE GUI >>>>>>>>> ##### Header Section
playback_app = App(title="Playback App",bg = (255, 255, 255),layout='grid')
# User Window
user_window = Window(playback_app, title="User",layout='grid')
annotation = Text(user_window,text="",size=30,grid=[0,0,1,1])
isvrlogo = Picture(user_window, image="isvrlogo.png",grid=[0,1,2,1])
isvrlogo.height = 60
isvrlogo.width = 170
annotation = Text(user_window,text="Choose a Speaker",size=30,grid=[2,1,3,1],align='right')
annotation = Text(user_window,text=" Press Twice to Turn Off ",size=20,grid=[2,2,3,1])
uoslogo = Picture(user_window, image="uoslogo.png",grid=[6,1,2,1],align='right')
uoslogo.height = 60
uoslogo.width = 210
uoslogo.when_clicked = open_admin
annotation = Text(user_window,text="         ",size=30,grid=[1,2,1,1])
annotation = Text(user_window,text="   ",size=30,grid=[0,3])
spkr1 = PushButton(user_window,command=select_speaker,args=[0],image="spkr1.png",grid=[1,3])
spkr2 = PushButton(user_window,command=select_speaker,args=[1],image="spkr2.png",grid=[3,3])
spkr3 = PushButton(user_window,command=select_speaker,args=[2],image="spkr3.png",grid=[5,3,2,1])

# Admin Window
admin_window = Window(playback_app, title="Admin",layout='grid')
spacer = Text(admin_window,text="                                                      ",size=10,grid=[1,0])
spacer = Text(admin_window,text="Admin Window",size=30,grid=[2,1,3,1])
spacer = Text(admin_window,text="           ",size=5,grid=[0,2])
spacer = Text(admin_window,text="Select Master Volume",size=15,grid=[2,3,3,1])
slider = Slider(admin_window, command=change_master_volume,start=1,end=100,grid=[2,5,3,1])
slider.value = master_volume*100
# admin_freq1 = TextBox(admin_window,grid=[2,4,3,1])
# admin_freq2 = TextBox(admin_window,grid=[2,5,3,1])
# admin_freq3 = TextBox(admin_window,grid=[2,6,3,1])
# admin_freq4 = TextBox(admin_window,grid=[2,7,3,1])
spacer = Text(admin_window,text="           ",size=5,grid=[2,8,3,1])
# btn_save_freqs = PushButton(admin_window,command=change_freqs,text="Save Frequencies",grid=[2,9,3,1])

spacer = Text(admin_window,text="           ",size=20,grid=[2,10])
spacer = Text(admin_window,text="General Admin",size=15,grid=[2,11,3,1])

spacer = Text(admin_window,text="           ",size=5,grid=[2,12])
tempText = Text(admin_window, text = 'temp',grid=[2,13,3,1])
updateTemp()
btn_close_admin = PushButton(admin_window,command=close_admin,text="Close Admin Page",grid=[2,14])
btn_kill = PushButton(admin_window,command=kill_process,text="Kill Application",grid=[4,14])
spacer = Text(admin_window,text="           ",size=20,grid=[2,15])
txt = Text(admin_window,text="Interface programmed by Charlie House (SPCG, ISVR)",size=10,grid=[2,16,3,1])
txt = Text(admin_window,text="c.house@soton.ac.uk",size=10,grid=[2,17,3,1])

playback_app.display()