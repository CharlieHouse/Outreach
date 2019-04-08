#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:27:45 2018

@author: ch20g13
"""

##### IMPORTS #####

import pyaudio
import numpy as np
import math
import atexit
import time
from guizero import App,PushButton,Slider,Text,CheckBox
from scipy.io import savemat
import scipy.signal as sig

########## <<<<<<<<< VARIABLES >>>>>>>>> ##########
# Audio Stream Variables
fs = 48000
audio_dev_index = 2
frame_size = 2048

# Global Variables
gain = 1
duration = 30   # Duration of Measurement (in Seconds)
last_t = 0  # Counter used for Sine Wave

########## <<<<<<<<< FUNCTION DEFS >>>>>>>>> ##########

input_recording = np.zeros(((duration+1)*fs))
noise_recording = np.zeros(((duration+1)*fs))
# MAIN CALLBACK FUNCTION
def playingCallback(in_data, frame_count, time_info, status):
    audio_frame_int = np.frombuffer(in_data,dtype=np.float32)  # Convert Bytes to Numpy Array
    global last_t
    global input_recording

    audio_frame = np.reshape(audio_frame_int, (frame_size, 2))

    mic_in = audio_frame[:,0]

    # Record Input Channel 1
    input_recording[last_t : last_t+frame_size] = mic_in

    # Generate Noise Signals
    t = np.arange(last_t,last_t + frame_size)/fs
    noise = np.random.uniform(-1,1,(frame_size,))
    

    # Play & Save Noise Signals
    noise_recording[last_t : last_t+frame_size] = noise*gain

    last_t = last_t+frame_size

    out_mat = (noise*gain,np.zeros(frame_size,))    # Channel1 = Control, Channel2 = Primary
    out_mat = np.vstack(out_mat).reshape((-1,), order='F')


    # Ouput Processing
    out_data = out_mat.astype(np.float32)
    out_data = out_data.tobytes()
    return out_data, pyaudio.paContinue



########## <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> ##########
p = pyaudio.PyAudio()

stream = p.open(format = pyaudio.paFloat32,
    channels = 2,
    rate = fs,
    output = True,
    input = True,
    stream_callback=playingCallback,
    output_device_index=audio_dev_index,
    input_device_index=audio_dev_index,
    frames_per_buffer=frame_size)


########## <<<<<<<<< RUN >>>>>>>>> ##########


def measure_plant():
    global noise_recording
    global input_recording
    print('Measuring Plant Response')

    print('Starting Measurement')
    stream.start_stream()
    time.sleep(duration)

    print('Stopping Measurement')
    stream.stop_stream()

    print('Saving Data')
    savemat('plant_mes.mat', {'mic':input_recording,'noise':noise_recording})
    
measure_plant()

