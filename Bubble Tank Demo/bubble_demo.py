"""
________________________________________________

BUBBLE ABSORPTION DEMO
University of Southampton
Institute of Sound and Vibration Research

Programmed by Charlie House - July 2018
Contact: c.house@soton.ac.uk
________________________________________________
This script is a controller for the bubble absorption demo within the ISVR, University of Southampton. It uses PyAudio to setup
an audio stream in callback mode. This allows a sine wave to be generated in real-time such that the frequency and amplitude can
be varied by a user. This is set to output at 48kHz using the default audio device.

GUIZero (a wrapper for TKInter) is used to generate a GUI for the end-user, with a number of features. One of these is to
enable or disable the bubble generation device. This sends a digital 1 or 0 out of GPIO pin 21, which should then be interfaced
with the bubble generation machine's switch.

"""

##### IMPORTS #####
from guizero import App,PushButton,Slider,Text,Picture,Window,TextBox
import numpy as np
import pyaudio
import time
import math
from itertools import count
import scipy.signal
import sys,os
import RPi.GPIO as GPIO
import subprocess
import re
import os

##### <<<<<<<<< VARIABLES >>>>>>>>> #####

audio_gain = 1                  # Initial Gain
fs = 48000                      # Sampling Frequency
output_index = 1                # Select Audio Output Device
output_channels = 1             # Number of Output Channels
frame_size = 1024               # Side of Buffer in Callback Fnc
last_t = 0                      # Counter used for Sine Wave
freqs = np.load('freqs.npy')     # Vector of Frequencies for Button Selection
freq = 100
gpio_pin = 21			# The GPIO pin used for the bubble pump
admin_max = 10                # Number of Times the Admin Button Must be Pressed
admin_count = 0                 # Counter for Admin Window
##### <<<<<<<<< FUNCTION DEFS >>>>>>>>> #####

# MAIN CALLBACK FUNCTION
def playingCallback(in_data, frame_count, time_info, status):
    global last_t
    # global current_freq

    # Way to Limit last_t increasing indefinitely
    if audio_gain == 0:
        last_t = 0

    # Generate Sine Wave
    t = np.arange(last_t,last_t + frame_size)/fs
    signal = np.sin(2*np.pi*freq*t)

    # Apply Gain
    output = signal * audio_gain
    # Increment counter
    last_t = last_t + frame_size

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  OUTPUT PROCESSING   >>>>>>>>>>>>>>>>>>>>>>>>>>>
    out_data = output.astype(np.float32)
    out_data = out_data.tobytes()
    return out_data, pyaudio.paContinue




# GUI FUNCTIONS
def push_play():
    global admin_count
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    stream.start_stream()
    

def push_stop():
    global admin_count
    global last_t
    stream.stop_stream()
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    last_t = 0     # Way to Limit last_t increasing indefinitely

# def change_freq():
    # global freq
    # global last_t
    # last_t = 0     # Way to Limit last_t increasing indefinitely

def push_freq(num):
    global admin_count
    global freq
    global freqs
    freq = freqs[num]
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    last_t = 0     # Way to Limit last_t increasing indefinitely

def bubbles_on():
    global admin_count
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    GPIO.output(gpio_pin,GPIO.LOW)
    print('bubbles on')

def bubbles_off():
    global admin_count
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    GPIO.output(gpio_pin,GPIO.HIGH)
    print('bubbles off')

def open_admin():
    global admin_count
    if admin_count < admin_max:
        admin_count+=1
    elif admin_count==admin_max:
        admin_window.show()
        updateTemp()
    else:
        admin_count=0
    # user_window.destroy()

def close_admin():
    global admin_count
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    np.save('freqs.npy',freqs)
    admin_window.hide()
    user_window.update()

def change_freqs():
    global admin_count
    global freqs
    admin_count = 0 # Reset Admin Counter When Any OTher Button Pressed
    print(admin_freq1.value)
    print(admin_freq2.value)
    print(admin_freq3.value)
    print(admin_freq4.value)
    freqs[0] = float(admin_freq1.value)
    freqs[1] = float(admin_freq2.value)
    freqs[2] = float(admin_freq3.value)
    freqs[3] = float(admin_freq4.value)
    
    f1_button.Text = admin_freq1.value
    f2_button.Text = admin_freq2.value
    f3_button.Text = admin_freq3.value
    f4_button.Text = admin_freq4.value

def updateTemp():
    proc = subprocess.Popen(['vcgencmd', 'measure_temp'], stdout=subprocess.PIPE)
    tempString = str(proc.communicate()[0])
    temp = re.findall('\D*(\d*.\d)',tempString)[0]
    tempText.value = 'CPU = ' + temp + 'deg C'

def kill_process():
    os.system('pkill python')

##### <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> #####
p = pyaudio.PyAudio()

stream = p.open(
    format = pyaudio.paFloat32, 
    channels = output_channels, 
    rate = fs, 
    output = True, 
    stream_callback=playingCallback,
    output_device_index=output_index,
    frames_per_buffer=frame_size)

stream.stop_stream()

##### <<<<<<<<< SETUP GPIO PINS FOR BUBBLES >>>>>>>>> #####

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(gpio_pin,GPIO.OUT)

GPIO.output(gpio_pin,GPIO.HIGH)
##### <<<<<<<<< CONFIGURE GUI >>>>>>>>> #####

# Header Section
playback_app = App(title="Playback App",bg = (255, 255, 255),layout='grid')
user_window = Window(playback_app, title="User",layout='grid')
isvrlogo = Picture(user_window, image="isvrlogo.jpg",grid=[0,0,2,1])
isvrlogo.height = 50
isvrlogo.width = 150
spacer = Text(user_window,text="          ",size=30,grid=[0,1,1,1])
# annotation = Text(user_window,text="Bubble Demo",size=30,grid=[2,0,2,1])
spacer = Text(user_window,text="             ",size=30,grid=[2,0,2,1])
spacer = Text(user_window,text="    ",size=30,grid=[4,1,1,1])
uoslogo = Picture(user_window, image="uoslogo.jpg",grid=[4,0],align='right')
# uoslogo = PushButton(user_window, command=open_admin,image="uoslogo.jpg",grid=[4,0],align='right')
uoslogo.height = 50
uoslogo.width = 167
uoslogo.when_clicked = open_admin

# Main Section
play_button = PushButton(user_window,command=push_play,text="Tone On",grid=[1,1,2,1],padx=50)
stop_button = PushButton(user_window,command=push_stop,text="Tone Off",grid=[3,1,2,1],padx=50)

spacer = Text(user_window,text="           ",size=30,grid=[0,5,4,1])

spacer = Text(user_window,text="           ",size=30,grid=[0,7,4,1])
f1_button = PushButton(user_window,command=push_freq,args=[0],grid=[1,6],image="button1.jpg",padx = 30,text="Frequency 1")
f2_button = PushButton(user_window,command=push_freq,args=[1],grid=[2,6],image="button2.jpg",padx = 30,text="Frequency 2")
f3_button = PushButton(user_window,command=push_freq,args=[2],grid=[3,6],image="button3.jpg",padx = 30,text="Frequency 3")
f4_button = PushButton(user_window,command=push_freq,args=[3],grid=[4,6],image="button4.jpg",padx = 10,text="Frequency 4",align="left")

spacer = Text(user_window,text="           ",size=30,grid=[0,7,4,1])

bubon_button = PushButton(user_window,command=bubbles_on,image="bubble.jpg",grid=[1,8,2,1],padx=50)
buboff_button = PushButton(user_window,command=bubbles_off,image="nobubble.jpg",grid=[3,8,2,1],padx=50)

# Admin Window
admin_window = Window(playback_app, title="Admin",layout='grid')
spacer = Text(admin_window,text="                                                      ",size=10,grid=[1,0])
spacer = Text(admin_window,text="Admin Window",size=30,grid=[2,1,3,1])
spacer = Text(admin_window,text="           ",size=5,grid=[0,2])
spacer = Text(admin_window,text="Change Frequencies",size=15,grid=[2,3,3,1])
admin_freq1 = TextBox(admin_window,grid=[2,4,3,1])
admin_freq2 = TextBox(admin_window,grid=[2,5,3,1])
admin_freq3 = TextBox(admin_window,grid=[2,6,3,1])
admin_freq4 = TextBox(admin_window,grid=[2,7,3,1])
spacer = Text(admin_window,text="           ",size=5,grid=[2,8,3,1])
btn_save_freqs = PushButton(admin_window,command=change_freqs,text="Save Frequencies",grid=[2,9,3,1])

admin_freq1.value = freqs[0]
admin_freq2.value = freqs[1]
admin_freq3.value = freqs[2]
admin_freq4.value = freqs[3]

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
