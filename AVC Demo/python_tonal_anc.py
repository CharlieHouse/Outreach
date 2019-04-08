#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:27:45 2018

@author: ch20g13 and djw1g12 @ soton.ac.uk
"""

# <<<<<<<<<< IMPORTS >>>>>>>>>> #
import pyaudio
import numpy as np
import atexit
from scipy.signal import lfilter, butter
from guizero import App, PushButton, Text, Picture, Window, Waffle
import random
import re
import subprocess
import pickle
import os

# <<<<<<<<< VARIABLES >>>>>>>>> #
# Audio Stream Variables
fs = 44100
output_index = 2  # python -m sounddevice
input_index = 2  # python -m sounddevice
frame_size = 4096
chan_count = 2
con_gain = 0
out_gain = 0.2 #0.08 for 104Hz, 0.2 for 25.5

# LMS Variables
gamma = 0.999999999
alpha = 1e-2
adaptive = 0

# Global Variables
freq = 25.2
controlState = 0
eqState = 0
adminCount = 0  # timer for administrative page
thermalTimer = 0  # timer counts up
timerMax = 1*60 #minute thermal timer
#run = -1
micSum = 0

last_t = 0  # Counter used for Sine Wave

# Estimate of Plant Response - uncomment to load last measured
# h = np.load('/home/pi/Documents/AVC/plantdata.npy')
# measured, averaged h, at gain = 0.1. Multiplied by 10 for TF with unit gain
h = -(1.786269421 + 8.214203719j)*10
#h = -(0.011642594960987 + 0.010710587901811j)*1000 #- GOOD SET (25.2)
#h = (0.032754130060299 + 0.053213797611848j)*1000 #- GOOD SET (104)

# Calculate HPF at 20Hz
normal_cutoff = 20 / (0.5 * fs)
bh, ah = butter(4, normal_cutoff, btype='high', analog=False)

# Calculate LPF at 50Hz
normal_cutoff = 50/(0.5*fs)
bl,al = butter(6,normal_cutoff,btype='low',analog=False)


# Initial LMS Coefficients
u = np.zeros((2, 1))
update = np.zeros((2, 1))
r = np.zeros((2, frame_size))

# error_log = np.array([])

# <<<<<<<<< FUNCTION DEFS >>>>>>>>> #

def change_earthquake_text():
    global eqState
    global controlState
    global thermalTimer
 #   global u
    if eqState:  # earthquake is on, turn off earthquake
        eqState = 0
        eqOff.show()
        eqOn.hide()
        thermalTimer = 0  # reset thermal timer
        print('Earthquake off')
        if controlState:  # if control is on, also turn this off
            controlState = 0
            ctrOff.show()
            ctrOn.hide()
 #           u = np.zeros((2, 1))  # kills control gain
            print('Control off')
    else:  # earthquake is not on
        eqState = 1
        eqOn.show()
        eqOff.hide()
        print('Earthquake on')


def change_control_text():
    global eqState
    global controlState
#    global u
    if controlState:  # if control is on
        controlState = 0
#        u = np.zeros((2, 1))  # kills control gain
        ctrOff.show()
        ctrOn.hide()
        print('Control Off')
    else:  # control is off
        if eqState:  # and earthquake is running
            controlState = 1
            ctrOn.show()
            ctrOff.hide()
            print('Control On')
        else:
            print('Cannot turn on control when earthquake is off')


def info():
    infoWindow.show()


def adminCountUp():
    global adminCount
    adminCount += 1
    print('AdminCount = {}'.format(adminCount))


def adminPrintIn():
    global adminCount
    # print('In {}'.format(adminCount))
    control_app.repeat(1000, adminCountUp)


def adminPrintOut():
    global adminCount
    # print('Out')
    if adminCount > 3:                # must hold the button for 3s
        countFeedback()
        adminWindow.show(wait=True)   # launches admin window
    adminCount = 0                    # resets the count
    control_app.cancel(adminCountUp)  # stops the count


def closeInfo():
    infoWindow.hide()


def closeAdmin():
    adminWindow.hide()


def launchFeedback():
    FeedbackWindow.show()


def goodFeedback():
    """Opens the feedback text file and appends '<Run number>, Bad Feedback'
    to indicate positive feedback"""
    feedback[-1][0] += 1
    with open('feedback.pkl','wb') as f:
        pickle.dump(feedback,f)
    FeedbackWindow.hide()


def badFeedback():
    """Opens the feedback file and appends '<Run number>, Bad Feedback'
    to indicate negative feedback"""
    feedback[-1][1] += 1
    with open('feedback.pkl','wb') as f:
        pickle.dump(feedback,f)
    infoWindow.show()
    FeedbackWindow.hide()


def updateBar():
    """Updates progress bar with the current value of micSum, the log sum
    of squared input samples from the accelerometer"""
    # deprecated - generate random bar graph values depending on state
    #if eqState:
    #    if controlState:
    #        mx = 5
    #        mn = 0
    #    else:
    #        mx = 20
    #        mn = 10
    #else:
    #    mx = 0
    #    mn = 0

    #v = random.randint(mn, mx)  # set random level
    global micSum
    if micSum>0:
        v = np.int(np.round((np.log(micSum)+7)*3)) # +n is a shift, *y adjusts dynamic range
        v = min(20,v)
        v = max(0,v)
        barChart.set_all('white')   # clear bar
        for i in range(v):
            barChart.set_pixel(0, 19-i, 'green')  # fill bar from bottom

def countFeedback():
    run = len(feedback)-1
    scoreStr = 'Feedback\nCurrent: Run {:3.0f}: {:3.0f}+/{:3.0f}-\n'.format(run,feedback[run][0],feedback[run][1])
    for i in range(run-1,run-5,-1):
        if i>=0:
            scoreStr += '         Run {:3.0f}: {:3.0f}+/{:3.0f}-\n'.format(i,feedback[i][0],feedback[i][1])
        else:
            scoreStr += '.......\n'
    scoreStr += '.......\n        All time: {:3.0f}+/{:3.0f}-'.format(sum(np.array(feedback))[0], sum(np.array(feedback))[1])
    feedbackText.value = scoreStr


# MAIN CALLBACK FUNCTION
def playingCallback(in_data, frame_count, time_info, status):
    # Convert Bytes to Numpy Array
    audio_frame_int = np.frombuffer(in_data, dtype=np.float32)
    global u
    global last_t
    global controlState
    global eqState
    global r
    global update
    global micSum
    global out_gain
    global con_gain
    # global error_log

    # Extract Only Input Channel 1
    audio_frame = np.reshape(audio_frame_int, (frame_size, 2))
    mic_in = audio_frame[:, 0]  # first channel of input
    #mic_in = mic_in - np.mean(mic_in)  # remove mean value?
    mic_in = lfilter(bh, ah, mic_in)  # APply HPF
    mic_in = lfilter(bl,al,mic_in) # Apply LPF

    # get accel level for bar
    micSum = sum(mic_in**2)
    # Generate Ref Signals
    t = np.arange(last_t, last_t + frame_size)/fs
    ref_sine = np.sin((2*np.pi*freq*t))
    ref_cos = np.cos((2*np.pi*freq*t))
    last_t = last_t+frame_size

    # Calculate Output Signal for Current Sample
    control_out = (ref_sine * u[0]) + (ref_cos * u[1])
    control_out = -control_out

    # Force LMS Filters to 0 when ANC is Turned Off
    if controlState == 0:
        # print('Resetting Filters to 0')
        u = np.zeros((2,1))
        update = np.zeros((2, 1))
        r = np.zeros((2, frame_size))
        con_gain = 0

    else:

        con_gain = 1
        if adaptive ==1:
        	# Filter Ref Signal by Plant Estimate

            r[0, :] = (ref_sine * np.real(h)) - (ref_cos * np.imag(h))
            r[1, :] = (ref_sine * np.imag(h)) + (ref_cos * np.real(h))

        	#Update Filter Coefficients for Next Sample Using Block LMS
            update[0] = (alpha/frame_size) * np.sum(r[0, :] * mic_in)
            update[1] = (alpha/frame_size) * np.sum(r[1, :] * mic_in)

            u[0] = ((1-alpha*gamma) * u[0]) - update[0]
            u[1] = ((1-alpha*gamma) * u[1]) - update[1]
            print(u)
            # print("mag = {}, ang = {}".format(abs(u[0] + (u[1]*1j)), np.arctan(u[1]/u[0])))
        else:
            # cheating - apply pre-measured filter
            u = np.array([[ 0.11814752],[ 0.06006437]]) #- GOOD filter djw
            ucomplex = u[0]+1j*u[1]
            angleChange = 0.08
            magFactor = 1
            ucomplex = magFactor*ucomplex*(np.cos(angleChange)+1j*np.sin(angleChange))
            u = np.array([np.real(ucomplex),np.imag(ucomplex)])
            # u = np.array([[ 0.0909988 ],[ 0.04948614]]) #- GOOD SET (25.2)
            #u = np.array([[ 0.00629916], [ 0.01227185]]) #- GOOD SET (104)

    out_mat = (con_gain*control_out, out_gain*ref_sine*eqState)  # Channel1 = Control,Channel2 = Primary
    out_mat = np.vstack(out_mat).reshape((-1,), order='F')

    # error_log = np.append(error_log,mic_in)
    # Ouput Processing
    out_data = out_mat.astype(np.float32)
    out_data = out_data.tobytes()
    return out_data, pyaudio.paContinue


def exit_function():
    stream.stop_stream()
    # savemat('error_log.mat', {'error_log':error_log})
    print('Function Terminated')


def updateTemp():
    proc = subprocess.Popen(['vcgencmd', 'measure_temp'], stdout=subprocess.PIPE)
    tempString = str(proc.communicate()[0])
    temp = re.findall('\D*(\d*.\d)',tempString)[0]
    tempText.value = 'CPU = ' + temp + 'deg C'


def thermalCutout():
    global controlState
    global eqState
    global thermalTimer
    global timerMax
    print('Thermal Cutout!')
    thermalTimer = 0  # reset thermal count-up
    change_earthquake_text() # as if earthquake button pressed


def thermalCountUp():
    global thermalTimer
    global eqState
    if eqState:
        thermalTimer += 1
        if thermalTimer > timerMax:
            thermalCutout()

def timerIncrement():
    global timerMax
    timerMax += 60
    timerText.value = '{} mins'.format(timerMax//60)

def restart(stream):
    exit_function()
    stream.start_stream()

def timerDecrement():
    global timerMax
    if timerMax >60:
        timerMax -= 60
        if timerMax == 60:
            timerText.value = '{} min '.format(timerMax//60)
        else:
            timerText.value = '{} mins'.format(timerMax//60)

def respawn():
    os.system('pkill python')


# GUI
control_app = App(title="Control App", bg='#000000', layout='grid')
control_app.tk.attributes('-fullscreen', True) # FULLSCREEN
control_app.font = 'Inter UI Bold'
control_app.text_size = 10
control_app.height = 480
control_app.width = 800
control_app.tk.config(cursor='none')
control_app.repeat(1000, thermalCountUp)

# Feedback Window
FeedbackWindow = Window(control_app)
FeedbackWindow.text_size = 18
FeedbackWindow.font = 'Nexa Bold'
FeedbackWindow.tk.attributes('-fullscreen',True)
FeedbackWindow.height = 480
FeedbackWindow.width = 800
FeedbackWindow.hide()
FeedbackWindow.bg = 'black'
FeedbackWindow.tk.config(cursor='none')

# buttons on feedback window
gap = Text(FeedbackWindow,'')
happy = Picture(FeedbackWindow, 'happy.gif')
happy.when_clicked = goodFeedback
gap = Text(FeedbackWindow,'')
#happyText = Text(FeedbackWindow, text = 'I learned something\n new today!', grid = [1,0], color = 'white', align = 'left')
confused = Picture(FeedbackWindow,'confused.gif')
confused.when_clicked = badFeedback
#confText = Text(FeedbackWindow, text = "I'm still not sure...", grid = [1,1], color = 'white', align = 'left')

#smiley1 = PushButton(FeedbackWindow, text=':)', command=smiley1_callback)
#smiley2 = PushButton(FeedbackWindow, text=':(', command=smiley2_callback)

# Additional information window
infoWindow = Window(control_app, layout='grid')
infoWindow.hide()
infoWindow.bg = 'black'
infoWindow.height = 480
infoWindow.width = 800
infoWindow.font = 'Nexa Bold'
infoWindow.text_color = 'white'
infoWindow.text_size = 18
infoWindow.tk.attributes('-fullscreen', True)
infoWindow.tk.config(cursor='none')
gap = Text(infoWindow, text=' ',grid = [0,0])
gap.width = 1
infotext = Text(infoWindow, align='left', text='An accelerometer is used\n'
                                 'to measure the vibration\n'
                                 'at the top of the tower.\n\n'
                                 'This signal is processed\n'
                                 'by a Raspberry Pi, then\n'
                                 'amplified and sent to\n'
                                 'the shaker.\n\n'
                                 'This cancels out vibrations\n'
                                 'from the earthquake.', grid=[1, 0])
# gifs play if PIL installed
img = Picture(infoWindow, 'shake.gif', grid=[2, 0,1,2])
closeButton = Picture(infoWindow, 'closeInfo.gif', grid = [1,1])
closeButton.when_clicked = closeInfo

# need to include an admin button. press and hold?
adminWindow = Window(control_app, title='Admin', layout='grid')
adminWindow.hide()
adminWindow.height = 480
adminWindow.width = 800
adminWindow.tk.attributes('-fullscreen', True)
adminWindow.text_size = 16
adminWindow.font = 'FreeMono'
tempText = Text(adminWindow, text = 'temp', grid = [2,2])
adminWindow.repeat(5000,updateTemp)
feedbackText = Text(adminWindow, text='Feedback Scores', grid = [0,0,1,3])
RespawnButton = PushButton(adminWindow, text = 'Respawn', grid = [3,5],
                              command = respawn)
closeAdminButton = PushButton(adminWindow, text = 'x', grid = [4,0],
                              command = closeAdmin)
closeAdminButton.bg = 'red'
gapAdmnin = Text(adminWindow,text='  ',grid = [1,0,1,2])
upButton = PushButton(adminWindow,text='+', command=timerIncrement, grid=[4,1])
downButton = PushButton(adminWindow, text='-', command=timerDecrement, grid=[2,1])
timerText = Text(adminWindow, text='{} min '.format(timerMax//60), grid=[3,1])
timerinfo = Text(adminWindow, text='Thermal cutout', grid = [2,0,3,1])
# restartButton = PushButton(adminWindow,text='Reset',grid = [4,4], command = restart, args = stream)

# Padding is implemented using empty text boxes (!)
TopGap = Text(control_app, text='', grid=[0, 0])
LGap = Text(control_app, text='', grid=[0, 1])
LGap.width = 5
midGap = Text(control_app, text='', grid=[2, 1])
midGap.width = 5
hGap = Text(control_app, text='', grid=[1, 2])
hGap.height = 3
barPad = Text(control_app, text='', grid=[4, 1])
barPad.width = 3

# bar chart on right hand edge
barChart = Waffle(control_app, height=20, width=1, pad=0, grid=[5, 1, 1, 4])
barChart.repeat(100, updateBar)  # update the bar every 100ms

# Push buttons
#earthquake = PushButton(control_app, text="Earthquake Off",
#                        command=change_earthquake_text, grid=[1, 1])
#earthquake.bg = 'red'
#earthquake.width = 12
#earthquake.height = 3

eqOn = Picture(control_app, 'eqon.gif',grid = [1, 1])
eqOn.hide()
eqOff = Picture(control_app, 'eqoff.gif', grid = [1, 1])
eqOff.show()

#control = PushButton(control_app, text="Control Off",
#                     command=change_control_text, grid=[3, 1])
#control.bg = 'red'
#control.width = 12
#control.height = 3
ctrOn = Picture(control_app, 'ctrlon.gif',grid = [3, 1])
ctrOn.hide()
ctrOff = Picture(control_app, 'ctrloff.gif', grid = [3, 1])
ctrOff.show()


# mini = PushButton(control_app,text='Minimise',command=minimise)
# mini.bg = 'orange'
#infoButton = PushButton(control_app, text='i', command=info, grid=[3, 3])
#infoButton.bg = [135, 215, 255]  # would be better to control UI colours above
infoButton = Picture(control_app, 'FindMore.gif', grid = [3, 3])
infoButton.when_clicked = info

feedbackButton = Picture(control_app, 'Feedback.gif', grid = [1, 3])
feedbackButton.when_clicked = launchFeedback

#feedbackButton = PushButton(control_app, text='Feedback',
#                            command=launchFeedback, grid=[1, 3])
#feedbackButton.bg = [135, 215, 255]

adminButton = Picture(control_app, 'corner.gif', grid=[0, 0], align='top')
adminButton.when_mouse_enters = adminPrintIn
adminButton.when_mouse_leaves = adminPrintOut


# <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> #
p = pyaudio.PyAudio()
#try:
stream = p.open(format=pyaudio.paFloat32,
                    channels=chan_count,
                    rate=fs,
                    output=True,
                    input=True,
                    stream_callback=playingCallback,
                    output_device_index=output_index,
                    input_device_index=input_index,
                    frames_per_buffer=frame_size)
stream.start_stream()

#except OSError:
 #   print('Running without sound')
# <<<<<<<<< RUN >>>>>>>>> #

atexit.register(exit_function)
print('Starting Function')

eqOn.when_clicked = change_earthquake_text
eqOff.when_clicked = change_earthquake_text
ctrOn.when_clicked = change_control_text
ctrOff.when_clicked = change_control_text

# feedback is handled using pickle
try:
    with open('feedback.pkl','rb') as f:
        feedback = pickle.load(f)
        feedback.append([0,0])
except FileNotFoundError:
    feedback = [[0,0]]

with open('feedback.pkl','wb') as f:
    pickle.dump(feedback,f)


control_app.display()  # show the app

# time.sleep(60);
