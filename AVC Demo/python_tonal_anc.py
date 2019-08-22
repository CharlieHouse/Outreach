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
from scipy.signal import lfilter, butter, csd, welch, coherence
from scipy.io import savemat
from guizero import App, PushButton, Text, Picture, Window, Waffle
import random
import re
import subprocess
import pickle
import os
import time

# <<<<<<<<< VARIABLES >>>>>>>>> #
# Audio Stream Variables
fs = 44100
output_index = 2 # python -m sounddevice
input_index = 2 # python -m sounddevice
frame_size = 4096
chan_count = 2
con_gain = 0
out_gain = 0.08 #0.08 for 104Hz, 0.2 for 25.5#

# measurement parameters
mes_gain = 0.02  # 
duration = 20
input_recording = np.zeros(((duration+1)*fs))
noise_recording = np.zeros(((duration+1)*fs))
measureState = 0  # turn on if performing measurements

# LMS Variables
gamma = 0.1
alpha = 1e-2
adaptive = 0

# Global Variables
freq = 40.3
controlState = 0
eqState = 0
adminCount = 0  # timer for administrative page
thermalTimer = 0  # timer counts up
timerMax = 1*60 #minute thermal timer
#run = -1
micSum = 0

fullScreen = True

last_t = 0  # Counter used for Sine Wave

# Estimate of Plant Response - uncomment to load last measured
h = np.load('/home/pi/Documents/AVC/plantdata.npy')
# measured, averaged h, at gain = 0.1. Multiplied by 10 for TF with unit gain
# = -(1.786269421 + 8.214203719j)*10
#h = -(0.011642594960987 + 0.010710587901811j)*1000 #- GOOD SET (25.2)
#h = (0.032754130060299 + 0.053213797611848j)*1000 #- GOOD SET (104)

# Calculate HPF at 20Hz
normal_cutoff = 20 / (0.5 * fs)
bh, ah = butter(4, normal_cutoff, btype='high', analog=False)

# Calculate LPF at 70Hz
normal_cutoff = 70/(0.5*fs)
bl,al = butter(6,normal_cutoff,btype='low',analog=False)


# Initial LMS Coefficients
u = np.zeros((2, 1))
u_opt = np.load('/home/pi/Documents/AVC/uopt.npy')
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


def launchSetupWizard():
    s1Window.show()
    adminWindow.hide()
    acceptButton.disable()


def closeS1():
    adminWindow.show()
    s1Window.hide()


def closeS2():
    adminWindow.show()
    s2Window.hide()

def step2():
    s2Window.show()
    s1Window.hide()


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
        v = np.int(np.round((np.log(micSum)+3)*4)) # +n is a shift, *y adjusts dynamic range
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
    global u_opt
    global last_t
    global controlState
    global eqState
    global r
    global update
    global micSum
    global out_gain
    global con_gain
    global measureState
    global mes_gain
    global noise_recording
    global input_recording
    global adaptive
    global h
    global micSum
    # global error_log
    # print("I'm alive, load = " + str(stream.get_cpu_load()))

    # Extract Only Input Channel 1
    audio_frame = np.reshape(audio_frame_int, (frame_size, 2))
    mic_in = audio_frame[:, 0]  # first channel of input
    #mic_in = mic_in - np.mean(mic_in)  # remove mean value?
    #mic_in = lfilter(bh, ah, mic_in)  # APply HPF
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
    if measureState:
        out_mat = (ref_sine*mes_gain, ref_sine*0)  # control, primary
        out_mat = np.vstack(out_mat).reshape((-1,), order='F')

        noise_recording[last_t : last_t+frame_size] = ref_sine*mes_gain
        input_recording[last_t : last_t+frame_size] = mic_in

    else:

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
                u = u_opt
                #ucomplex = u[0]+1j*u[1]
                #angleChange = 0.08
                #magFactor = 1
                #ucomplex = magFactor*ucomplex*(np.cos(angleChange)+1j*np.sin(angleChange))
                #u = np.array([np.real(ucomplex),np.imag(ucomplex)])
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
    

def restartPyaudio():
    global stream
    global p
    stream.stop_stream
    stream .close()
    p.terminate()
    time.sleep(1)
    p = pyaudio.PyAudio()
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
    

def turnOnMeasurement():
    global measureState
    global last_t
    global input_recording
    global noise_recording 
    # script enables measurement of plant matrix
    if measureState == 1:
        pass
    else:
        print('Measurement turned on')
        plantButton.text = 'Measuring...'
        input_recording = np.zeros(((duration+1)*fs))
        noise_recording = np.zeros(((duration+1)*fs))
        last_t = 0
        measureState = 1
        control_app.after(1000*duration, turnOffMeasurement)


def turnOffMeasurement():
    global measureState
    global input_recording
    global noise_recording
    global h
    global stream
    global p
    measureState = 0
    stream.stop_stream
    stream .close()
    p.terminate()
    plantButton.text = 'Measure Plant'
    print('Complete')
    nfft = 2**16
    [f,sxy] = csd(noise_recording,input_recording,fs=fs,window='hanning',nperseg=nfft,noverlap=nfft/2,nfft=nfft)
    [f,sxx] = welch(noise_recording,fs=fs,window='hanning',nperseg=nfft,noverlap=nfft/2,nfft=nfft)
    [f,coh] = coherence(noise_recording, input_recording, fs=fs, window='hanning', nperseg=nfft, noverlap=nfft/2, nfft=nfft)
    H = np.divide(sxx,sxy)

    freq_ind = (np.abs(f - freq)).argmin()
    plant_est = H[freq_ind]
    print('Coherence:')
    print(coh[freq_ind])
    print('Plant Estimate')
    print(plant_est)
    h = -1*plant_est*10    # gain and inversion to account for low level input
    print('Saving Data')
    savemat('plant.mat', {'mic':input_recording,'noise':noise_recording,'f':f,'H':H,'sxx':sxx,'sxy':sxy,'coh':coh})
    np.save('plantdata.npy',-1*plant_est*10)
    print('Data saved')
    plantTxt.value= 'Plant =' + str(np.real(h))[:7] + ',' + str(np.imag(h))[:7]
    coherenceTxt.value = 'Coherence = {:.5f}'.format(coh[freq_ind])
    acceptButton.enable()
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
    print('Is stream active?')
    print(stream.is_active())

def testLMS():
    global adaptive
    global eqState
    global controlState
    global micSum
    global h
    global u
    global u_opt

    if eqState == 1:
        eqState = 0
        controlState = 0
        adaptive = 0
        uText.value = 'U = [{}, {}]'.format(u[0],u[1])
        testButton.text = 'Test'
        u_opt = u
        np.save('uopt.npy',u_opt)
    else:
        adaptive = 1
        eqState = 1
        controlState = 1
        testButton.text = 'Stop test'


def updateError():
    global micSum
    errorText.value = 'Error = {:.4f}'.format(micSum)


# GUI
control_app = App(title="Control App", layout='grid')
control_app.tk.attributes('-fullscreen', fullScreen) # FULLSCREEN
control_app.bg = 'black'
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
FeedbackWindow.tk.attributes('-fullscreen', fullScreen)
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
infoWindow.tk.attributes('-fullscreen', fullScreen)
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
adminWindow.hide() # start in admin window
adminWindow.height = 480
adminWindow.width = 800
adminWindow.tk.attributes('-fullscreen', fullScreen)
adminWindow.text_size = 16
adminWindow.font = 'FreeMono'

tempText = Text(adminWindow, text = 'temp', grid = [2,2])
adminWindow.repeat(5000,updateTemp)
feedbackText = Text(adminWindow, text='Feedback Scores', grid = [0,0,1,3])
RespawnButton = PushButton(adminWindow, text = 'Respawn', grid = [2,5],
                              command = respawn)
RestartStream = PushButton(adminWindow, text = 'Restart Stream', grid = [2,6],
                           command = restartPyaudio)
closeAdminButton = PushButton(adminWindow, text = 'x', grid = [4,0],
                              command = closeAdmin)
closeAdminButton.bg = 'red'
gapAdmnin = Text(adminWindow,text='  ',grid = [1,0,1,2])
upButton = PushButton(adminWindow,text='+', command=timerIncrement, grid=[4,1])
downButton = PushButton(adminWindow, text='-', command=timerDecrement, grid=[2,1])
timerText = Text(adminWindow, text='{} min '.format(timerMax//60), grid=[3,1])
timerinfo = Text(adminWindow, text='Thermal cutout', grid = [2,0,3,1])
setupButton = PushButton(adminWindow, text = 'Setup', command = launchSetupWizard, grid = [0,5])
setuptext = Text(adminWindow, text = 'Run Setup before first launch ', grid = [0,6], align = 'left')
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

# setup Wizard available from admin root
s1Window = Window(control_app, title='Setup', layout='grid')
s1Window.hide()
s1Window.height = 480
s1Window.width = 800
s1Window.tk.attributes('-fullscreen', fullScreen)
s1Window.text_size = 16
s1Window.font = 'FreeMono'

infotext = Text(s1Window, text = 'Press "Measure Plant" to take a 10 second measurement \n'
                'do not touch the tower during the measurement', grid = [1,1,2,1], align = 'left')
closeS1 = PushButton(s1Window, text='x', command = closeS1, grid = [3,1], align = 'right')
closeS1.bg = 'red'
igap = Text(s1Window, text = '   ', grid = [1,2])
infoText2 = Text(s1Window, text = 'If coherence is high (> 0.98), accept the measurement \n'
                 'otherwise, re-run the measurement', grid = [1,3,2,1], align = 'left')
igap2 = Text(s1Window, text = '--------------------------------------------', grid = [1,4, 2,1], align = 'center')
plantButton = PushButton(s1Window,text='Measure Plant', command=turnOnMeasurement, grid=[1,5], align='left')
#topgap = Text(s1Window, text = '     ', grid = [2,1], align = 'left')
plantTxt = Text(s1Window, text='Plant = [xxxxxx + xxxxx j]', grid = [1,6],  align = 'left')
coherenceTxt = Text(s1Window, 'Coherence = xxxx', grid = [1,7], align = 'left')

acceptButton = PushButton(s1Window, text = 'Accept measurement', command = step2, grid = [1,8], align = 'left')
acceptButton.disable()

s2Window = Window(control_app, title='Setup 2', layout = 'grid')
s2Window.hide()
s2Window.height = 480
s2Window.width = 800
s2Window.tk.attributes('-fullscreen', fullScreen)
s2Window.text_size = 16
s2Window.font = 'FreeMono'

infoText = Text (s2Window, text = 'Press "Test" to run active vibration control\n'
                 'The error should decrease.\n' 'When it reaches a steady low value, stop the test',
                 grid = [1,1,2,1], align = 'left')
infoText2 = Text(s2Window, text = 'If the error increases, stop the test immediately', color = 'red',
                 grid = [1,2,2,1], align = 'left')
infoText3 = Text(s2Window, text = '---------------------------------------------',grid = [1,3,2,1])
testButton = PushButton(s2Window, text = 'Test', command = testLMS, grid = [1,4])
closeS2 = PushButton(s2Window, text = 'x', command = closeS2, grid = [3,1], align = 'right')
closeS2.bg = 'red'
errorText = Text(s2Window, text = 'Error = xxxxx', grid = [1,5])
errorText.repeat(200,updateError)
uText = Text(s2Window, text = 'u = xxxxxxx', grid = [2,5])


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

countFeedback()
adminWindow.show()

control_app.display()  # show the app
stream.stop_stream()
p.terminate()
print('now at bottom of program')
# time.sleep(60);
