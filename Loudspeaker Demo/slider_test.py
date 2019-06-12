import numpy as np
from guizero import App,PushButton,Slider,Text,CheckBox,Picture,Window,TextBox

master_volume = 0

def change_master_volume():
	global master_volume
	master_volume = slider.value/100.0
	print(master_volume)


playback_app = App(title="Playback App",bg = (255, 255, 255),layout='grid')
slider = Slider(playback_app, command=change_master_volume,start=0,end=100,grid=[2,5,3,1])
slider.value = 10
