import numpy as np
import pandas as pd
import neurokit2 as nk

class Simulation:
    def __init__(self):
        self.channels = 8
        self.noise = 0.5
        self.duration = 10
        self.gesture_labels = "Fist"
        self.sample_frequency = 200
        self.emg_signal = []
    
    def set_simulation_data(self, chnls, duration, sm_frq, noise, label):
        self.channels = chnls
        self.noise = noise
        self.duration = duration
        self.sample_frequency = sm_frq
        self.gesture_labels = label
    
    def generate_emg(self):
        self.emg_signal = nk.emg_simulate(duration=self.duration, burst_number=3, burst_duration=1.0, sampling_rate=self.sample_frequency, noise=self.noise)
        return self.emg_signal
    
    def get_signal(self):
        return self.emg_signal

# linode password proj3ctk@r3n@i12322