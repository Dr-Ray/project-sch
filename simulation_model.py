from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from typing import Annotated
from secrets import token_hex

import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
import os
import joblib

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