from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from typing import Annotated

import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
class simRealreq(BaseModel):
    channels : int
    noise: float
    duration : float
    gesture_labels : str
    sample_frequency : int

class SelectModel(str, Enum):
    svm_model = "svm"
    ann_model = "ann"
    cnn_model = "cnn"
    cnn_ann_svm_model = "cnn_ann_svm"
    cnn_ann_model = "cnn_ann"
    cnn_svm_model = "cnn_svm"
    ann_svm_model = "ann_svm"

class Mymodel:
    def __init__(self, model):
        self.model = model
    
    def read_file(self, filename):
        pass
    
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model
    
    def train_model(self, x_train, y_train):
        if(self.model == "svm"):
            return "training svm model for classification"
        if(self.model == "ann"):
            return "training ann model for classification"
        if(self.model == "cnn"):
            return "training cnn model for classification"
        else:
            return "No model selected for training"
    
    # A band pass filter 
    def bandpass_filter(self, input_signal, lp_freq, hp_freq, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        low = lp_freq / nyquist
        high = hp_freq / nyquist
        
        b, a = signal.butter(order, [low, high], 'bandpass', analog=False)
        y = signal.filtfilt(b, a, input_signal, axis=0)
        
        return y
    
    def rectify(self, signal):
        return abs(signal)
    
    def preprocess(self, data, l_freq, h_freq):
        # Noise removal and filter signal with cut off frquencies 20hz and 60hz
        filtered_signal = []

        # Rectify fullwave
        filtered_signal_rectified = []

        for label in data.columns[:-1]:
            filtered_signal.append(bandpass_filter(data[label], l_freq, h_freq))
            filtered_signal_rectified.append(self.rectify(self.bandpass_filter(data[label], l_freq, h_freq)))

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

ai_model = Mymodel('CNN')
simul = Simulation()

app = FastAPI()
app.mount("/stylesheet", StaticFiles(directory="stylesheet"), name="stylesheet")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/js", StaticFiles(directory="js"), name="js")

template = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request, "current_model":ai_model.get_model()})

@app.get('/train', response_class=HTMLResponse)
def train(request: Request):
    return template.TemplateResponse("train.html", {"request":request})

@app.post('/train/dataset')
def train(file: Annotated[UploadFile, File()]):
    return {"filename":file.filename}

@app.get('/train/view', response_class=HTMLResponse)
def train_view(request: Request):
    return template.TemplateResponse("train_view.html", {"request":request})

@app.get('/train/analysis', response_class=HTMLResponse)
def train_analysis(request: Request):
    return template.TemplateResponse("train_analysis.html", {"request":request})

@app.get('/predict', response_class=HTMLResponse)
def predict(request: Request):
    return template.TemplateResponse("predict.html", {"request": request})

@app.get('/predict/analysis', response_class=HTMLResponse)
def predict_analysis(request: Request):
    return template.TemplateResponse("predict_analysis.html", {"request": request})

@app.get('/realtime', response_class=HTMLResponse)
def realtime(request: Request):
    return template.TemplateResponse("realtime.html", {"request": request})

@app.get('/realtime/analysis', response_class=HTMLResponse)
def realtime_analysis(request: Request):
    return template.TemplateResponse("real_analysis.html", {"request":request, "data":simul.get_signal()})

@app.post('/realtime/simulate')
def realtime_analysis(data: simRealreq):
    simul.set_simulation_data(data.channels, data.duration, data.sample_frequency, data.noise, data.gesture_labels)
    return  { "redirect_link":"/realtime/analysis" }

@app.get('/realtime/simulate_data')
def simulate_data():
    signal = simul.generate_emg()
    return signal.tolist()

@app.get('/select_model', response_class=HTMLResponse)
def select_model(request: Request):
    return template.TemplateResponse("model_select.html", {"request":request})

@app.get('/selected_model/{model}')
def selected_model(model: SelectModel):
    if model.value == "cnn_ann_svm":
        ai_model.set_model("CNN_ANN_SVM")
        return {"selected_model":"Ensemble with Convolutional neural network, Artificial neural network and Support vector machine"}
    
    if model.value == "cnn_ann":
        ai_model.set_model("CNN_ANN")
        return {"selected_model":"Ensemble with Convolutional neural network and Artificial neural network"}
    
    if model.value == "cnn_svm":
        ai_model.set_model("CNN_SVM")
        return {"selected_model":"Ensemble with Convolutional neural network and Support vector machine"}
    
    if model.value == "ann_svm":
        ai_model.set_model("ANN_SVM")
        return {"selected_model":"Ensemble with Artificial neural network and Support vector machine"}
    
    if model.value == "ann":
        ai_model.set_model("ANN")
        return {"selected_model":"Artificial neural network"}
    
    if model.value == "svm":
        ai_model.set_model("SVM")
        return {"selected_model":"Support vector machine"}
    
    if model.value == "cnn":
        ai_model.set_model("CNN")
        return {"selected_model":"Convolutional Neural Network"}