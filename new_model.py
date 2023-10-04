import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
import joblib

class Mymodel:
    def __init__(self, model):
        self.model = model
    
    def read_file(self, filename):
        pass
    
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model

    def save_model(self, model):
        joblib.dump(self.model, f"./saved_model/{model}")
        return model

    
    def load_saved_model(self, model_name):
        self.model = joblib.load('./saved_model/'+model_name)

        return model_name
    
    def train_model(self, dataset, percentage):
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

