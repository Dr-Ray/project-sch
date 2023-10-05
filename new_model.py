import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
import joblib

from signal_processing import SignalProcessing

processor = SignalProcessing()
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
    
    def selectModel(self, model):
        if model.value == "cnn_ann_svm":
            self.set_model("CNN_ANN_SVM")
            return {"selected_model":"Ensemble with Convolutional neural network, Artificial neural network and Support vector machine"}
        
        if model.value == "cnn_ann":
            self.set_model("CNN_ANN")
            return {"selected_model":"Ensemble with Convolutional neural network and Artificial neural network"}
        
        if model.value == "cnn_svm":
            self.set_model("CNN_SVM")
            return {"selected_model":"Ensemble with Convolutional neural network and Support vector machine"}
        
        if model.value == "ann_svm":
            self.set_model("ANN_SVM")
            return {"selected_model":"Ensemble with Artificial neural network and Support vector machine"}
        
        if model.value == "ann":
            self.set_model("ANN")
            return {"selected_model":"Artificial neural network"}
        
        if model.value == "svm":
            self.set_model("SVM")
            return {"selected_model":"Support vector machine"}
        
        if model.value == "cnn":
            self.set_model("CNN")
            return {"selected_model":"Convolutional Neural Network"}
    
    def preprocess(self, data, l_freq, h_freq):
        # Noise removal and filter signal with cut off frquencies 20hz and 60hz
        filtered_signal = []

        # Rectify fullwave
        filtered_signal_rectified = []

        for label in data.columns[:-1]:
            filtered_signal.append(processor.bandpass_filter(data[label], l_freq, h_freq))
            filtered_signal_rectified.append(processor.rectify(processor.bandpass_filter(data[label], l_freq, h_freq)))