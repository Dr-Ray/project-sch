import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
import joblib
import os

from signal_processing import SignalProcessing
from ml_models import SVM_Model, ANN_Model

processor = SignalProcessing()
class Mymodel:
    def __init__(self, model):
        self.model_type = model
        self.allowed_type = ['csv', 'json', 'xlsx', 'txt']
    
    def read_dataset(self, filename, folder='datasets'):
        file_ext = filename.split(".").pop()
        # Check if filetype is allowed
        if(file_ext in self.allowed_type):
            # Check if file exists in folder
            if(os.path.exists(f"./{folder}/{filename}")):
                if(file_ext == 'csv'):
                    df = pd.read_csv(f"./{folder}/{filename}")

                    return {"read": True, "dataframe":df}
                
                if(file_ext == 'json'):
                    df1 = pd.read_json(f"./{folder}/{filename}")
                    dfc = df1.to_csv(f"./{folder}/{filename.split('.')[0]}.csv")
                    df = df = pd.read_csv(f"./{folder}/{filename}")
                    
                    return {"read": True, "dataframe":df}
                
                if(file_ext == 'txt'):
                    df = pd.read_csv(f"./{folder}/{filename}", sep=" ", header=None)
                    return {"read": True, "dataframe":df}
                    
            return {"read": False, "message": "File does not exist / Unable to read file "}
        return {"read": False, "message": "Invalid / maliciious file"}
    
    def get_model(self):
        return self.model_type
    
    def set_model(self, model):
        self.model = model

    def save_model(self, model):
        joblib.dump(self.model, f"./saved_model/{model}")
        return model
    
    def load_saved_model(self, model_name):
        self.model = joblib.load('./saved_model/'+model_name)

        return model_name
    
    def train_model(self, dataset, percentage):
        if(self.model_type.lower() == "svm"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            self.model = SVM_Model()
            self.model.train(dataset, percentage)
            return "hello"
        
        if(self.model_type.lower() == "ann"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            
            self.model = ANN_Model((len(dataset)-1), 4)
            
            return "Hello"
        
        if(self.model_type.lower() == "cnn"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            return "Hello"
        
        if(self.model_type.lower() == "cnn_svm"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            return "Hello"
        
        if(self.model_type.lower() == "cnn_ann"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            return "Hello"
        
        if(self.model_type.lower() == "cnn_ann_svm"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            return "Hello"
        
        if(self.model_type.lower() == "ann_svm"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            return "Hello"
        else:
            return "No model selected for Training"
    
    def selectModel(self, model):
        if model.value == "cnn_ann_svm":
            self.set_model("CNN_ANN_SVM")
            return {
                "selected_model":"Ensemble with Convolutional neural network, Artificial neural network and Support vector machine"
            }
        
        if model.value == "cnn_ann":
            self.set_model("CNN_ANN")
            return {
                "selected_model":"Ensemble with Convolutional neural network and Artificial neural network"
            }
        
        if model.value == "cnn_svm":
            self.set_model("CNN_SVM")
            return {
                "selected_model":"Ensemble with Convolutional neural network and Support vector machine"
            }
        
        if model.value == "ann_svm":
            self.set_model("ANN_SVM")
            return {
                "selected_model":"Ensemble with Artificial neural network and Support vector machine"
            }
        
        if model.value == "ann":
            self.set_model("ANN")
            return {
                "selected_model":"Artificial neural network"
            }
        
        if model.value == "svm":
            self.set_model("SVM")
            return {
                "selected_model":"Support vector machine"
            }
        
        if model.value == "cnn":
            self.set_model("CNN")
            return {
                "selected_model":"Convolutional Neural Network"
            }
    
    # def preprocess(self, data, l_freq, h_freq):
    #     # Noise removal and filter signal with cut off frquencies 20hz and 60hz
    #     filtered_signal = []

    #     # Rectify fullwave
    #     filtered_signal_rectified = []

    #     for label in data.columns[:-1]:
    #         filtered_signal.append(processor.bandpass_filter(data[label], l_freq, h_freq))
    #         filtered_signal_rectified.append(processor.rectify(processor.bandpass_filter(data[label], l_freq, h_freq)))