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
                    df = pd.read_csv(f"./{folder}/{filename}", sep="\t")
                    return {"read": True, "dataframe":df}
                    
            return {"read": False, "message": "File does not exist / Unable to read file "}
        return {"read": False, "message": "Invalid / maliciious file"}
    
    def get_model(self):
        return self.model_type
    
    def get_model_fname(self):
        return self.model_fname
    
    def set_model(self, model):
        self.model_type = model

    def save_model(self, model):
        bin_file = joblib.dump(self.model, f"./saved_model/{model}")
        return bin_file
    
    def load_saved_model(self, model_name):
        self.model = joblib.load(f"./saved_model/{model_name}")
        return model_name
    
    def train_model(self, dataset, percentage, g_set=4, learning_rate=0.0001):
        if(self.model_type.lower() == "svm"):
            df = self.read_dataset(dataset)

            dataset = df['dataframe']
            self.model = SVM_Model()
            acc = self.model.train(dataset, percentage)

            self.report = self.model._getreport()
            
            return self.model._getreport()
        
        if(self.model_type.lower() == "ann"):
            df = self.read_dataset(dataset)
            dataset = df['dataframe']
            
            self.model = ANN_Model(len(dataset.columns)-1, g_set)

            resp = self.model.train(dataset, percentage, "adam", learning_rate, 100, 100)

            self.report = self.model._getreport()
            
            return self.model._getreport()
        
        # if(self.model_type.lower() == "cnn"):
        #     df = self.read_dataset(dataset)
        #     dataset = df['dataframe']
        #     return "Hello"
        
        # if(self.model_type.lower() == "cnn_svm"):
        #     df = self.read_dataset(dataset)
        #     dataset = df['dataframe']
        #     return "Hello"
        
        # if(self.model_type.lower() == "cnn_ann"):
        #     df = self.read_dataset(dataset)
        #     dataset = df['dataframe']
        #     return "Hello"
        
        # if(self.model_type.lower() == "cnn_ann_svm"):
        #     df = self.read_dataset(dataset)
        #     dataset = df['dataframe']
        #     return "Hello"
        
        # if(self.model_type.lower() == "ann_svm"):
        #     df = self.read_dataset(dataset)
        #     dataset = df['dataframe']
        #     return "Hello"
        else:
            return "No model selected for Training"
    
    def selectModel(self, model):
        if model.value == "cnn_ann_svm":
            self.set_model("CNN_ANN_SVM")
            self.model_fname = "Ensemble with Convolutional neural network, Artificial neural network and Support vector machine"
            return {
                "selected_model":"Ensemble with Convolutional neural network, Artificial neural network and Support vector machine"
            }
        
        if model.value == "cnn_ann":
            self.set_model("CNN_ANN")
            self.model_fname = "Ensemble with Convolutional neural network and Artificial neural network"
            return {
                "selected_model":"Ensemble with Convolutional neural network and Artificial neural network"
            }
        
        if model.value == "cnn_svm":
            self.set_model("CNN_SVM")
            self.model_fname = "Ensemble with Convolutional neural network and Support vector machine"
            return {
                "selected_model":"Ensemble with Convolutional neural network and Support vector machine"
            }
        
        if model.value == "ann_svm":
            self.set_model("ANN_SVM")
            self.model_fname = "Ensemble with Artificial neural network and Support vector machine"
            return {
                "selected_model":"Ensemble with Artificial neural network and Support vector machine"
            }
        
        if model.value == "ann":
            self.set_model("ANN")
            self.model_fname = "Artificial neural network"
            return {
                "selected_model":"Artificial neural network"
            }
        
        if model.value == "svm":
            self.set_model("SVM")
            self.model_fname = "Support vector machine"
            return {
                "selected_model":"Support vector machine"
            }
        
        if model.value == "cnn":
            self.set_model("CNN")
            self.model_fname = "Convolutional Neural Network"
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