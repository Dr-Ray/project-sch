from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

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
    
    def get_model(self):
        print(self.model)
        return self.model
    
    def set_model(self, model):
        self.model = model

# linode password proj3ctk@r3n@i12322

ai_model = Mymodel('CNN')

app = FastAPI()
app.mount("/stylesheet", StaticFiles(directory="stylesheet"), name="stylesheet")
app.mount("/images", StaticFiles(directory="images"), name="images")

template = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request, "current_model":ai_model.get_model()})

@app.get('/train', response_class=HTMLResponse)
def train(request: Request):
    return template.TemplateResponse("train.html", {"request":request})

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
    return template.TemplateResponse("real_analysis.html", {"request":request})

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
    
    else:
        return {"selected_model":"Convolutional Neural Network"}