from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# linode password proj3ctk@r3n@i12322

app = FastAPI()
app.mount("/stylesheet", StaticFiles(directory="stylesheet"), name="stylesheet")
app.mount("/images", StaticFiles(directory="images"), name="images")

template = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request})

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
def realtime_analysis(request: Request):
    return template.TemplateResponse("model_select.html", {"request":request})