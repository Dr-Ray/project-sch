from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from enum import Enum
from pydantic import BaseModel
from typing import Annotated

from simulation_model import Simulation
from new_model import Mymodel
from utility import Utility


class simRealreq(BaseModel):
    channels : int
    noise: float
    duration : float
    gesture_labels : str
    sample_frequency : int

class TrainInput(BaseModel):
    train_size : str
    filename: str
    g_set: int

class SelectModel(str, Enum):
    svm_model = "svm"
    ann_model = "ann"
    cnn_model = "cnn"
    cnn_ann_svm_model = "cnn_ann_svm"
    cnn_ann_model = "cnn_ann"
    cnn_svm_model = "cnn_svm"
    ann_svm_model = "ann_svm"


ai_model = Mymodel('CNN')
simul = Simulation()
my_utility = Utility()

app = FastAPI()
app.mount("/stylesheet", StaticFiles(directory="stylesheet"), name="stylesheet")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")
app.mount("/js", StaticFiles(directory="js"), name="js")

template = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request, "current_model":ai_model.get_model()})

@app.get('/train', response_class=HTMLResponse)
def train(request: Request):
    return template.TemplateResponse("train.html", {"request":request})

@app.post('/train/dataset')
async def training_dataset(file: UploadFile = File(...)):
    content = await file.read()
    saved_file = my_utility.save_file(content, file.filename, 'datasets')

    if(saved_file['saved']):
        return {
            "success":True, 
            "file":saved_file['filename']
        }
    else:
        return {
            "success":False, 
            "file":saved_file['filename'], 
            "message":"Unable to save file invalid / malicious file or requires root permission"
        }


@app.get('/train/view', response_class=HTMLResponse)
def train_view(request: Request):
    return template.TemplateResponse("train_view.html", {"request":request})

@app.get('/train/analysis', response_class=HTMLResponse)
def train_analysis(request: Request):
    return template.TemplateResponse("train_analysis.html", {
        "request":request, 
        "report": ai_model.report,
        "model": ai_model.get_model_fname()  
    })

@app.post('/train/percent')
def train_percent(inp: TrainInput):
    try:
        trainSize_ = int(inp.train_size)
        dataset = inp.filename
        df = ai_model.train_model(dataset, (trainSize_/100), inp.g_set)

        return df
    except Exception as e:
        return e

@app.post('/train/savemodel')
def savemodel():
    return "saving model...."
    
@app.post('/train/downloadmodel')
def downloadmodel():
    return "downloading...."


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
    return ai_model.selectModel(model)