from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from enum import Enum
from pydantic import BaseModel
from secrets import token_hex

from simulation_model import Simulation
from new_model import Mymodel


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
async def training_dataset(file: UploadFile = File(...)):
    file_ext = file.filename.split(".").pop()
    file_name = token_hex(10)
    # file_path = f"{file_name}.{file_ext}"

    file_path = file.filename

    if(os.path.exists('./datasets/'+file_path)):
        file_path = f"{file.filename.split('.')[0]}{token_hex(5)}.{file_ext}"

    with open('./datasets/'+file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return {"success":True, "file":file.filename}

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