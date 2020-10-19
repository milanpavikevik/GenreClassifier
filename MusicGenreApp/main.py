from typing import Optional
from fastapi import FastAPI, Request, Depends, BackgroundTasks,  File, UploadFile
from fastapi.templating import Jinja2Templates
import os
from fastapi.responses import HTMLResponse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
# Audio
import librosa
import librosa.display

import models
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import Client
import numpy as np
import boto3
#import itertool
import pandas as pd

app = FastAPI()
models.Base.metadata.create_all(bind=engine)
tempplates = Jinja2Templates(directory="templetes")
model = None
client = boto3.client(
        's3',
        aws_access_key_id='AKIARU7DMDP44IK76JOC',
        aws_secret_access_key='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',
        # aws_default_region='eu-central-1'
    )

def upload_new_file(id,pred):
    global client
    filename = 'newSounds/' + pred + '_' + str(id) + '.mp3'
    client.upload_file('test.wav',
                       'heartsoundsfiles', filename)

def get_model():
    global client
    client.download_file('heartsoundsfiles', 'genre_classifier(test_98%).h5', 'genre_classifier(test_98%).h5')

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
def read_root(request: Request, db: Session = Depends(get_db)):
    patients = db.query(Client).all()
    get_model()
    global model
    if model is None:
        model = load_model("genre_classifier(test_98%).h5")

    return tempplates.TemplateResponse("dashboard.html",{
        "request":request,
        "patients":patients
    })


def extract_features(audio_path, offset):
    y, sr = librosa.load(audio_path, offset=offset, duration=4.5)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048,
                                       hop_length=512,
                                       n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    return mfccs


def predict_the_data(wavfile):

    content = wavfile.file
    # load model

    # classification
    classify_file = content
    x_test = []
    Fpred="nema"
    Fconf=0
    x_test.append(extract_features(classify_file, 60))
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    pred = model.predict(x_test, verbose=1)
    print(pred)
    pred_class = model.predict_classes(x_test)
    print(pred_class[0])
    if pred_class[0] == 0:
        Fpred = "Classical song"
        Fconf = round((pred[0][0]*100), 2)
    elif (pred_class[0] == 1):
        Fpred = "Hip-hop song"
        Fconf = round((pred[0][1] * 100), 2)
    elif (pred_class[0] == 2):
        Fpred = "Jazz song"
        Fconf = round((pred[0][2] * 100), 2)
    elif (pred_class[0] == 3):
        Fpred = "Metal song"
        Fconf = round((pred[0][3] * 100), 2)
    elif (pred_class[0] == 4):
        Fpred = "Folk song"
        Fconf = round((pred[0][4] * 100), 2)
    elif (pred_class[0] == 5):
        Fpred = "Pop song"
        Fconf = round((pred[0][5] * 100), 2)
    elif (pred_class[0] == 6):
        Fpred = "Rock song"
        Fconf = round((pred[0][6] * 100), 2)
    else:
        Fpred = "Techno song"
        Fconf = round((pred[0][7] * 100), 2)

    celConf = [round((pred[0][0] * 100), 2), round((pred[0][1] * 100), 2), round((pred[0][2]*100), 2), round((pred[0][3]*100), 2), round((pred[0][4]*100), 2), round((pred[0][5]*100), 2), round((pred[0][6]*100), 2), round((pred[0][7]*100), 2)]
    return Fpred, Fconf, celConf


@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...),  db: Session = Depends(get_db)):

    result1, conf, result2 = predict_the_data(file)
    with open("test.mp3", 'wb') as f:
        f.write(file.file.read())
    clasa = Client()
    clasa.prediction = result1
    clasa.confidenceLevel = conf
    clasa.confidenceLevel1 = result2[0]
    clasa.confidenceLevel2 = result2[1]
    clasa.confidenceLevel3 = result2[2]
    clasa.confidenceLevel4 = result2[3]
    clasa.confidenceLevel5 = result2[4]
    clasa.confidenceLevel6 = result2[5]
    clasa.confidenceLevel7 = result2[6]
    clasa.confidenceLevel8 = result2[7]


    db.add(clasa)
    db.commit()
    patients = db.query(Client).all()

    upload_new_file(patients[-1].id, patients[-1].prediction)

    return tempplates.TemplateResponse("dashboardPost.html", {
        "request": request,
        "patients": patients
    })
