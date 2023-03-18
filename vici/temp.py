from __future__ import absolute_import, division
from flask import Flask, request
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import torchvision.transforms as transforms
import json
import base64
import warnings
import timm

warnings.filterwarnings("ignore")

app = Flask(__name__)
@app.route("/")
def hello_world():
    return "<p>Hello, ML Server!</p>"

def get_transform_2():
    return transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],\
                                std=[0.229, 0.224, 0.225]),
            ])

def make_json(pred):
    if int(pred) == 0:
        json_string = {
            "emotion" : "natural",
        }
    elif int(pred) == 1:
        json_string = {
            "emotion" : "angry",
        }
    elif int(pred) == 2:
        json_string = {
            "emotion" : "natural",
        }
    elif int(pred) == 3:
        json_string = {
            "emotion" : "embarrass",
        }
    elif int(pred) == 4:
        json_string = {
            "emotion" : "fear",
        }
    elif int(pred) == 5:
        json_string = {
            "emotion" : "happy",
        }
    elif int(pred) == 6:
        json_string = {
            "emotion" : "hurt",
        }
    else:
        json_string = {
            "emotion" : "sad",
        }
    
    json_object = json.dumps(json_string)
    return json_object

def infer_model():
    model = timm.create_model('efficientnet_b0', num_classes=7)
    model.load_state_dict(torch.load('model_best.pth', map_location="cpu")['state_dict'])
    return model

def pred_image(img):
    model = infer_model()           # 모델 불러오기
    model.eval()
    transform = get_transform_2()
    temp_img = transform(img)
    temp_img = temp_img.unsqueeze(0)
    a = model(temp_img)
    w = torch.argmax(a)         # w 예측 index
    json_object = make_json(w)

    return json_object

@app.route("/submit", methods = ['GET', 'POST'],)
def get_output():
    if request.method == "POST":
        params = request.get_json()
        img = Image.open(BytesIO(base64.b64decode(params["value"][0])))
        
        json_object = pred_image(img)

        return json_object


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = "8080", debug = True)