from utils import device, transform, classes
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io
import torch
from torch import nn
import torchvision
from PIL import Image


modelPath = ".\\models\\model_(92.71%)_acc.pth"

class Pred(BaseModel):
    label: str
    confidence: float

app = FastAPI()

@app.post("/predict", response_model=Pred)
async def image_predd(file: UploadFile = File(...)):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    model_weights = torch.load(modelPath, map_location=device)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
       img = Image.open(io.BytesIO(await file.read())).convert("RGB")
       img = transform(img).unsqueeze(0).to(device)
       loggit = model(img)
       preds = nn.Softmax(dim=1)(loggit)
       pred = torch.argmax(preds, dim=1).item()
       label, confidence = classes[pred], preds[0][pred]
    return Pred(label=label, confidence=confidence)


