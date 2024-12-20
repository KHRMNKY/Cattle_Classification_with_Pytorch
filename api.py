import torchvision
import io
import torch
from torch import nn
from utils import device, transform, classes, image_pred
from PIL import Image
import pickle
import base64

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from SQLite import models, schemas, database, crud
from pydantic import BaseModel
import uuid


import clip
model, preprocess = clip.load("ViT-B/32", device)

#modelPath = ".\\models\\model_(92.71%)_acc.pth"
modelPath = "./models/model_(92.71%)_acc.pth"


class Pred(BaseModel):
    label: str
    confidence: list[float]
    



models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL'si
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP yöntemlerine izin
    allow_headers=["*"],  # Tüm header'lara izin
)


def create_Id():

    return str(uuid.uuid4())



def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image)
    return image_embedding  



def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.get("/")
async def read_root():
    return {"Don't forget to add docs in url http://127.0.0.1:8000"}





@app.post("/images/", response_model=schemas.img_base)
async def save_image_to_database(image_file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_Name = image_file.filename

    img = Image.open(io.BytesIO(await image_file.read())).convert("RGB")

    img_embedding = get_image_embedding(img)
    img_embedding_bytes = pickle.dumps(img_embedding.cpu().numpy())

    db_image = db.query(models.Image).filter(models.Image.image_embedding == img_embedding_bytes).first()

    if db_image:
        raise HTTPException(status_code=404, detail="Image already exists")

    db_image = models.Image(image_embedding=img_embedding_bytes, image_Name=image_Name, image_Id=create_Id())

    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    print("image added database successfully..")
    img_embedding_b64 = base64.b64encode(db_image.image_embedding).decode('utf-8')

    return models.Image(image_Id=db_image.image_Id, image_Name=db_image.image_Name, image_embedding=img_embedding_b64)




@app.post("/predict", response_model=Pred)
async def image_predd(image_file: UploadFile = File(...), db: Session = Depends(get_db)):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    model_weights = torch.load(modelPath, map_location=device)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
       img = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
       img = transform(img).unsqueeze(0).to(device)
       loggit = model(img)
       preds = nn.Softmax(dim=1)(loggit)
       pred = torch.argmax(preds, dim=1).item()
    return Pred(label=classes[pred], confidence=[preds[0][pred]*100])





@app.post("/predicts", response_model=Pred)
async def image_preddd(image_file: UploadFile = File(...), db: Session = Depends(get_db)):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    model_weights = torch.load(modelPath, map_location=device)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    

    with torch.no_grad():
       img = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
       img = transform(img).unsqueeze(0).to(device)
       loggit = model(img)
       preds = nn.Softmax(dim=1)(loggit)
       #print(f"preds : {preds}") # tensor([[0.1019, 0.1149, 0.0221, 0.0827, 0.1414, 0.3940, 0.1432]], device='cuda:0')
       preds_float_list = preds[0].cpu().detach().numpy().tolist()
       pred = torch.argmax(preds, dim=1).item()
    return Pred(label=classes[pred], confidence=preds_float_list)






@app.get("/images/{image_Id}", response_model=schemas.img_base)
def get_imagee(image_Id: str, db: Session = Depends(get_db)):
    db_image = crud.get_image(db, image_Id)
    name, id=db_image.image_Id, db_image.image_Name
    return schemas.img_base(image_Id=db_image.image_Id, image_Name=db_image.image_Name, image_embedding=base64.b64encode(db_image.image_embedding).decode('utf-8'))






@app.put("/images/", response_model=schemas.img_base)
def update_image(image_Id: str, db: Session = Depends(get_db), update_Id: str = None, update_Name: str = None):
    update_imagee = db.query(models.Image).filter(models.Image.image_Id == image_Id).first()

    if update_imagee is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    update_imagee.image_Name = update_Name
    update_imagee.image_Id = update_Id

    db.commit()
    db.refresh(update_imagee)
    return schemas.img_base(image_Id=update_imagee.image_Id, image_Name=update_imagee.image_Name, image_embedding=base64.b64encode(update_imagee.image_embedding).decode('utf-8'))



@app.delete("/images/", response_model=schemas.img_base)
async def delete_image(image_Id: str, db: Session = Depends(get_db)):

    image = db.query(models.Image).filter(models.Image.image_Id == image_Id).first()

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    db.delete(image)
    db.commit()
    return schemas.img_base(image_Id=image.image_Id, image_Name=image.image_Name, image_embedding=base64.b64encode(image.image_embedding).decode('utf-8'))



if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")