import torchvision
import io
import torch
from torch import nn
from utils import device, transform, classes
from PIL import Image
import pickle
import base64

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from SQLite import models, schemas, database, crud
from pydantic import BaseModel
import uuid

import clip
model, preprocess = clip.load("ViT-B/32", device)

modelPath = ".\\models\\model_(92.71%)_acc.pth"


class Pred(BaseModel):
    label: str
    confidence: float

#models.Base.metadata.drop_all(bind=database.engine) 
models.Base.metadata.create_all(bind=database.engine)


app = FastAPI()


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
def image_predd(image_Id: str, db: Session = Depends(get_db)):
    db_image = db.query(models.Image).filter(models.Image.image_Id == image_Id).first()

    if db_image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    model_weights = torch.load(modelPath, map_location=device)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()


    
    with torch.no_grad():
       
       image_embedding_tensor = torch.tensor(pickle.loads(db_image.image_embedding))
       print(image_embedding_tensor.shape)
       img = transform(image_embedding_tensor).unsqueeze(0).to(device)
       loggit = model(img)
       preds = nn.Softmax(dim=1)(loggit)
       pred = torch.argmax(preds, dim=1).item()
       label, confidence = classes[pred], preds[0][pred]*100
    return Pred(label=label, confidence=confidence)




@app.get("/images/{image_Id}", response_model=schemas.img_base)
def get_image(image_Id: str, db: Session = Depends(get_db)):
    db_image = crud.get_image(db, image_Id)
    db_image.image_embedding = base64.b64decode(db_image.image_embedding)
    return db_image



    

@app.put("/images/", response_model=schemas.img_base)
def update_image(image_Id: str, db: Session = Depends(get_db), update_Id: str = None, update_Name: str = None):
    update_image = db.query(models.Image).filter(models.Image.image_Id == image_Id).first()

    if update_image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    update_image.image_Name = update_Name
    update_image.image_Id = update_Id

    db.commit()
    db.refresh(update_image)
    return update_image


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


