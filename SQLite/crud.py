from sqlalchemy.orm import Session
from api import HTTPException
from . import models, schemas


def get_image(db: Session, image_Id: str):
    image = db.query(models.Image).filter(models.Image.image_Id == image_Id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return image  

