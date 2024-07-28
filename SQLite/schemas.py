from pydantic import BaseModel
from typing import Optional
from sqlalchemy import JSON


class img_base(BaseModel):
    image_Id: str
    image_Name: str
    image_embedding: bytes

    class Config:
        from_attributes = True
