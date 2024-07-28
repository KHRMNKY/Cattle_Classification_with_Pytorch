from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, JSON, LargeBinary
from sqlalchemy.orm import relationship

from .database import Base

class Image(Base):
    __tablename__ = "images"

    image_Id = Column(String, primary_key=True, index=True)

    image_Name = Column(String)

    image_embedding = Column(LargeBinary)


