# app/models.py

from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from .database import Base

class PlateRecord(Base):
    __tablename__ = "plate_records"
    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String, index=True)
    image_path = Column(String)
    detected_at = Column(DateTime, default=datetime.utcnow)
