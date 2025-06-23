# app/main.py

import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .database import engine, Base
from .predict import router as predict_router

# Ortam değişkenlerini yükle (.env)
load_dotenv()

# SQLAlchemy tablolarını oluştur
Base.metadata.create_all(bind=engine)

# FastAPI uygulaması başlatılıyor
app = FastAPI(
    title="Plaka Tespit API",
    description="YOLOv8, EasyOCR ve PostgreSQL ile plaka tespiti ve okuma",
    version="2.0.0"
)

# CORS middleware EKLENDİ!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme için * uygundur. Yayında sadece kendi frontend adresin olmalı!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction ve batch endpointlerini import edilen router ile ekle
app.include_router(predict_router)

# Sağlık kontrolü ve bilgilendirme endpointleri
@app.get("/")
async def root():
    return {
        "message": "Plaka Tespit API çalışıyor",
        "status": "healthy",
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "plaka-tespit-api",
        "version": "2.0.0"
    }
