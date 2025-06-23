# app/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# .env dosyasından veritabanı bağlantı adresini alır
load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
# Örnek: postgresql://kullanici:sifre@host:5432/plakatespit

# SQLAlchemy engine ve session oluşturuluyor
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
