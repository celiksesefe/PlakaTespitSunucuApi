import os
from pathlib import Path

# --- Model Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8best.pt")
LANG_LIST = ['en']  # OCR için dil ayarı

# --- OCR Karakter Seti Kısıtlaması (Allowlist) ---
OCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- API Ayarları ---
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# --- Performans ve Görsel Ayarları ---
MAX_IMAGE_DIMENSION = 2048
THUMBNAIL_SIZE = (1920, 1920)
JPEG_QUALITY = 85

# --- YOLO Algılama Ayarı ---
DETECTION_MIN_CONFIDENCE = 0.25

# --- Yollar ve Klasörler ---
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- AWS S3 Ayarları ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_S3_REGION = os.getenv("AWS_S3_REGION")

# (Opsiyonel) S3 URL template (eğer özel domain yoksa)
AWS_S3_BASE_URL = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com"
