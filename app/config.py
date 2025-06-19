# app/config.py
import os
from pathlib import Path

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8best.pt")
LANG_LIST = ['en']  # English only, matching training code

# API configuration - Increased to 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Performance configuration for high-resolution images
MAX_IMAGE_DIMENSION = 2048  # Max dimension for processing
THUMBNAIL_SIZE = (1920, 1920)  # Thumbnail size for very large images
JPEG_QUALITY = 85  # Quality for image compression if needed

# OCR Configuration (matching training implementation)
OCR_DETAIL = True
OCR_WIDTH_THS = 0.7
OCR_HEIGHT_THS = 0.7
OCR_MIN_CONFIDENCE = 0.6
DETECTION_MIN_CONFIDENCE = 0.5

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)