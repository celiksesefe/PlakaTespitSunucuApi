# app/model.py
from ultralytics import YOLO
import easyocr
import logging
from .config import MODEL_PATH, LANG_LIST
from .exceptions import ModelLoadError

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.ocr_reader = None
        self._load_models()
    
    def _load_models(self):
        """Load YOLO and OCR models with error handling"""
        try:
            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")
        
        try:
            logger.info(f"Loading OCR reader with languages: {LANG_LIST}")
            self.ocr_reader = easyocr.Reader(LANG_LIST)
            logger.info("OCR reader loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OCR reader: {e}")
            raise ModelLoadError(f"Failed to load OCR reader: {str(e)}")
    
    def get_model(self):
        if self.model is None:
            raise ModelLoadError("YOLO model not loaded")
        return self.model
    
    def get_ocr_reader(self):
        if self.ocr_reader is None:
            raise ModelLoadError("OCR reader not loaded")
        return self.ocr_reader

# Global model manager instance
model_manager = ModelManager()