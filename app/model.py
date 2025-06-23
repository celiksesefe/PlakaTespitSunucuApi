from ultralytics import YOLO
import logging
from .config import MODEL_PATH
from .exceptions import ModelLoadError

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")
    
    def get_model(self):
        if self.model is None:
            raise ModelLoadError("YOLO model not loaded")
        return self.model

# Global model manager instance
model_manager = ModelManager()
