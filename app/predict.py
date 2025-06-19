# app/predict.py
import numpy as np
from PIL import Image
import cv2
import logging
from typing import List, Dict, Any
import re
from .model import model_manager
from .utils import preprocess_image, preprocess_crop_for_ocr
from .config import DETECTION_MIN_CONFIDENCE
from .exceptions import APIException

logger = logging.getLogger(__name__)

def clean_plate_text(text: str) -> str:
    """Clean detected plate text (simple approach matching training)"""
    if not text:
        return ""
    
    # Remove extra spaces and convert to uppercase
    cleaned = ' '.join(text.split()).upper()
    
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
    
    # Remove excessive spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def predict_plate(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Predict license plates - EXACTLY matching training implementation
    """
    try:
        # Load image as PIL first for EXIF correction
        image_pil, image_np_rgb = preprocess_image(image_bytes)
        
        # Convert RGB to BGR for OpenCV operations (matching training)
        image = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        
        # Get models
        model = model_manager.get_model()
        ocr_reader = model_manager.get_ocr_reader()
        
        # Run YOLO detection (exactly as training: model(image))
        logger.info("Running YOLO detection")
        results = model(image)
        
        plates = []
        
        # Check detection results (exactly as training)
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                try:
                    # Extract detection info (exactly as training)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    
                    # Validate class (exactly as training)
                    if 0 <= class_id < len(model.names):
                        class_name = model.names[class_id]
                        if class_name != 'plate':
                            continue  # Only process 'plate' class
                        
                        # Check minimum confidence
                        if confidence < DETECTION_MIN_CONFIDENCE:
                            continue
                        
                        # Crop image (exactly as training)
                        cropped_image = image[y1:y2, x1:x2]
                        
                        if cropped_image.size == 0:
                            continue
                        
                        # Preprocess for OCR (exactly as training)
                        processed_image = preprocess_crop_for_ocr(cropped_image)
                        
                        # Run OCR with EXACT same parameters as training
                        logger.info(f"Running OCR on plate {i+1}")
                        result = ocr_reader.readtext(processed_image, detail=1)
                        
                        # Process OCR results (exactly as training)
                        if result:
                            all_texts = []
                            all_confidences = []
                            
                            for res in result:
                                text = res[1]  # Get text
                                ocr_conf = res[2] if len(res) > 2 else 1.0  # Get confidence
                                
                                if text.strip():  # Only non-empty text
                                    all_texts.append(text)
                                    all_confidences.append(ocr_conf)
                            
                            if all_texts:
                                # Combine all detected text
                                combined_text = ' '.join(all_texts)
                                cleaned_text = clean_plate_text(combined_text)
                                
                                # Calculate average OCR confidence
                                avg_ocr_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
                                
                                # Overall confidence
                                overall_confidence = (confidence + avg_ocr_conf) / 2
                                
                                plate_info = {
                                    "text": cleaned_text,
                                    "raw_text": combined_text,
                                    "confidence": round(overall_confidence, 3),
                                    "bbox": [x1, y1, x2, y2],
                                    "detection_confidence": round(confidence, 3),
                                    "ocr_confidence": round(avg_ocr_conf, 3),
                                    "plate_id": i + 1,
                                    "ocr_results_count": len(all_texts)
                                }
                                
                                plates.append(plate_info)
                                
                                logger.info(f"Detected plate {i+1}: '{cleaned_text}' (raw: '{combined_text}') (conf: {overall_confidence:.3f})")
                
                except Exception as e:
                    logger.warning(f"Error processing detection {i}: {e}")
                    continue
        else:
            logger.info("No plates detected")
        
        # Sort by confidence
        plates.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Total plates detected: {len(plates)}")
        return plates
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise APIException(f"Prediction failed: {str(e)}", 500)