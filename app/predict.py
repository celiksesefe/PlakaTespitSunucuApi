# app/predict.py
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Any
import re
from .model import model_manager
from .utils import preprocess_image, preprocess_crop_for_ocr
from .config import (
    OCR_DETAIL, OCR_WIDTH_THS, OCR_HEIGHT_THS, 
    OCR_MIN_CONFIDENCE, DETECTION_MIN_CONFIDENCE
)
from .exceptions import APIException

logger = logging.getLogger(__name__)

def clean_plate_text(text: str) -> str:
    """Clean and format detected plate text (enhanced for Turkish plates)"""
    if not text:
        return ""
    
    # Remove extra spaces and convert to uppercase
    cleaned = ' '.join(text.split()).upper()
    
    # Remove common OCR errors and non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
    
    # Common OCR corrections
    cleaned = cleaned.replace('O', '0').replace('I', '1').replace('S', '5')
    
    # Remove excessive spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def predict_plate(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Predict license plates from image bytes (fully compatible with yolov8best.pt)
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        List of detected plates with text and confidence
    """
    try:
        # Preprocess image (includes orientation correction and optimization)
        image_pil, image_np = preprocess_image(image_bytes)
        
        # Get models
        model = model_manager.get_model()
        ocr_reader = model_manager.get_ocr_reader()
        
        # Run YOLO detection on numpy array
        logger.info("Running YOLO detection")
        results = model(image_np)
        
        plates = []
        
        # Check if any detections were made
        if results[0].boxes is None:
            logger.info("No objects detected in image")
            return plates
        
        for i, box in enumerate(results[0].boxes):
            try:
                class_id = int(box.cls.item())
                
                # Ensure class_id is valid
                if class_id >= len(model.names):
                    logger.warning(f"Invalid class_id {class_id}, skipping")
                    continue
                
                class_name = model.names[class_id]
                confidence = float(box.conf.item())
                
                # Only process 'plate' class with sufficient confidence
                if class_name != "plate" or confidence < DETECTION_MIN_CONFIDENCE:
                    logger.debug(f"Skipping detection: class={class_name}, conf={confidence}")
                    continue
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure valid coordinates within image bounds
                height, width = image_np.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bounding box coordinates: ({x1},{y1}) to ({x2},{y2})")
                    continue
                
                # Crop plate region
                crop = image_np[y1:y2, x1:x2]
                
                if crop.size == 0:
                    logger.warning(f"Empty crop for detection {i}")
                    continue
                
                # Preprocess crop for OCR (matching training code exactly)
                processed_crop = preprocess_crop_for_ocr(crop)
                
                # Run OCR on processed crop
                logger.info(f"Running OCR on plate {i+1}")
                ocr_results = ocr_reader.readtext(
                    processed_crop,
                    detail=OCR_DETAIL,
                    width_ths=OCR_WIDTH_THS,
                    height_ths=OCR_HEIGHT_THS,
                    # Allowlist for alphanumeric characters and space
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '
                )
                
                if ocr_results:
                    # Process OCR results
                    texts = []
                    confidences = []
                    
                    for result in ocr_results:
                        if len(result) >= 3:  # Ensure result has bbox, text, confidence
                            bbox, text, conf = result[0], result[1], result[2]
                            
                            # Filter by OCR confidence
                            if conf > OCR_MIN_CONFIDENCE:
                                texts.append(text)
                                confidences.append(conf)
                    
                    if texts:
                        # Combine all text results
                        combined_text = ' '.join(texts)
                        cleaned_text = clean_plate_text(combined_text)
                        
                        # Calculate average OCR confidence
                        avg_ocr_confidence = sum(confidences) / len(confidences)
                        
                        # Calculate overall confidence (weighted average)
                        overall_confidence = (confidence * 0.6 + avg_ocr_confidence * 0.4)
                        
                        plate_info = {
                            "text": cleaned_text,
                            "confidence": round(overall_confidence, 3),
                            "bbox": [x1, y1, x2, y2],
                            "detection_confidence": round(confidence, 3),
                            "ocr_confidence": round(avg_ocr_confidence, 3),
                            "plate_id": i + 1
                        }
                        
                        plates.append(plate_info)
                        
                        logger.info(f"Detected plate {i+1}: '{cleaned_text}' (confidence: {overall_confidence:.3f})")
                    else:
                        logger.info(f"No readable text found in plate {i+1} (low OCR confidence)")
                else:
                    logger.info(f"No OCR results for plate {i+1}")
            
            except Exception as e:
                logger.warning(f"Error processing detection {i}: {e}")
                continue
        
        # Sort plates by confidence (highest first)
        plates.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Total plates detected and processed: {len(plates)}")
        return plates
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise APIException(f"Prediction failed: {str(e)}", 500)