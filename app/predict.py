# app/predict.py
# This version keeps your original database model and just adds S3 upload

import numpy as np
import time
from fastapi import APIRouter, File, UploadFile, Depends, Request
from sqlalchemy.orm import Session
import os
import uuid
import logging

from .model import model_manager
from .preprocess import preprocess_plate_crop
from .ocr import get_all_ocr_results
from .utils import preprocess_image, validate_image
from .config import DETECTION_MIN_CONFIDENCE, UPLOAD_DIR
from .database import SessionLocal
from .models import PlateRecord
from .exceptions import APIException
from .s3_utils import s3_manager

logger = logging.getLogger(__name__)
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_image_url(request: Request, filename: str) -> str:
    """Generate a proper URL for the uploaded image that works in both local and cloud environments"""
    base_url = str(request.base_url).rstrip('/')
    image_url = f"{base_url}/static/uploads/{filename}"
    return image_url

def get_relative_path(filename: str) -> str:
    """Get relative path for database storage (cloud-friendly)"""
    return f"static/uploads/{filename}"

@router.post("/predict", summary="Görselden plaka tespiti ve gelişmiş OCR")
async def predict_plate_api(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    total_start_time = time.time()
    local_image_path = None
    
    try:
        # 1. Dosya validasyonu ve kaydı
        contents = await file.read()
        validate_image(contents, file.filename)
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        
        # 2. Save locally (as before)
        local_image_path = os.path.join(UPLOAD_DIR, filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        with open(local_image_path, "wb") as f:
            f.write(contents)

        # 3. URL ve path oluşturma
        image_url = get_image_url(request, filename)
        relative_path = get_relative_path(filename)
        
        # 4. AWS S3'e upload et (yeni özellik)
        s3_url = None
        try:
            s3_key, s3_url = s3_manager.upload_image(contents, filename, file.content_type)
            logger.info(f"Successfully uploaded to S3: {s3_key}")
            # S3 upload başarılı olursa, S3 URL'i kullan
            final_image_url = s3_url
        except Exception as s3_error:
            logger.warning(f"S3 upload failed, using local storage: {s3_error}")
            # S3 upload başarısız olursa, local URL kullan
            final_image_url = image_url

        # 5. Görseli preprocess et (orijinal kod)
        image_pil, image_np_rgb = preprocess_image(contents)
        image_bgr = np.ascontiguousarray(image_np_rgb[..., ::-1])

        # 6. Model ile plaka tespiti (orijinal kod)
        model = model_manager.get_model()
        results = model(image_bgr)

        plates = []

        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())

                    if 0 <= class_id < len(model.names):
                        class_name = model.names[class_id]
                        if class_name != 'plate':
                            continue
                        if confidence < DETECTION_MIN_CONFIDENCE:
                            continue

                        crop = image_bgr[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        processed = preprocess_plate_crop(crop)
                        ocr_results = get_all_ocr_results(processed)

                        paddleocr_text = ocr_results['paddleocr']
                        cleaned_text = ocr_results['ensemble']

                        # Veritabanına kaydet - S3 URL'i kullan (eğer varsa)
                        record = PlateRecord(
                            plate_text=paddleocr_text,
                            image_path=final_image_url  # S3 URL veya local URL
                        )
                        db.add(record)
                        db.commit()
                        db.refresh(record)

                        # Response (orijinal format)
                        plates.append({
                            "id": record.id,
                            "text": cleaned_text,
                            "bbox": [x1, y1, x2, y2],
                            "detection_confidence": round(confidence, 3),
                            "confidence": round(ocr_results['ensemble_confidence'], 3),
                            "ocr_confidence": round(ocr_results['ensemble_confidence'], 3),
                            "ocr_easyocr": ocr_results['easyocr'],
                            "ocr_easyocr_confidence": round(ocr_results['easyocr_confidence'], 3),
                            "ocr_easyocr_time": round(ocr_results['easyocr_processing_time'], 3),
                            "ocr_paddleocr": ocr_results['paddleocr'],
                            "ocr_paddleocr_confidence": round(ocr_results['paddleocr_confidence'], 3),
                            "ocr_paddleocr_time": round(ocr_results['paddleocr_processing_time'], 3),
                            "ensemble": cleaned_text,
                            "ensemble_source": ocr_results['ensemble_source'],
                            "image_url": final_image_url,  # S3 URL (preferred) or local URL
                            "image_path": relative_path,
                            "detected_at": str(record.detected_at)
                        })

                except Exception as e:
                    logger.warning(f"Error processing plate {i}: {e}")
                    continue

        plates.sort(key=lambda x: x['detection_confidence'], reverse=True)
        total_processing_time = time.time() - total_start_time
        logger.info(f"Total plates detected: {len(plates)}")
        
        return {
            "plates": plates,
            "processing_time": round(total_processing_time, 3)
        }

    except Exception as e:
        total_processing_time = time.time() - total_start_time
        logger.error(f"Prediction failed: {e}")
        
        # Clean up local file if error occurs
        if local_image_path and os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
            except:
                pass
                
        raise APIException(f"Prediction failed: {str(e)}", 500)