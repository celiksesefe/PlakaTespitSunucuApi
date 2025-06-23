import numpy as np
import time
from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
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

logger = logging.getLogger(__name__)
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/predict", summary="Görselden plaka tespiti ve gelişmiş OCR")
async def predict_plate_api(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    total_start_time = time.time()  # Total processing time başlangıcı
    
    try:
        # 1. Dosya validasyonu ve kaydı
        contents = await file.read()
        validate_image(contents, file.filename)
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        image_path = os.path.join(UPLOAD_DIR, filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(contents)

        # 2. Görseli yükle ve preprocess et
        image_pil, image_np_rgb = preprocess_image(contents)
        image_bgr = np.ascontiguousarray(image_np_rgb[..., ::-1])

        # 3. YOLO ile plaka tespiti
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

                        # 4. Plaka crop
                        crop = image_bgr[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        # 5. Gelişmiş ön işleme (preprocess.py)
                        processed = preprocess_plate_crop(crop)

                        # 6. Tüm OCR sonuçlarını al (ocr.py)
                        ocr_results = get_all_ocr_results(processed)

                        # 7. Post-processing, skor hesabı vs.
                        cleaned_text = ocr_results['ensemble']

                        # 8. Veritabanına kaydet (sadece ensemble sonucunu)
                        record = PlateRecord(
                            plate_text=cleaned_text,
                            image_path=image_path
                        )
                        db.add(record)
                        db.commit()
                        db.refresh(record)

                        plates.append({
                            "id": record.id,
                            "text": cleaned_text,
                            "bbox": [x1, y1, x2, y2],
                            "detection_confidence": round(confidence, 3),
                            "confidence": round(ocr_results['ensemble_confidence'], 3),  # Overall confidence
                            "ocr_confidence": round(ocr_results['ensemble_confidence'], 3),  # OCR confidence
                            "ocr_easyocr": ocr_results['easyocr'],
                            "ocr_easyocr_confidence": round(ocr_results['easyocr_confidence'], 3),
                            "ocr_easyocr_time": round(ocr_results['easyocr_processing_time'], 3),
                            "ocr_paddleocr": ocr_results['paddleocr'],
                            "ocr_paddleocr_confidence": round(ocr_results['paddleocr_confidence'], 3),
                            "ocr_paddleocr_time": round(ocr_results['paddleocr_processing_time'], 3),
                            "ensemble": cleaned_text,
                            "ensemble_source": ocr_results['ensemble_source'],
                            "image_path": image_path,
                            "detected_at": str(record.detected_at)
                        })

                except Exception as e:
                    logger.warning(f"Error processing plate {i}: {e}")
                    continue

        plates.sort(key=lambda x: x['detection_confidence'], reverse=True)
        total_processing_time = time.time() - total_start_time  # Total süreyi hesapla
        logger.info(f"Total plates detected: {len(plates)}")
        
        return {
            "plates": plates,
            "processing_time": round(total_processing_time, 3)  # Ana response seviyesinde
        }

    except Exception as e:
        total_processing_time = time.time() - total_start_time
        logger.error(f"Prediction failed: {e}")
        raise APIException(f"Prediction failed: {str(e)}", 500)