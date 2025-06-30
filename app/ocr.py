import easyocr
from paddleocr import PaddleOCR
import re
import cv2
import numpy as np
import logging
from PIL import Image
import threading
from collections import Counter
import time

# YENİ IMPORT - Enhanced functionality için
from .ocr_enhancement import enhanced_clean_text, enhanced_validation, smart_ensemble_decision

logger = logging.getLogger(__name__)

# Karakter allowlist (Türk plakası için)
PLAKA_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class OCRManager:
    """OCR motorlarını yöneten singleton sınıf - Detaylı sonuç"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OCRManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            try:
                logger.info("Initializing detailed OCR engines...")
                # EasyOCR reader - detaylı sonuç için
                self.easyocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,
                    verbose=False
                )
                # PaddleOCR - hızlı ayarlar
                self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
                self._initialized = True
                logger.info("OCR engines initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OCR engines: {e}")
                raise

# Global OCR manager instance
ocr_manager = OCRManager()

def enhance_plate_image(img):
    """Plaka görüntüsünü OCR için optimize et"""
    # Kontrast artırma
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur ile gürültü azalt
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Keskinleştirme
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # RGB'ye geri dönüştür
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

def easyocr_plate(img):
    """EasyOCR ile gelişmiş okuma - Detaylı sonuç"""
    start_time = time.time()
    try:
        # Görüntüyü enhance et
        enhanced_img = enhance_plate_image(img)
        
        # EasyOCR okuma - optimize edilmiş parametreler
        results = ocr_manager.easyocr_reader.readtext(
            enhanced_img,
            detail=True,
            allowlist=PLAKA_ALLOWLIST,
            width_ths=0.5,
            height_ths=0.5,
            paragraph=False,
            batch_size=1,
            workers=0,
            text_threshold=0.5,
            low_text=0.2,
            link_threshold=0.2,
            canvas_size=1280,
            mag_ratio=1.0
        )
        
        # Sonuçları işle
        texts = []
        for result in results:
            if len(result) >= 3:
                bbox = result[0]        # Koordinatlar
                text = result[1]        # Metin
                confidence = result[2]  # Güven skoru
                
                if isinstance(text, str) and text.strip() and confidence > 0.3:
                    texts.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        # En yüksek güven skoruna göre sırala
        texts.sort(key=lambda x: x['confidence'], reverse=True)
        
        processing_time = time.time() - start_time
        
        # En iyi sonucu döndür
        if texts:
            best_result = texts[0]
            return {
                'text': best_result['text'],
                'confidence': best_result['confidence'],
                'processing_time': processing_time,
                'all_results': texts
            }
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'processing_time': processing_time,
                'all_results': []
            }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"EasyOCR Error: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'processing_time': processing_time,
            'all_results': []
        }

def paddleocr_plate(img):
    """PaddleOCR ile okuma - Detaylı sonuç"""
    start_time = time.time()
    try:
        # Görüntüyü enhance et
        enhanced_img = enhance_plate_image(img)
        
        # BGR formatına çevir (PaddleOCR için)
        if len(enhanced_img.shape) == 3:
            img_for_paddle = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        else:
            img_for_paddle = enhanced_img
        
        result = ocr_manager.paddle_ocr.ocr(img_for_paddle)
        
        texts = []
        
        if result is not None and result:
            # PaddleX format işleme
            for line_result in result:
                try:
                    # PaddleX format - OCRResult object
                    if hasattr(line_result, 'rec_texts') and hasattr(line_result, 'rec_scores'):
                        rec_texts = line_result.rec_texts
                        rec_scores = line_result.rec_scores
                        
                        if rec_texts and rec_scores:
                            for text, score in zip(rec_texts, rec_scores):
                                if isinstance(text, str) and text.strip() and score > 0.3:
                                    texts.append({
                                        'text': text.strip(),
                                        'confidence': score
                                    })
                    
                    # Dictionary format fallback
                    elif isinstance(line_result, dict):
                        if 'rec_texts' in line_result and 'rec_scores' in line_result:
                            rec_texts = line_result['rec_texts']
                            rec_scores = line_result['rec_scores']
                            
                            if rec_texts and rec_scores:
                                for text, score in zip(rec_texts, rec_scores):
                                    if isinstance(text, str) and text.strip() and score > 0.3:
                                        texts.append({
                                            'text': text.strip(),
                                            'confidence': score
                                        })
                    
                    # Legacy format fallback
                    elif isinstance(line_result, list):
                        for detection in line_result:
                            if isinstance(detection, list) and len(detection) >= 2:
                                text_info = detection[1]
                                if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                                    text = text_info[0]
                                    confidence = text_info[1]
                                    if isinstance(text, str) and text.strip() and confidence > 0.3:
                                        texts.append({
                                            'text': text.strip(),
                                            'confidence': confidence
                                        })
                            
                except Exception as e:
                    logger.debug(f"Error processing PaddleOCR result: {e}")
                    continue
        
        # En yüksek güven skoruna göre sırala
        texts.sort(key=lambda x: x['confidence'], reverse=True)
        
        processing_time = time.time() - start_time
        
        # En iyi sonucu döndür
        if texts:
            best_result = texts[0]
            return {
                'text': best_result['text'],
                'confidence': best_result['confidence'],
                'processing_time': processing_time,
                'all_results': texts
            }
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'processing_time': processing_time,
                'all_results': []
            }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"PaddleOCR Error: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'processing_time': processing_time,
            'all_results': []
        }

def clean_plate_text(text):
    """Gelişmiş metin temizleme - Enhanced logic kullanıyor"""
    return enhanced_clean_text(text)

def validate_turkish_plate(text):
    """Gelişmiş format kontrolü - Enhanced logic kullanıyor"""
    return enhanced_validation(text)

def get_all_ocr_results(img):
    """Tüm OCR sonuçlarını detaylı döndür - Smart position-aware ensemble logic ile"""
    total_start_time = time.time()
    
    try:
        # Her OCR motorundan detaylı sonuç al
        easyocr_result = easyocr_plate(img)
        paddleocr_result = paddleocr_plate(img)
        
        # Smart ensemble decision - CORE IMPROVEMENT
        ensemble_text, ensemble_confidence, decision_reason = smart_ensemble_decision(
            easyocr_result['text'], easyocr_result['confidence'],
            paddleocr_result['text'], paddleocr_result['confidence']
        )
        
        # Determine ensemble source based on decision
        if "both_agree" in decision_reason:
            ensemble_source = "both"
        elif "easyocr" in decision_reason:
            ensemble_source = "easyocr"
        elif "paddleocr" in decision_reason:
            ensemble_source = "paddleocr"
        else:
            ensemble_source = "fallback"
        
        # Clean individual results for backward compatibility
        easyocr_cleaned = clean_plate_text(easyocr_result['text'])
        paddleocr_cleaned = clean_plate_text(paddleocr_result['text'])
        
        total_processing_time = time.time() - total_start_time
        
        # Log the decision for debugging
        logger.debug(f"OCR Decision: {decision_reason}, Easy: '{easyocr_result['text']}' → '{easyocr_cleaned}', "
                    f"Paddle: '{paddleocr_result['text']}' → '{paddleocr_cleaned}', Final: '{ensemble_text}'")
        
        return {
            "easyocr": easyocr_cleaned,
            "easyocr_confidence": easyocr_result['confidence'],
            "easyocr_processing_time": easyocr_result['processing_time'],
            "paddleocr": paddleocr_cleaned,
            "paddleocr_confidence": paddleocr_result['confidence'],
            "paddleocr_processing_time": paddleocr_result['processing_time'],
            "ensemble": ensemble_text,
            "ensemble_confidence": ensemble_confidence,
            "ensemble_source": ensemble_source,
            "total_processing_time": total_processing_time
        }
        
    except Exception as e:
        total_processing_time = time.time() - total_start_time
        logger.error(f"Error in get_all_ocr_results: {e}")
        return {
            "easyocr": "",
            "easyocr_confidence": 0.0,
            "easyocr_processing_time": 0.0,
            "paddleocr": "",
            "paddleocr_confidence": 0.0,
            "paddleocr_processing_time": 0.0,
            "ensemble": "",
            "ensemble_confidence": 0.0,
            "ensemble_source": "none",
            "total_processing_time": total_processing_time
        }