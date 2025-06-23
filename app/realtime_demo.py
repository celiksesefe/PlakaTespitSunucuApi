# app/realtime_demo.py

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os

def advanced_preprocess_plate(crop):
    """Gelişmiş OCR ön işleme fonksiyonu."""
    if crop is None or crop.size == 0:
        return None
    # Griye çevir
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Histogram eşitleme
    gray = cv2.equalizeHist(gray)
    # Gürültü azaltma
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptif threshold
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binary

def clean_plate_text(text):
    """Plaka metninde sık yapılan OCR hatalarını düzeltir, formatlar."""
    import re
    if not text:
        return ""
    text = text.upper().replace(" ", "")
    # Karışan harf/rakamları düzelt
    text = (
        text.replace('O', '0')
            .replace('I', '1')
            .replace('B', '8')
    )
    # Sadece Türk plaka formatına uygunsa dön
    match = re.match(r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$', text)
    if match:
        return text
    return text  # Uymuyorsa en yakın sonucu döndür

def main():
    # Model ve OCR başlat
    model_path = "yolov8best.pt"
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        return
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(0)  # Bilgisayar kamerası
    print("Başlamak için kameraya bakın. Çıkmak için Q'ya basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO ile plaka tespiti
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            processed = advanced_preprocess_plate(crop)
            plate = ""
            if processed is not None:
                ocr_result = reader.readtext(
                    processed,
                    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    detail=0,
                    paragraph=False
                )
                plate = clean_plate_text(ocr_result[0]) if ocr_result else ""
            # Görüntüye kutu ve plaka yazısını ekle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Realtime Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
