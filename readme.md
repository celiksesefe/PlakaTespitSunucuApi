# Plate Recognition API

## Kurulum
1. Gerekli paketleri yükle:
    ```
    pip install -r requirements.txt
    ```
2. Model dosyasını (`yolov8best.pt`) ana dizine koy.

3. API'yi başlat:
    ```
    uvicorn app.main:app --reload
    ```

## Kullanım
- `/predict` endpoint'ine `POST` ile görsel (form-data, key: file) gönderin.
- Sonuç: JSON içinde plakalar döner.
