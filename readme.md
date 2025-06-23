# Plaka Tespit API (YOLOv8 + EasyOCR + FastAPI + PostgreSQL)

Türkiye plakaları için hızlı, doğru ve bulut tabanlı otomatik plaka tespiti & karakter okuma sistemi  

---

## Özellikler
- YOLOv8 tabanlı plaka tespiti (görüntü işleme & veri artırma desteği)
- EasyOCR ile karakter okuma (Türk plakalarına uygun karakter seti kısıtlaması)
- Gelişmiş görsel ön işleme ve hata düzeltme (O/0, I/1, B/8 vs.)
- PostgreSQL veritabanına plakaların ve görsellerin otomatik kaydı
- Canlı demo: Webcam ile gerçek zamanlı plaka okuma (opsiyonel)
- API ile batch ve tekli görselden sorgulama
- Modern, modüler ve taşınabilir Python kod yapısı

---

## Kurulum

```bash
git clone https://github.com/kullanici/plaka-tespit-api.git
cd plaka-tespit-api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
