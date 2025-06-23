# app/preprocess.py

import cv2
import numpy as np
from skimage import exposure

def add_padding(crop, pad_percent=0.15):
    """Plaka crop'una kenarlardan dolgu ekle (yüzde olarak)."""
    h, w = crop.shape[:2]
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)
    return cv2.copyMakeBorder(crop, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def resize_optimal(crop, target_width=250):
    """OCR için crop'u ideal boyuta getir (en iyi sonuç için 200-300px arası genişlik)."""
    h, w = crop.shape[:2]
    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def apply_clahe(gray_img):
    """CLAHE ile lokal kontrastı artır."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def denoise_and_sharpen(img):
    """Gürültü azaltma + keskinleştirme (unsharp mask)."""
    # Gaussian Blur ile gürültüyü azalt
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Unsharp mask (img + sharpened - blur)
    sharpen = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharpen

def adaptive_threshold(img):
    """Uyarlanabilir eşikleme ile siyah-beyaz'a dönüştür."""
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15
    )

def perspective_correction(crop, corners):
    """
    Eğer plaka köşeleri (4 nokta) belli ise, perspektif dönüşümü uygula.
    corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (saat yönüyle)
    """
    w = max(
        int(np.linalg.norm(np.array(corners[0]) - np.array(corners[1]))),
        int(np.linalg.norm(np.array(corners[2]) - np.array(corners[3])))
    )
    h = max(
        int(np.linalg.norm(np.array(corners[0]) - np.array(corners[3]))),
        int(np.linalg.norm(np.array(corners[1]) - np.array(corners[2])))
    )
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)
    return cv2.warpPerspective(crop, M, (w, h))

def preprocess_plate_crop(crop, corners=None, add_padding_percent=0.15, target_width=250):
    """
    Tam pipeline:
    - (isteğe bağlı) perspektif düzeltme
    - padding
    - optimal boyuta resize
    - griye çevir, CLAHE uygula
    - gürültü azalt + sharpen
    - adaptif threshold
    """
    if corners is not None:
        crop = perspective_correction(crop, corners)
    crop = add_padding(crop, add_padding_percent)
    crop = resize_optimal(crop, target_width)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray)
    sharpened = denoise_and_sharpen(clahe)
    thresh = adaptive_threshold(sharpened)
    return thresh  # En son OCR'ye gönderilecek görüntü (veya hem thresh hem sharpened döndürülebilir)
