import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ExifTags
import io
import logging
from .config import (
    MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MAX_IMAGE_DIMENSION
)
from .exceptions import InvalidImageError, FileSizeError

logger = logging.getLogger(__name__)

def validate_image(file_content: bytes, filename: str) -> None:
    """
    Yüklenen görselin format ve boyut kontrolü yapar.
    Eğer dosya büyükse veya desteklenmeyen bir format ise uygun hata fırlatır.
    """
    if len(file_content) > MAX_FILE_SIZE:
        raise FileSizeError(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
    
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Görselin geçerli ve bozuk olmadığını kontrol eder
    except Exception:
        raise InvalidImageError("Invalid or corrupted image file")

def correct_orientation(image: Image.Image) -> Image.Image:
    """
    Görselin EXIF metadata'sına bakarak yönünü düzeltir.
    Bazı cihazlarda fotoğrafın oryantasyonu metadata ile saklanır.
    """
    try:
        exif = image._getexif()
        if exif:
            for tag, value in exif.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
    except (AttributeError, KeyError):
        # EXIF verisi yoksa veya işlenemezse, hata vermez
        pass
    return image

def optimize_for_processing(image: Image.Image) -> Image.Image:
    """
    Çok büyük görsellerin işlem yükünü azaltmak için 
    maksimum bir boyuta göre orantılı küçültme yapar.
    """
    original_size = image.size
    max_dimension = max(original_size)
    if max_dimension > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max_dimension
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        logger.info(f"Resizing image from {original_size} to {new_size} for processing")
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def preprocess_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """
    Ham bayt olarak gelen görseli,
    - PIL Image formatına dönüştürür,
    - Oryantasyonunu düzeltir,
    - RGB'ye çevirir,
    - Optimizasyon için boyutlandırır,
    - NumPy array formatına çevirir ve
    ikisini tuple olarak döner.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = correct_orientation(image)
        image = image.convert("RGB")
        image = optimize_for_processing(image)
        image_np = np.array(image)
        return image, image_np
    except Exception as e:
        raise InvalidImageError(f"Failed to process image: {str(e)}")
