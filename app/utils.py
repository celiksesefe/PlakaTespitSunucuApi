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
    """Yüklenen görselin format ve boyut kontrolü"""
    if len(file_content) > MAX_FILE_SIZE:
        raise FileSizeError(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()
    except Exception:
        raise InvalidImageError("Invalid or corrupted image file")

def correct_orientation(image: Image.Image) -> Image.Image:
    """EXIF dataya göre oryantasyon düzeltir"""
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
        pass
    return image

def optimize_for_processing(image: Image.Image) -> Image.Image:
    """Büyük görselleri orantılı şekilde boyutlandırır (işlem kolaylığı için)"""
    original_size = image.size
    max_dimension = max(original_size)
    if max_dimension > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max_dimension
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        logger.info(f"Resizing image from {original_size} to {new_size} for processing")
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def preprocess_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """Ana görseli tespit için hazırlar (PIL ve NumPy formatında döndürür)"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = correct_orientation(image)
        image = image.convert("RGB")
        image = optimize_for_processing(image)
        image_np = np.array(image)
        return image, image_np
    except Exception as e:
        raise InvalidImageError(f"Failed to process image: {str(e)}")