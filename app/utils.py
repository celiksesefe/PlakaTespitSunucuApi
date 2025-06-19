# app/utils.py
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ExifTags, UnidentifiedImageError
import io
import logging
from .config import (
    MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MAX_IMAGE_DIMENSION, 
    THUMBNAIL_SIZE, JPEG_QUALITY
)
from .exceptions import InvalidImageError, FileSizeError

logger = logging.getLogger(__name__)

def validate_image(file_content: bytes, filename: str) -> None:
    """Validate uploaded image file"""
    
    # Check file size - now 50MB
    if len(file_content) > MAX_FILE_SIZE:
        raise FileSizeError(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
    
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Try to open image to validate format
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify image integrity
    except Exception:
        raise InvalidImageError("Invalid or corrupted image file")

def correct_orientation(image: Image.Image) -> Image.Image:
    """Correct image orientation based on EXIF data (matching training code)"""
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
    """Optimize high-resolution images for processing without losing quality"""
    original_size = image.size
    max_dimension = max(original_size)
    
    # If image is very large, create a processing version
    if max_dimension > MAX_IMAGE_DIMENSION:
        # Calculate new size maintaining aspect ratio
        ratio = MAX_IMAGE_DIMENSION / max_dimension
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        logger.info(f"Resizing image from {original_size} to {new_size} for processing")
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def preprocess_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """
    Preprocess image for detection (matching training code approach)
    Returns both PIL image and numpy array for different uses
    """
    try:
        # Load and convert image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Correct orientation based on EXIF
        image = correct_orientation(image)
        
        # Convert to RGB
        image = image.convert("RGB")
        
        # Optimize for processing if too large
        image = optimize_for_processing(image)
        
        # Convert to numpy array for OpenCV operations
        image_np = np.array(image)
        
        return image, image_np
        
    except Exception as e:
        raise InvalidImageError(f"Failed to process image: {str(e)}")

def preprocess_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Preprocess cropped plate region for OCR (EXACTLY matching training code)
    """
    try:
        # The crop should already be in BGR format from cv2.imread equivalent
        # Convert BGR to grayscale (exactly as training)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization (exactly as training)
        equalized = cv2.equalizeHist(gray)
        
        # Gaussian blur with (5,5) kernel (exactly as training)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # Binary threshold using Otsu's method (exactly as training)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    except Exception as e:
        logger.warning(f"OCR preprocessing failed, using original: {e}")
        # Fallback to grayscale if preprocessing fails
        if len(crop.shape) == 3:
            return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return crop