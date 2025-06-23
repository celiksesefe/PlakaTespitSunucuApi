# app/exceptions.py

class APIException(Exception):
    """Genel API exception sınıfı"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ModelLoadError(APIException):
    """Model yükleme hatası"""
    def __init__(self, message: str = "Failed to load model"):
        super().__init__(message, 500)

class InvalidImageError(APIException):
    """Görsel formatı/bozukluğu hatası"""
    def __init__(self, message: str = "Invalid image format"):
        super().__init__(message, 400)

class FileSizeError(APIException):
    """Görsel dosyası boyutu sınırı aşma hatası"""
    def __init__(self, message: str = "File size too large"):
        super().__init__(message, 413)

# (Opsiyonel) FastAPI ile global error handler'da bu sınıflar yakalanabilir.
