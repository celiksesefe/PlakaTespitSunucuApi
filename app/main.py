# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .predict import predict_plate
from .utils import validate_image
from .exceptions import APIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="License Plate Detection API",
    description="API for detecting and reading license plates from images using YOLOv8",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Exception handler for custom API exceptions
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    logger.error(f"API Exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message, 
            "status_code": exc.status_code,
            "success": False
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "success": False
        }
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "License Plate Detection API is running",
        "status": "healthy",
        "version": "2.0.0",
        "model": "yolov8best.pt"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "license-plate-detection-api",
        "version": "2.0.0",
        "max_file_size_mb": 50,
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    }

@app.get("/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "License Plate Detection API",
        "version": "2.0.0",
        "model": "yolov8best.pt",
        "capabilities": [
            "License plate detection",
            "OCR text recognition",
            "High-resolution image support",
            "EXIF orientation correction"
        ],
        "limits": {
            "max_file_size": "50MB",
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
            "max_processing_dimension": 2048
        }
    }

def run_prediction(image_bytes: bytes) -> list:
    """Run prediction in thread pool to prevent blocking"""
    return predict_plate(image_bytes)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict license plates from uploaded image
    
    Args:
        file: Image file (jpg, png, etc.) - up to 50MB
        
    Returns:
        JSON with detected plates and their information
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise APIException("No file provided", 400)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Read file content
        image_bytes = await file.read()
        
        # Validate image
        validate_image(image_bytes, file.filename)
        
        # Run prediction in thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        plates = await loop.run_in_executor(executor, run_prediction, image_bytes)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Processed {file.filename} in {processing_time:.2f}s, found {len(plates)} plates")
        
        response = {
            "success": True,
            "plates": plates,
            "count": len(plates),
            "processing_time": round(processing_time, 3),
            "filename": file.filename,
            "file_size_mb": round(len(image_bytes) / (1024 * 1024), 2)
        }
        
        return response
        
    except APIException:
        raise  # Re-raise API exceptions
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename if file.filename else 'unknown'}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict license plates from multiple uploaded images
    
    Args:
        files: List of image files - up to 5 files
        
    Returns:
        JSON with results for each file
    """
    if len(files) > 5:
        raise APIException("Maximum 5 files allowed per batch", 400)
    
    start_time = time.time()
    results = []
    
    for file in files:
        try:
            if not file.filename:
                results.append({
                    "filename": "unknown",
                    "success": False,
                    "error": "No filename provided"
                })
                continue
            
            image_bytes = await file.read()
            validate_image(image_bytes, file.filename)
            
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            plates = await loop.run_in_executor(executor, run_prediction, image_bytes)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "plates": plates,
                "count": len(plates),
                "file_size_mb": round(len(image_bytes) / (1024 * 1024), 2)
            })
            
        except APIException as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": e.message
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    total_time = time.time() - start_time
    
    return {
        "success": True,
        "results": results,
        "total_files": len(files),
        "successful_files": sum(1 for r in results if r["success"]),
        "total_processing_time": round(total_time, 3)
    }