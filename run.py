# run.py
import uvicorn
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Get port from environment variable (for Railway deployment)
    try:
        port = int(os.getenv("PORT", 8000))
    except (ValueError, TypeError):
        logger.warning("Invalid PORT environment variable, using default 8000")
        port = 8000
    
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting License Plate Detection API on {host}:{port}")
    logger.info(f"PORT environment variable: {os.getenv('PORT', 'not set')}")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=False,  # Disable reload for production
            workers=1,     # Single worker for better memory management
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)