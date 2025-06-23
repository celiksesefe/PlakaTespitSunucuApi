# Use Python 3.10 slim image for Railway compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for OpenCV and EasyOCR)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy application code
COPY app/ ./app/
COPY run.py .

# Copy the trained model (ensure yolov8best.pt is in repository)
COPY yolov8best.pt .

# Create necessary directories
RUN mkdir -p uploads && \
    chmod -R 755 /app

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Expose port (Railway will set PORT env var)
EXPOSE $PORT

# Use Railway-compatible start command
CMD python run.py