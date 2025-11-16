# Multi-stage Dockerfile for Dog Re-Identification System
# Optimized for Raspberry Pi (ARM64) and x86_64

# Stage 1: Base image with Python
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libhdf5-dev \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Use piwheels for ARM if available, fallback to PyPI
RUN pip install --upgrade pip && \
    pip install numpy==1.26.4 && \
    pip install -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/logs && \
    chmod -R 755 /app/data /app/logs

# Expose ports
EXPOSE 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)" || exit 1

# Default command (can be overridden)
CMD ["python", "backend/inference_service.py"]
