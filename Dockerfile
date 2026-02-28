FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies (CPU-only PyTorch)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code
COPY main.py .
COPY yolov8m.pt .

# Create directories for models and indices
RUN mkdir -p models indices

# Expose port
EXPOSE 8001

# Use PORT env var (Render sets this automatically)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
