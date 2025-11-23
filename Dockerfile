# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# # Download YOLO model during build to avoid downloading at runtime
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
