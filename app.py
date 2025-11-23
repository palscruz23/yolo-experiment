"""
YOLOv8 Object Detection with Real-time Webcam Support

Features:
- Image upload with adjustable confidence/IOU thresholds
- Real-time webcam streaming with threading optimization
- Queue-based frame buffering (inspired by phone.py)
- FPS tracking with rolling average
- GPU acceleration (CUDA) if available, CPU fallback
- Compatible with Hugging Face Spaces

Threading Architecture:
- Uses daemon threads for non-blocking frame processing
- Queue with maxsize=2 prevents frame buildup
- Intelligent frame skipping when processing is slow
- Thread-safe with locks for concurrent access

Performance:
- Automatically uses GPU (CUDA) for faster inference
- FPS display shows device being used (CUDA/CPU)
- Optimized for real-time performance
"""

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import threading
import queue
import time
from collections import deque
import torch
import spaces

# Suppress ultralytics verbose output
os.environ['YOLO_VERBOSE'] = 'False'

# Check for GPU availability
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 0:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
model.to(device)  # Move model to GPU if available
print(f"Model loaded successfully on {device}!")

# Threading components for faster processing (inspired by phone.py)
frame_queue = queue.Queue(maxsize=2)  # Keep only 2 frames to prevent lag
processing_lock = threading.Lock()
latest_result = None
is_processing = False

# FPS tracking
fps_list = deque(maxlen=5)  # Rolling average of last 20 frames
last_process_time = time.time()

@spaces.GPU
def detect_objects(image, conf_threshold, iou_threshold):
    """
    Detect objects in an image using YOLOv8

    Args:
        image: Input image (numpy array)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS

    Returns:
        Annotated image with bounding boxes
    """
    if image is None:
        return None

    # Run inference on GPU if available
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False
    )

    # Plot results
    annotated_image = results[0].plot()

    # # Convert BGR to RGB for display
    # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image

@spaces.GPU
def process_frame_thread(image, conf=0.25):
    """Thread worker to process frames with FPS tracking"""
    global latest_result, is_processing, fps_list, last_process_time

    try:
        start_time = time.time()

        # Run YOLO inference on GPU if available
        results = model.predict(source=image, conf=conf, device=device, verbose=False)
        annotated_image = results[0].plot()
        # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        fps_list.append(fps)
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else fps

        # Add FPS overlay to image with device info
        cv2.putText(annotated_image, f'FPS: {avg_fps:.1f}',
                    (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with processing_lock:
            latest_result = annotated_image
            last_process_time = time.time()
    finally:
        with processing_lock:
            is_processing = False

def detect_objects_webcam(image):
    """Threaded version for webcam streaming with frame skipping and FPS display"""
    global latest_result, is_processing

    if image is None:
        return None

    # Clear old frames from queue if full (frame skipping like phone.py)
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

    # Check if currently processing
    with processing_lock:
        if is_processing:
            # Skip this frame and return last result
            return latest_result if latest_result is not None else image
        else:
            # Start processing this frame
            is_processing = True

    # Process in background thread
    thread = threading.Thread(target=process_frame_thread, args=(image.copy(),), daemon=True)
    thread.start()

    # Return latest result immediately (non-blocking)
    with processing_lock:
        return latest_result if latest_result is not None else image

# Create Gradio interface with Blocks for better layout
with gr.Blocks(title="YOLOv8 Object Detection") as demo:
    
    gr.Markdown(
        """
        # 🎯 YOLOv8 Object Detection
        
        Upload an image or use your webcam to detect objects in real-time using YOLOv8 nano model.
        
        **Detects 80 object classes** including: person, car, bicycle, dog, cat, and more!
        """
    )
    
       
    # Tab 1: Webcam
    with gr.Tab("🎥 Webcam (Real-time)"):
        gr.Markdown(
            """
            ### Live Detection from Webcam
            Click on the image below and select **webcam** to start real-time detection.

            **⚡ Features:**
            - Optimized threading with queue-based frame buffering
            - Intelligent frame skipping for smooth performance
            - Real-time FPS counter displayed on video
            - Rolling average FPS calculation (last 20 frames)

            **Note**: Works best in Chrome/Edge browsers. Compatible with Hugging Face Spaces.
            """
        )
        
        with gr.Row():
            webcam_input = gr.Image(
                label="Webcam Input",
                sources=["webcam"],
                streaming=True,
                type="numpy"
            )
            webcam_output = gr.Image(
                label="Detection Output",
                streaming=True,
                type="numpy"
            )
        
        webcam_input.stream(
            fn=detect_objects_webcam,
            inputs=webcam_input,
            outputs=webcam_output,
            stream_every=0.033  # ~30 FPS target, actual FPS depends on inference speed
        )

    gr.Markdown(
        """
        ---
        ### 📝 About
        
        This app uses **YOLOv8n** (nano) for fast object detection. The model can detect:
        
        - 👤 People and body parts
        - 🚗 Vehicles (car, truck, bus, motorcycle, etc.)
        - 🐕 Animals (dog, cat, horse, bird, etc.)
        - ⚽ Sports equipment
        - 🍎 Food items
        - 📱 Electronics
        - And 70+ more classes!
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Adjust confidence threshold if you get too many/few detections
        - Webcam works best with stable internet connection
        
        Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Gradio](https://gradio.app/)
        """
    )

# Launch the app
# Disable hot-reloading to prevent Spaces version conflicts
demo.launch(show_error=True)