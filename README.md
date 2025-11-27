# YOLO Experiment

A collection of Python scripts for experimenting with YOLO (You Only Look Once) models for various computer vision tasks including object detection and pose estimation.

## Overview

This repository contains implementations for running YOLO models on different input sources (images, webcam, phone camera) with support for object detection, pose estimation, and instance segmentation tasks. Includes both command-line scripts and an interactive Streamlit web application for easy experimentation.

## Repository Structure

```
yolo-experiment/
├── app.py               # Streamlit web application
├── object-detection/     # Object detection scripts
│   ├── image.py         # Process static images
│   ├── webcam.py        # Webcam stream detection
│   └── phone.py         # Phone camera stream detection
├── pose-estimation/     # Pose estimation scripts
│   └── webcam_pose.py   # Real-time pose estimation
├── archive/             # Legacy non-threaded implementations
├── data/                # Input data directory
├── output/              # Output directory for results
├── *.pt                 # Pre-trained YOLO model weights
└── requirements.txt     # Python dependencies
```

## Models Included

The repository includes several pre-trained YOLO model weights:

**Object Detection Models:**
- `yolov8n.pt` - YOLOv8 Nano
- `yolov8m.pt` - YOLOv8 Medium
- `yolov8m-oiv7` - YOLOv8 Medium trained on Open Images V7
- `yolov8l.pt` - YOLOv8 Large
- `yolo12n.pt` - YOLO12 Nano
- `yolo12s.pt` - YOLO12 Small
- `yolo12l.pt` - YOLO12 Large

**Pose Estimation Models:**
- `yolov8n-pose.pt` - YOLOv8 Nano Pose
- `yolo11n-pose.pt` - YOLO11 Nano Pose

**Instance Segmentation Models:**
- `yolov8n-seg.pt` - YOLOv8 Nano Segmentation
- `yolo11n-seg.pt` - YOLO11 Nano Segmentation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yolo-experiment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- ultralytics
- opencv-python
- pillow
- fiftyone
- streamlit (required for the web application)

## Usage

### Object Detection

#### 1. Image Detection
Process a single image with YOLO object detection:

```bash
python object-detection/image.py
```

- Place your input image at `data/image.jpg`
- Output will be saved to `data/output.jpg`
- Displays detection results in a window

#### 2. Webcam Detection
Real-time object detection using your computer's webcam:

```bash
python object-detection/webcam.py
```

Features:
- Real-time object detection with threading for optimal performance
- FPS counter overlay
- Automatic video recording to `output/webcam/recording_<timestamp>.mp4`
- Press 'q' to quit

#### 3. Phone Camera Detection
Stream and detect objects from your phone camera:

```bash
python object-detection/phone.py
```

Features:
- Connects to phone camera via IP webcam app (default: `https://192.168.20.19:8080/video`)
- Multi-threaded frame capture for reduced latency
- Frame rotation support (configurable via `rot` variable)
- Saves recording to `output/phone/recording_<timestamp>.mp4`
- Press 'q' to quit

**Setup:**
1. Install an IP webcam app on your phone (e.g., "IP Webcam" for Android)
2. Update the `phone_url` variable in `phone.py` with your phone's IP address
3. Ensure phone and computer are on the same network

### Pose Estimation

#### Webcam Pose Estimation
Real-time human pose estimation using your webcam:

```bash
python pose-estimation/webcam_pose.py
```

Features:
- Detects 17 body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- Real-time FPS display
- Press 'q' to quit

### Streamlit Web Application

Interactive web-based interface for running YOLO models with real-time webcam processing:

```bash
streamlit run app.py
```

**Features:**
- Web-based UI accessible via browser
- Three application modes:
  - **Object Detection** - Detect and classify objects in real-time
  - **Pose Recognition** - Track human body keypoints
  - **Instance Segmentation** - Segment and identify individual object instances
- Configurable settings via sidebar:
  - Model selection (mode-specific models available)
  - Confidence threshold slider (0.0-1.0)
  - IoU (Intersection over Union) threshold slider (0.0-1.0)
- Real-time webcam stream processing
- Live preview of processed frames

**Available Models:**
- Object Detection: `yolo12n.pt`, `yolov8n.pt`, `yolov8m-oiv7`
- Pose Recognition: `yolo11n-pose.pt`, `yolov8n-pose.pt`
- Instance Segmentation: `yolo11n-seg.pt`, `yolov8n-seg.pt`

**Requirements:**
In addition to the base requirements, you'll need:
```bash
pip install streamlit
```

The app automatically opens in your default browser at `http://localhost:8501`

## Configuration

### Model Selection
Each script uses a specific YOLO model. To change the model, edit the model initialization line:

```python
model = YOLO("yolov8n.pt")  # Change to desired model
```

### Detection Confidence
Adjust the confidence threshold in the detection scripts:

```python
results = model(frame, conf=0.5)  # Change conf value (0.0-1.0)
```

### Class Filtering
To detect specific object classes only (see `webcam.py` example):

```python
results = model(frame, conf=0.5, classes=[41, 45, 64, 66])  # Cup, bowl, keyboard, mouse
```

## Performance Optimization

The webcam and phone camera scripts use threading to separate frame capture from inference, reducing latency and improving FPS. Key optimizations:

- Frame queue with max size 2 to prevent buffering lag
- Separate capture thread for continuous frame grabbing
- Minimal buffer size in OpenCV capture

## Troubleshooting

**Camera not detected:**
- Ensure no other application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

**Phone camera connection issues:**
- Verify phone and computer are on the same network
- Check the IP address and port in the phone app
- Update `phone_url` in `phone.py` accordingly

**Low FPS:**
- Use a smaller/faster model (e.g., yolo12n.pt instead of yolo12l.pt)
- Reduce input resolution
- Lower confidence threshold may reduce processing time

## License

See LICENSE file for details.

## Future Work

- Additional pose estimation scripts for image and phone inputs
- Integration with more YOLO variants
- Support for custom trained models
- Batch processing capabilities
