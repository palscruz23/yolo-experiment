import gradio as gr
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_from_webcam(image):
    """Process webcam frames in real-time"""
    if image is None:
        return None
    
    results = model(image, verbose=False)
    annotated = results[0].plot()
    return annotated

# Create interface with webcam streaming
demo = gr.Interface(
    fn=detect_from_webcam,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=gr.Image(streaming=True),
    title="🎥 Real-time YOLO Detection",
    description="Allow webcam access for live object detection",
    live=True  # Process frames continuously
)

demo.launch()