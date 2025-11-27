import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image
import time

st.set_page_config(
        page_title="YOLO Experience",
        page_icon=":eyes:",
        layout="wide",
        )

st.title("YOLO Experience")

# Apply CSS
st.markdown("""
    <style>
        [data-testid="stImage"] {
            width: 50% !important;
            margin-left: auto;
            margin-right: auto;
        }
        
        [data-testid="stImage"] > img {
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    application_mode = st.selectbox("Application", ["Object Detection", "Pose Recognition", "Instance Segmentation"])
    if application_mode == "Object Detection":
        model_choice = st.selectbox("Choose Model", ["yolo12n.pt", "yolov8n.pt", "yolov8m-oiv7"])
    elif application_mode == "Pose Recognition":
        model_choice = st.selectbox("Choose Model", ["yolo11n-pose.pt", "yolov8n-pose.pt"])
    elif application_mode == "Instance Segmentation":
        model_choice = st.selectbox("Choose Model", ["yolo11n-seg.pt", "yolov8n-seg.pt"])  

    confidence = st.slider("Confidence Threshold",
                              min_value = 0.0,
                              max_value = 1.0,
                              value = 0.5,
                              step = 0.05)
    iou = st.slider("Intersect Over Union (IoU) Threshold",
                              min_value = 0.0,
                              max_value = 1.0,
                              value = 0.5,
                              step = 0.05)

model = YOLO(model_choice)

if application_mode == "Object Detection":
    st.subheader("Detection Output")
elif application_mode == "Pose Recognition":
    st.subheader("Pose Recognition")
elif application_mode == "Instance Segmentation":
    st.subheader("Instance Segmentation")

proc_frame = st.empty()

cap = cv2.VideoCapture(0)

fps_list = []

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        break

    # orig_frame.image(frame, channels="BGR", caption="Original")
    results = model(frame,
                    conf = confidence,
                    iou = iou, 
                    verbose=False
                    )
    
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)
    if len(fps_list) > 20:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    img_box = results[0].plot() # Draw bounding box
    cv2.putText(img_box, f'FPS: "{avg_fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB) # Convert color from BGR to RGB

    # Edge detection inside instances
    if results[0].masks is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for mask in results[0].masks:
            points = mask.xy[0].astype(np.int32)
            
            # Create mask
            mask_img = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask_img, [points], 255)
            
            # Masked region
            masked_region = cv2.bitwise_and(gray, gray, mask=mask_img)
            
            # Detect edges
            edges = cv2.Canny(masked_region, 50, 150)
            edges_in_polygon = cv2.bitwise_and(edges, edges, mask=mask_img)
            
            # Overlay edges in green on the result
            img_box[edges_in_polygon > 0] = [0, 255, 0]


    proc_frame.image(img_box, caption="Processed Frame", width="stretch")



cap.release()





