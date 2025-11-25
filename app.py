import streamlit as st
from ultralytics import YOLO
import cv2
import os

st.title("Object Detection with YOLO")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose Model", ["yolo12n.pt", "yolov8n.pt"])
    application_mode = st.selectbox("Application", ["Object Detection", "Pose Recognition"])
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

model = YOLO(st.model_choice)

if st.application_mode == "Object Detection":
    st.subheader("Object Detection Mode")
    # camera_image = st.camera_input("Take a picture")
    camera_image = 0
    if camera_image is not None:

        results = model.predict(source=0,
                                conf = confidence,
                                iou = iou, 
                                verbose=False,
                                stream=True
                                )

        result = results[0] # Get first results
        img_box = result.plot() # Draw bounding box
        img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB) # Convert color from BGR to RGB
        st.image(img_box, caption = "Object Detection", use_container_width=True)

    

