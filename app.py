import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image

st.title("YOLO Experience")

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

model = YOLO(model_choice)

if application_mode == "Object Detection":
    col1, col2 = st.columns(2)
    with col1:

        st.subheader("Object Detection Mode")
        run = st.button("▶️ Start Detection", type="primary")
        stop = st.button("⏹️ Stop Detection")

        if run:
            st.success("🔴 Detection Active")

            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                # Auto-reload for continuous capture
                st.rerun()
    with col2:
        if run:
            if camera_image is not None:
                img = Image.open(camera_image)
                img_array = np.array(img)
                results = model(source=img_array,
                                        conf = confidence,
                                        iou = iou, 
                                        verbose=False,
                                        stream=True
                                        )

                result = results[0] # Get first results
                img_box = result.plot() # Draw bounding box
                img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB) # Convert color from BGR to RGB
                st.image(img_box, caption = "Object Detection", use_container_width=True)

            

