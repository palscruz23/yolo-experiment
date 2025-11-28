import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image
import time
from streamlit_tags import st_tags

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

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
    with st.spinner("Loading model..."):
        model = YOLO(model_choice)
    st.success("Model successfully loaded!")
    classes_input = st.multiselect("Select Classes", 
                                   options=list(model.names.values()), 
                                   default=None,
                                   )  # Default to all classes
    confidence = st.slider("Confidence Threshold",
                              min_value = 0.0,
                              max_value = 1.0,
                              value = 0.5,
                              step = 0.05)
    # iou = st.slider("Intersect Over Union (IoU) Threshold",
    #                           min_value = 0.0,
    #                           max_value = 1.0,
    #                           value = 0.5,
    #                           step = 0.05)

    classes = []
    if classes_input == []:
        all_class = list(range(len(model.names)))
        classes = all_class
    else:
        for input in classes_input:
            key = next(k for k, v in model.names.items() if v == input)
            classes.append(key)


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
                    # iou = iou, 
                    verbose=False,
                    classes=classes
                    )
    
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)
    if len(fps_list) > 20:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    img_box = results[0].plot(boxes=True, masks=True) # Draw bounding box
    # img_box = frame
    cv2.putText(img_box, f'FPS: "{avg_fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB) # Convert color from BGR to RGB

    # # Edge detection inside instances
    # if results[0].masks is not None:
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    #     for mask in results[0].masks:
    #         points = mask.xy[0].astype(np.int32)
            
    #         # Create mask
    #         mask_img = np.zeros(gray.shape, dtype=np.uint8)
    #         cv2.fillPoly(mask_img, [points], 255)
            
    #         # Masked region
    #         masked_region = cv2.bitwise_and(gray, gray, mask=mask_img)
            
    #         # Detect edges
    #         edges = cv2.Canny(masked_region, 50, 150)
    #         edges_in_polygon = cv2.bitwise_and(edges, edges, mask=mask_img)
            
    #         # Overlay edges in green on the result
    #         img_box[edges_in_polygon > 0] = [255, 255, 0]

    # Mask outline
    if results[0].masks is not None:
        for mask in results[0].masks:
            points = mask.xy[0].astype(np.int32)
            
            # Draw only the contour (outline)
            color = (255, 0, 0)  # Green outline
            cv2.polylines(img_box, [points], isClosed=True, color=color, thickness=10)
            cv2.fillPoly(img_box, [points], color=(0, 0, 0))
    # img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)


    proc_frame.image(img_box, caption="Processed Frame", width="stretch")



cap.release()

