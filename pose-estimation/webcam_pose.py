from ultralytics import YOLO
import cv2
import os
import time

# Add folders if does not exist
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('output'):
    os.mkdir('output')

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

keypoints = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}
fps_list = []
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow('Pose Estimation', annotated_frame)

    
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)
    if len(fps_list) > 20:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)
    cv2.putText(annotated_frame, f'FPS: "{avg_fps:.1f}',
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO phone camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()