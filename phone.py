from ultralytics import YOLO
import cv2
import os

# Add folders if does not exist
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('output'):
    os.mkdir('output')

model = YOLO('yolov8n.pt')

phone_url = 'https://192.168.20.19:8080/video'

cap = cv2.VideoCapture(phone_url)

if not cap.isOpened():
    print("Failed to connect to phone!")
    exit()

print("Connected to phone app!")
print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO phone camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destrotAllWindows()