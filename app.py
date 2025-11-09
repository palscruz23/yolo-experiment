from ultralytics import YOLO
import cv2
import os

# Add folders if does not exist
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('output'):
    os.mkdir('output')

model = YOLO('yolov8m.pt')

results = model('data/image.jpg')

for result in results:
    result.show()
    result.save('output/output.jpg')