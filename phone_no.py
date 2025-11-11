from ultralytics import YOLO
import cv2
import os
import time 


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

fps_list = []
rot=1

# Save stream
output_filename = f'output/phone/recording_{int(time.time())}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_recording = 20

# Get first frame to determine size
print("Getting frame dimensions...")
ret, first_frame = cap.read()

if rot == 1:
    first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

results = model(first_frame, conf=0.5, verbose=False)
annotated_first = results[0].plot()
frame_height, frame_width = annotated_first.shape[:2]

print(f"Frame dimensions: {frame_width}x{frame_height}")

# frame_width, frame_height = 480, 720
out = cv2.VideoWriter(output_filename, fourcc, fps_recording, (frame_width, frame_height))

print(f'Recording to: {output_filename}')
cv2.namedWindow('YOLO phone camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO phone camera', frame_width, frame_height)
fps_list = []
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    if rot == 1:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  
    results = model(frame, conf=0.5, verbose=True)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

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

time.sleep(0.1)
out.release()
cv2.destroyAllWindows()

print(f'\nVideo saved to: {output_filename}')