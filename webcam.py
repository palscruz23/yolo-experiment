# from ultralytics import YOLO
# import cv2
# import os

# # Add folders if does not exist
# if not os.path.exists('data'):
#     os.mkdir('data')
# if not os.path.exists('output'):
#     os.mkdir('output')

# model = YOLO('yolov8n.pt')

# model.predict(source=0, 
#               show=True,
#               save=True,
#               project='output',
#               name='cam',
#               exist_ok=True,
#               )


from ultralytics import YOLO
import cv2
import os
import time
import queue
import threading

# Add folders if does not exist
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('output'):
    os.mkdir('output')
if not os.path.exists('output/webcam'):
    os.mkdir('output/webcam')

model = YOLO("yolo12n.pt")

# phone_url = 'https://192.168.20.19:8080/video'

# Frame queue for buffering
frame_queue = queue.Queue(maxsize=2)  # Keep only 2 frames

def capture_frames(url, frame_queue):
    """Separate thread to capture frames"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Clear old frames and add new one
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Remove old frame
            except queue.Empty:
                pass
        
        frame_queue.put(frame)
    
    cap.release()

# Start capture thread
capture_thread = threading.Thread(target=capture_frames, args=(0, frame_queue))
capture_thread.daemon = True
capture_thread.start()

print("Connecting to phone camera...")
time.sleep(2)  # Give thread time to start

print("Connected! Press 'q' to quit")

# Rotation setting
rotation_angle = 90  # Change this: 0, 90, 180, or 270

fps_list = []
rot=1

# Save stream
output_filename = f'output/webcam/recording_{int(time.time())}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_recording = 20

# Get first frame to determine size
print("Getting frame dimensions...")
first_frame = frame_queue.get(timeout=5)

# if rot == 1:
    # first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

results = model(first_frame, conf=0.5, verbose=False)
annotated_first = results[0].plot()
frame_height, frame_width = annotated_first.shape[:2]

print(f"Frame dimensions: {frame_width}x{frame_height}")

# frame_width, frame_height = 480, 720
out = cv2.VideoWriter(output_filename, fourcc, fps_recording, (frame_width, frame_height))

print(f'Recording to: {output_filename}')

cv2.namedWindow('YOLO phone camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO phone camera', frame_width, frame_height)
time.sleep(0.1)
while True:
    start_time = time.time()
    
    try:
        frame = frame_queue.get(timeout=1)
    except queue.Empty:
        print("No frames received")
        continue
    
    # if rot == 1:
    #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)    
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