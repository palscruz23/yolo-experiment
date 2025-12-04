import cv2
from ultralytics import YOLO
import os

# 1. Load the YOLOv8 model
model = YOLO('yolo12n.pt')  # load a pretrained model (recommended for training)

# Add folders if does not exist
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('output'):
    os.mkdir('output')
if not os.path.exists('output/video'):
    os.mkdir('output/video')

# 2. Open the input video
input_path = "data/mining.mp4"
output_path = "output/video/output_video.mp4"
cap = cv2.VideoCapture(input_path)

# 3. Get video properties (width, height, fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 4. Define the codec and create VideoWriter object
# 'mp4v' is the standard codec for .mp4 files on most systems
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 4.5 Define danger zone (right side of the frame)
danger_zone_x1 = 0  # Start at 70% of frame width
danger_zone_y1 = int(frame_height * 0.5)
danger_zone_x2 = int(frame_width * 0.5)
danger_zone_y2 = frame_height

print("Processing video...")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 5. Crop the frame to the danger zone region
        danger_zone_crop = frame[danger_zone_y1:danger_zone_y2, danger_zone_x1:danger_zone_x2]

        # 6. Run YOLOv8 inference only on the danger zone
        results = model(danger_zone_crop, conf=0.5, verbose=True, classes=[0])

        # 7. Start with original frame
        annotated_frame = frame.copy()

        # 8. Check if any person is detected in the danger zone
        person_in_danger = False
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            person_in_danger = True
            for box in boxes:
                # Get bounding box coordinates (in cropped region coordinates)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Convert coordinates to full frame coordinates
                full_x1 = int(x1 + danger_zone_x1)
                full_y1 = int(y1 + danger_zone_y1)
                full_x2 = int(x2 + danger_zone_x1)
                full_y2 = int(y2 + danger_zone_y1)

                # Draw bounding box on full frame
                cv2.rectangle(annotated_frame, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 2)

                # Add label
                label = f"person {box.conf[0]:.2f}"
                cv2.putText(annotated_frame, label, (full_x1, full_y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw danger zone rectangle
        color = (0, 0, 255) if person_in_danger else (0, 255, 0)  # Red if danger, Green if safe
        cv2.rectangle(annotated_frame,
                     (danger_zone_x1, danger_zone_y1),
                     (danger_zone_x2, danger_zone_y2),
                     color, 3)

        # Display alert status
        status_text = "DANGER" if person_in_danger else "SAFE"
        text_color = (0, 0, 255) if person_in_danger else (0, 255, 0)
        cv2.putText(annotated_frame, status_text,
                   (danger_zone_x1 + 20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        # 7. Write the annotated frame to the output video
        out.write(annotated_frame)

        # Optional: Display the frame while processing (press 'q' to exit)
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# 8. Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved to {output_path}")