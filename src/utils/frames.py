import cv2
import os

# Configuration
video_path = 'warped_video.mp4'
output_dir = 'frames'
frame_interval = 1  # Save every 30th frame

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    if frame_count % frame_interval == 0:
        output_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
        cv2.imwrite(output_path, frame)
        saved_count += 1
        print(saved_count)

    frame_count += 1

cap.release()
print(f"Done. Saved {saved_count} frames to '{output_dir}'.")
