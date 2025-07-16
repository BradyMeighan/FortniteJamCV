import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# --- Configuration ---
IMAGE_FOLDER = 'frames'
# Corners of the trapezoid in the source image.
# Derived from user's ROIs, taking the center of each small square.
# Order: Top-left, Top-right, Bottom-right, Bottom-left
SRC_POINTS = np.float32([
    [769, 469],  # Top-left (from ROI x=764, y=463)
    [1156, 465], # Top-right (from ROI x=1150, y=459)
    [1301, 836], # Bottom-right (from ROI x=1294, y=830)
    [631, 829]   # Bottom-left (from ROI x=626, y=823)
])

# Desired size of the output image (width, height)
OUTPUT_WIDTH = 512
OUTPUT_HEIGHT = 512
DST_POINTS = np.float32([
    [0, 0],
    [OUTPUT_WIDTH - 1, 0],
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
    [0, OUTPUT_HEIGHT - 1]
])

OUTPUT_FILENAME = "warped_video.mp4"
FPS = 30

def main():
    """
    Loads all images in a folder, applies a perspective warp to each,
    and compiles them into an MP4 video.
    """
    # Find all images to process
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.png')))
    
    if not image_paths:
        print(f"Error: No .jpg or .png images found in '{IMAGE_FOLDER}'.")
        return

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for '{OUTPUT_FILENAME}'")
        return
        
    print(f"Processing {len(image_paths)} images from '{IMAGE_FOLDER}'...")

    # Calculate the perspective transform matrix (it's the same for all frames)
    matrix = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

    # Process each image
    for image_path in tqdm(image_paths, desc="Warping frames"):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {os.path.basename(image_path)}. Skipping.")
            continue

        # Apply the perspective warp
        warped_image = cv2.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        # Write the frame to the video
        video_writer.write(warped_image)

    # Release resources
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nSuccessfully created video '{OUTPUT_FILENAME}' at {FPS} FPS.")

if __name__ == "__main__":
    main() 