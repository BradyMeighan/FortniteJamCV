import cv2
import os
import glob
import json

# --- Configuration ---
SOURCE_IMAGE_FOLDER = 'frames'
OUTPUT_DATA_FOLDER = 'training_data_classified'
CLASSIFICATION_STATE_FILE = os.path.join(OUTPUT_DATA_FOLDER, 'classification_state.json')

# The ROIs you defined for the warped 512x512 image
ROIS = [
    {'x': 0, 'y': 61, 'w': 87, 'h': 48, 'id': 1},
    {'x': 104, 'y': 60, 'w': 91, 'h': 50, 'id': 2},
    {'x': 214, 'y': 60, 'w': 91, 'h': 50, 'id': 3},
    {'x': 321, 'y': 55, 'w': 90, 'h': 57, 'id': 4},
    {'x': 430, 'y': 61, 'w': 80, 'h': 48, 'id': 5},
    {'x': 2, 'y': 425, 'w': 87, 'h': 37, 'id': 6},
    {'x': 107, 'y': 424, 'w': 91, 'h': 38, 'id': 7},
    {'x': 218, 'y': 422, 'w': 87, 'h': 40, 'id': 8},
    {'x': 324, 'y': 424, 'w': 89, 'h': 40, 'id': 9},
    {'x': 433, 'y': 420, 'w': 78, 'h': 43, 'id': 10},
]

CLASSES = {
    '1': 'blank',
    '2': 'note',
    '3': 'liftoff',
    '4': 'line'
}

# --- Colors ---
ROI_COLOR = (255, 150, 0) # Default Blue
SELECTED_COLOR = (0, 255, 255) # Yellow
CLASS_COLORS = {
    'blank': (128, 128, 128), # Gray
    'note': (0, 255, 0),       # Green
    'liftoff': (255, 0, 0),   # Blue
    'line': (0, 0, 255)         # Red
}

# --- Global State ---
image_paths = []
classifications = {} # Holds the state, e.g., {'frame_001.jpg': {0: 'note', 1: 'blank'}}
current_frame_index = 0
selected_roi_index = None

def load_state():
    """Loads classification state from the JSON file."""
    global classifications
    if os.path.exists(CLASSIFICATION_STATE_FILE):
        with open(CLASSIFICATION_STATE_FILE, 'r') as f:
            classifications = json.load(f)
        print(f"Loaded existing classification state from '{CLASSIFICATION_STATE_FILE}'")

def save_state():
    """Saves the current classification state to a JSON file."""
    os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)
    with open(CLASSIFICATION_STATE_FILE, 'w') as f:
        json.dump(classifications, f, indent=2)
    print(f"Classification state saved to '{CLASSIFICATION_STATE_FILE}'")

def get_roi_at_pos(x, y):
    """Checks if a click is on an ROI and returns its index."""
    for i, roi in enumerate(ROIS):
        if roi['x'] <= x <= roi['x'] + roi['w'] and roi['y'] <= y <= roi['y'] + roi['h']:
            return i
    return None

def handle_mouse_click(event, x, y, flags, param):
    """Selects an ROI when clicked."""
    global selected_roi_index
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_roi_index = get_roi_at_pos(x, y)

def classify_and_save(image, frame_basename, roi_index, class_name):
    """Crops the ROI, saves it, and updates the state."""
    roi = ROIS[roi_index]
    
    # Define paths
    class_folder = os.path.join(OUTPUT_DATA_FOLDER, class_name)
    os.makedirs(class_folder, exist_ok=True)
    
    # Create a unique filename
    frame_number = os.path.splitext(frame_basename)[0]
    output_filename = f"{frame_number}_roi_{roi['id']}.jpg"
    output_path = os.path.join(class_folder, output_filename)
    
    # Crop and save
    cropped_image = image[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    cv2.imwrite(output_path, cropped_image)
    
    # Update state
    if frame_basename not in classifications:
        classifications[frame_basename] = {}
    classifications[frame_basename][str(roi_index)] = class_name
    
    print(f"Saved: {output_path}")

def main():
    global image_paths, current_frame_index, selected_roi_index

    # Load data
    image_paths = sorted(glob.glob(os.path.join(SOURCE_IMAGE_FOLDER, '*.jpg')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(SOURCE_IMAGE_FOLDER, '*.png')))
    
    if not image_paths:
        print(f"Error: No images found in '{SOURCE_IMAGE_FOLDER}'.")
        return
        
    load_state()

    # Setup window
    cv2.namedWindow("Classifier Tool")
    cv2.setMouseCallback("Classifier Tool", handle_mouse_click)

    while True:
        # Load image and frame info
        frame_path = image_paths[current_frame_index]
        frame_basename = os.path.basename(frame_path)
        image = cv2.imread(frame_path)
        display_image = image.copy()
        
        frame_classifications = classifications.get(frame_basename, {})

        # Draw all ROIs
        for i, roi in enumerate(ROIS):
            class_name = frame_classifications.get(str(i))
            color = CLASS_COLORS.get(class_name, ROI_COLOR)
            thickness = 3 if i == selected_roi_index else 2
            
            p1 = (roi['x'], roi['y'])
            p2 = (roi['x'] + roi['w'], roi['y'] + roi['h'])

            # Highlight if selected
            if i == selected_roi_index:
                cv2.rectangle(display_image, (p1[0]-2, p1[1]-2), (p2[0]+2, p2[1]+2), SELECTED_COLOR, thickness)
            
            cv2.rectangle(display_image, p1, p2, color, 2)
            
            label = class_name or str(roi['id'])
            cv2.putText(display_image, label, (p1[0], p1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display info text
        info_text_1 = f"{frame_basename} ({current_frame_index + 1}/{len(image_paths)})"
        info_text_2 = "[A/D] Nav | [1-4] Classify | [S] Save State | [Q] Quit & Save"
        cv2.putText(display_image, info_text_1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(display_image, info_text_1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(display_image, info_text_2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(display_image, info_text_2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.imshow("Classifier Tool", display_image)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            current_frame_index = (current_frame_index + 1) % len(image_paths)
            selected_roi_index = None # Deselect when changing frames
        elif key == ord('a'):
            current_frame_index = (current_frame_index - 1 + len(image_paths)) % len(image_paths)
            selected_roi_index = None # Deselect when changing frames
        elif key == ord('s'):
            save_state()
        elif chr(key) in CLASSES:
            if selected_roi_index is not None:
                class_name = CLASSES[chr(key)]
                classify_and_save(image, frame_basename, selected_roi_index, class_name)
    
    save_state()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 