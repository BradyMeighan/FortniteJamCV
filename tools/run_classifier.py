#!/usr/bin/env python3
"""
🎸 Fortnite Jam Battle ROI Classifier
====================================

This script helps you classify ROIs (Regions of Interest) from Fortnite Jam Battle
gameplay frames to train a computer vision model for automated gameplay.

Usage:
    python run_classifier.py

Controls:
    n = note (musical note detected)
    l = line (note line/trail)  
    o = liftoff (note release/liftoff)
    b = blank (nothing there)
    s = skip entire frame
    
    Arrow keys = navigate between ROIs and frames
    q = quit

The script will:
1. Show each frame with 5 ROIs marked (one per lane)
2. Highlight the current ROI being classified
3. Show an enlarged view of the current ROI
4. Save classified ROI crops to organized folders
5. Track progress and allow resuming
"""

from classifier_tool import ROIClassifier

def main():
    print(__doc__)
    
    # You can customize these paths if needed
    frames_folder = "classifier"      # Folder containing your frame_*.jpg files
    output_folder = "training_data"   # Where classified data will be saved
    
    try:
        classifier = ROIClassifier(frames_folder, output_folder)
        classifier.run()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure the '{frames_folder}' folder exists and contains frame_*.jpg files")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 