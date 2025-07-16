#!/usr/bin/env python3
"""
🎬 Frame-by-Frame Video Viewer
==============================

Allows you to step through the analyzed Fortnite Jam Battle video frame by frame
to review detection results and analyze performance.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

class FrameViewer:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.paused = True  # Start paused for frame-by-frame review
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.load_video()
    
    def load_video(self):
        """Load video and get properties"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video loaded: {self.video_path.name}")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.1f}")
        print(f"   Total frames: {self.total_frames}")
        print(f"   Duration: {self.total_frames/self.fps:.1f}s")
    
    def get_frame(self, frame_number):
        """Get specific frame by number"""
        if frame_number < 0 or frame_number >= self.total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame_number
            return frame
        return None
    
    def add_frame_info(self, frame):
        """Add frame information overlay"""
        # Create info panel
        info_height = 100  # Increased height for two lines of controls
        info_panel = np.zeros((info_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Frame info
        frame_text = f"Frame: {self.current_frame + 1}/{self.total_frames}"
        time_text = f"Time: {self.current_frame/self.fps:.2f}s"
        progress = (self.current_frame / self.total_frames) * 100
        progress_text = f"Progress: {progress:.1f}%"
        
        cv2.putText(info_panel, frame_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_panel, time_text, (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_panel, progress_text, (450, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Controls info - shorter text to fit better
        controls_text = "←/→ (frame) | ↑/↓ (10) | Space (play) | G (goto) | S (save) | Q (quit)"
        cv2.putText(info_panel, controls_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Additional controls on second line
        controls_text2 = "W/X (10 frames) | A/D (1 frame) | R (reset) | H (help)"
        cv2.putText(info_panel, controls_text2, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Combine with main frame
        combined = np.vstack([frame, info_panel])
        return combined
    
    def show_controls(self):
        """Display control instructions"""
        print("\n🎮 Frame Viewer Controls:")
        print("   ←/→     : Previous/Next frame")
        print("   ↑/↓     : Previous/Next 10 frames")
        print("   A/D     : Previous/Next frame (WASD)")
        print("   W/X     : Previous/Next 10 frames (WASD)")
        print("   Space   : Play/Pause")
        print("   P       : Play/Pause")
        print("   R       : Reset to frame 0")
        print("   G       : Go to specific frame")
        print("   S       : Save current frame")
        print("   Q       : Quit")
        print("   H       : Show this help")
        print()
    
    def save_frame(self, frame, frame_number):
        """Save current frame as image"""
        filename = f"frame_{frame_number:06d}.png"
        cv2.imwrite(filename, frame)
        print(f"💾 Saved frame {frame_number + 1} as {filename}")
    
    def go_to_frame(self):
        """Prompt user to go to specific frame"""
        try:
            frame_num = int(input(f"Enter frame number (0-{self.total_frames-1}): "))
            if 0 <= frame_num < self.total_frames:
                return frame_num
            else:
                print(f"❌ Frame number must be between 0 and {self.total_frames-1}")
                return self.current_frame
        except ValueError:
            print("❌ Please enter a valid number")
            return self.current_frame
    
    def run(self):
        """Main viewer loop"""
        window_name = f"Frame Viewer - {self.video_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set window size to video dimensions
        cv2.resizeWindow(window_name, self.width, self.height + 100)  # +100 for info panel
        
        # Show initial frame
        frame = self.get_frame(0)
        if frame is None:
            print("❌ Could not read first frame")
            return
        
        frame_with_info = self.add_frame_info(frame)
        cv2.imshow(window_name, frame_with_info)
        
        self.show_controls()
        
        while True:
            # Handle key events - use longer wait time to prevent blocking
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_controls()
            elif key == ord('p') or key == ord(' '):
                self.paused = not self.paused
                print(f"{'⏸️  Paused' if self.paused else '▶️  Playing'}")
            elif key == ord('r'):
                self.current_frame = 0
                frame = self.get_frame(0)
                if frame is not None:
                    frame_with_info = self.add_frame_info(frame)
                    cv2.imshow(window_name, frame_with_info)
                print("🔄 Reset to frame 0")
            elif key == ord('g'):
                new_frame = self.go_to_frame()
                if new_frame != self.current_frame:
                    frame = self.get_frame(new_frame)
                    if frame is not None:
                        frame_with_info = self.add_frame_info(frame)
                        cv2.imshow(window_name, frame_with_info)
            elif key == ord('s'):
                self.save_frame(frame, self.current_frame)
            elif key == ord('a') or key == 81:  # Left arrow
                if self.current_frame > 0:
                    frame = self.get_frame(self.current_frame - 1)
                    if frame is not None:
                        frame_with_info = self.add_frame_info(frame)
                        cv2.imshow(window_name, frame_with_info)
            elif key == ord('d') or key == 83:  # Right arrow
                if self.current_frame < self.total_frames - 1:
                    frame = self.get_frame(self.current_frame + 1)
                    if frame is not None:
                        frame_with_info = self.add_frame_info(frame)
                        cv2.imshow(window_name, frame_with_info)
            elif key == ord('w') or key == 82:  # Up arrow
                new_frame = max(0, self.current_frame - 10)
                if new_frame != self.current_frame:
                    frame = self.get_frame(new_frame)
                    if frame is not None:
                        frame_with_info = self.add_frame_info(frame)
                        cv2.imshow(window_name, frame_with_info)
            elif key == ord('x') or key == 84:  # Down arrow (changed from 's' to avoid conflict)
                new_frame = min(self.total_frames - 1, self.current_frame + 10)
                if new_frame != self.current_frame:
                    frame = self.get_frame(new_frame)
                    if frame is not None:
                        frame_with_info = self.add_frame_info(frame)
                        cv2.imshow(window_name, frame_with_info)
            
            # Auto-advance if playing
            if not self.paused:
                if self.current_frame < self.total_frames - 1:
                    frame = self.get_frame(self.current_frame + 1)
                    if frame is not None:
                        frame_with_info = self.add_frame_info(frame)
                        cv2.imshow(window_name, frame_with_info)
                else:
                    self.paused = True
                    print("⏹️  Reached end of video")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Frame viewer closed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Frame-by-frame video viewer")
    parser.add_argument("video", nargs="?", default="fortnitejam_analyzed.mp4", 
                       help="Video file to view (default: fortnitejam_analyzed_3class.mp4)")
    
    args = parser.parse_args()
    
    try:
        viewer = FrameViewer(args.video)
        viewer.run()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Make sure the analyzed video file exists in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 