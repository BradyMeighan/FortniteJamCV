#!/usr/bin/env python3
"""
GTuner Live Frame Analysis - FSM Version
=========================================

FSM-based live frame analysis for Fortnite Jam Battle using GTuner's CV framework.
Implements the two-stripe mental model with finite state machines for precise timing.
"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

class LightweightCNN(nn.Module):
    """
    Lightweight CNN optimized for real-time inference
    Target: <2ms inference time per ROI on RTX 3090
    """
    def __init__(self, num_classes=4, input_size=(72, 133)):  # Max ROI size
        super(LightweightCNN, self).__init__()
        # Efficient feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

from collections import defaultdict
from enum import Enum
import time

class LaneState(Enum):
    IDLE = "idle"
    TAP_HOLD = "tap_hold"
    LINE_HOLD = "line_hold"

class LaneEvent(Enum):
    PRESS = "PRESS"
    HOLD = "HOLD"
    RELEASE = "RELEASE"
    NONE = "NONE"

class LaneFSM:
    def __init__(self, lane_id, min_hold=2):
        self.lane_id = lane_id
        self.state = LaneState.IDLE
        self.frames_held = 0
        self.current_note_type = None
        self.min_hold = min_hold
        
        # Simplified line tracking (like the working GPC)
        self.is_line = False  # Is this lane currently a line?
        self.line_blank_count = 0  # Consecutive blanks for line end detection
        self.line_blank_threshold = 2  # Need 20 consecutive blanks to end a line (like GPC)
        
        # Anti-double-tap protection with both single-frame and multi-frame
        self.just_released = False  # Single-frame protection
        self.release_cooldown = 0  # Multi-frame cooldown
        self.release_cooldown_frames = 2  # Increased to 5-frame cooldown for maximum double-tap prevention
        
        # Additional protection against rapid FSM triggers
        self.last_press_frame = 0  # Track when we last triggered a PRESS event

    def step(self, label, liftoff_triggered=False, timeout_triggered=False, frame_counter=0):
        """Process one frame and return the event to emit"""
        event = LaneEvent.NONE
        
        # Apply release cooldown
        if self.release_cooldown > 0:
            self.release_cooldown -= 1
        
        # Main FSM logic with smart double-tap prevention
        if self.state == LaneState.IDLE:
            # Check for double-tap prevention
            if label in ["note", "line"]:
                # Single-frame protection (immediate)
                if self.just_released:
                    self.just_released = False  # Reset flag
                    return LaneEvent.NONE
                
                # Multi-frame cooldown protection - strict enforcement
                if self.release_cooldown > 0:
                    return LaneEvent.NONE
                
                # Additional protection: prevent PRESS events too close together
                if frame_counter - self.last_press_frame < 8:  # 8 frames minimum between presses
                    return LaneEvent.NONE
            
            if label == "note":
                self.state = LaneState.TAP_HOLD
                self.frames_held = 0
                self.current_note_type = "note"
                self.is_line = False
                self.just_released = False
                self.release_cooldown = 0  # Clear cooldown on new press
                self.last_press_frame = frame_counter  # Track this press
                event = LaneEvent.PRESS
            elif label == "line":
                self.state = LaneState.LINE_HOLD
                self.frames_held = 0
                self.current_note_type = "line"
                self.is_line = True
                self.line_blank_count = 0
                self.just_released = False
                self.release_cooldown = 0  # Clear cooldown on new press
                self.last_press_frame = frame_counter  # Track this press
                event = LaneEvent.PRESS
                
        elif self.state == LaneState.TAP_HOLD:
            self.frames_held += 1
            
            # Check for note → line transition (convert to line mode)
            if label == "line":
                self.state = LaneState.LINE_HOLD
                self.current_note_type = "line"
                self.is_line = True
                self.line_blank_count = 0
                self.just_released = False
                event = LaneEvent.HOLD  # Keep holding, don't release
            # Check for BPM timeout or liftoff
            elif timeout_triggered or liftoff_triggered:
                self.state = LaneState.IDLE
                self.just_released = True
                self.release_cooldown = self.release_cooldown_frames
                event = LaneEvent.RELEASE
            # Release after minimum hold time with blank
            elif self.frames_held >= self.min_hold and label == "blank":
                self.state = LaneState.IDLE
                self.just_released = True
                self.release_cooldown = self.release_cooldown_frames
                event = LaneEvent.RELEASE
            elif label == "liftoff":
                self.state = LaneState.IDLE
                self.just_released = True
                self.release_cooldown = self.release_cooldown_frames
                event = LaneEvent.RELEASE
            else:
                self.just_released = False
                event = LaneEvent.HOLD
                
        elif self.state == LaneState.LINE_HOLD:
            self.frames_held += 1
            
            # Simplified line logic like the working GPC
            if label == "blank":
                self.line_blank_count += 1
                # Only release after many consecutive blanks (like GPC)
                if self.line_blank_count >= self.line_blank_threshold:
                    self.state = LaneState.IDLE
                    self.just_released = True
                    self.release_cooldown = self.release_cooldown_frames
                    self.is_line = False
                    self.line_blank_count = 0
                    event = LaneEvent.RELEASE
                else:
                    # Still holding during blank count
                    self.just_released = False
                    event = LaneEvent.HOLD
            elif label in ["note", "line"]:
                # Any detection resets blank count (line continues)
                self.line_blank_count = 0
                self.just_released = False
                event = LaneEvent.HOLD
            # Check for BPM timeout or liftoff (override blank counting)
            elif timeout_triggered or liftoff_triggered:
                self.state = LaneState.IDLE
                self.just_released = True
                self.release_cooldown = self.release_cooldown_frames
                self.is_line = False
                self.line_blank_count = 0
                event = LaneEvent.RELEASE
            elif label == "liftoff":
                self.state = LaneState.IDLE
                self.just_released = True
                self.release_cooldown = self.release_cooldown_frames
                self.is_line = False
                self.line_blank_count = 0
                event = LaneEvent.RELEASE
            else:
                self.just_released = False
                event = LaneEvent.HOLD
        
        return event

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
        self.width = width
        self.height = height
        
        # --- TIMING CONFIGURATION ---
        # Adjust this value to fix early/late timing:
        # - Increase to delay button presses (if hitting too early)
        # - Decrease to speed up button presses (if hitting too late)
        # Each frame = ~16.67ms at 60fps
        self.timing_offset_frames = 0  # 12 frames = ~200ms delay (was 9 frames/150ms, still -47ms early)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Perspective Warp Configuration ---
        self.SRC_POINTS = np.float32([[769, 469], [1156, 465], [1301, 836], [631, 829]])
        self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT = 512, 512
        self.DST_POINTS = np.float32([[0, 0], [self.OUTPUT_WIDTH - 1, 0], [self.OUTPUT_WIDTH - 1, self.OUTPUT_HEIGHT - 1], [0, self.OUTPUT_HEIGHT - 1]])
        self.perspective_matrix = cv2.getPerspectiveTransform(self.SRC_POINTS, self.DST_POINTS)

        # --- ROI Configuration (on warped image) ---
        # Top stripe (early): ROIs 0-4, Bottom stripe (late): ROIs 5-9
        self.rois = [
            {'x': 0, 'y': 61, 'w': 87, 'h': 48, 'id': 1}, {'x': 104, 'y': 60, 'w': 91, 'h': 50, 'id': 2},
            {'x': 214, 'y': 60, 'w': 91, 'h': 50, 'id': 3}, {'x': 321, 'y': 55, 'w': 90, 'h': 57, 'id': 4},
            {'x': 430, 'y': 61, 'w': 80, 'h': 48, 'id': 5}, {'x': 2, 'y': 425, 'w': 87, 'h': 37, 'id': 6},
            {'x': 107, 'y': 424, 'w': 91, 'h': 38, 'id': 7}, {'x': 218, 'y': 422, 'w': 87, 'h': 40, 'id': 8},
            {'x': 324, 'y': 424, 'w': 89, 'h': 40, 'id': 9}, {'x': 433, 'y': 420, 'w': 78, 'h': 43, 'id': 10},
        ]

        # --- Consensus Mechanism ---
        self.consensus_frames = 2
        self.detection_history = [defaultdict(int) for _ in self.rois]
        self.stable_state = [None] * len(self.rois)
        
        # --- Liftoff Prediction Cache ---
        self.liftoff_countdown = [0] * 5  # Frames until liftoff should trigger release
        self.liftoff_detected = [False] * 5  # Whether liftoff was seen in top ROI
        
        # --- Per-lane Finite State Machines ---
        self.lane_fsms = [LaneFSM(i, min_hold=2) for i in range(5)]
        
        # --- Model and Transforms ---
        self.model, self.class_names = self._load_model("best_note_detector.pth")
        self.transform = self._get_transform()
        
        # --- Visualization ---
        self.colors = {
            'note': (0, 255, 0), 
            'line': (0, 255, 255), 
            'liftoff': (255, 165, 0), 
            'blank': (128, 128, 128)
        }
        self.event_colors = {
            LaneEvent.PRESS: (0, 0, 255),    # Red
            LaneEvent.HOLD: (0, 255, 255),   # Yellow
            LaneEvent.RELEASE: (255, 0, 0),  # Blue
            LaneEvent.NONE: (100, 100, 100)  # Gray
        }
        self.state_colors = {
            LaneState.IDLE: (100, 100, 100),
            LaneState.TAP_HOLD: (0, 255, 0),
            LaneState.LINE_HOLD: (0, 255, 255)
        }

        # --- Performance Tracking ---
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_counter = []
        
        # --- Button State Tracking ---
        self.button_states = [0] * 5  # Current button press states
        self.last_events = [LaneEvent.NONE] * 5  # Last events for each lane
        
        # --- Timing Offset for Perfect Notes ---
        self.pending_presses = [[] for _ in range(5)]  # Queue of pending button presses per lane
        self.pending_releases = [[] for _ in range(5)]  # Queue of pending button releases per lane
        
        # --- BPM Detection for Timeout Logic ---
        self.bpm_calibration_notes = 10  # Number of notes to calibrate BPM
        self.note_durations = [[] for _ in range(5)]  # Track note durations per lane
        self.note_start_frames = [0] * 5  # When current note started
        self.average_note_duration = [0] * 5  # Average note duration per lane
        self.bmp_calibrated = [False] * 5  # Whether BPM is calibrated per lane
        self.timeout_multiplier = 1.15  # More aggressive timeout (was 1.5)
        self.min_note_timeout = 7  # Minimum frames before timeout (for fast songs)

        print("GTuner FSM Live Analyzer Initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Classes: {self.class_names}")
        print(f"   - Frame resolution: {width}x{height}")

    def __del__(self):
        pass

    def _load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = LightweightCNN(num_classes=len(checkpoint.get('class_names', ['note', 'line', 'liftoff', 'blank']))).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            class_names = checkpoint['class_names']
            print(f"Model loaded successfully.")
            return model, class_names
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((72, 133)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_on_warped_frame(self, warped_frame):
        """Get raw CNN predictions for all ROIs"""
        roi_crops = []
        for roi in self.rois:
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            crop = warped_frame[y:y+h, x:x+w]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            roi_crops.append(self.transform(crop_pil))
        
        batch = torch.stack(roi_crops).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            predictions = torch.argmax(outputs, dim=1)
        
        return [self.class_names[p] for p in predictions.cpu().numpy()]

    def update_consensus(self, raw_predictions):
        """Update consensus-based stable state for all ROIs"""
        for i, pred_class in enumerate(raw_predictions):
            history = self.detection_history[i]

            # Disabled fast note bypass - timing offset system handles this better now
            # This was potentially causing double-taps
            
            # If the current prediction is 'blank', it's immediate
            if pred_class == 'blank':
                self.stable_state[i] = 'blank'
                # Reset history for this ROI
                for key in list(history.keys()):
                    history[key] = 0
                continue

            # For non-blank predictions, use consensus
            history[pred_class] += 1

            # Reset counter for other classes
            for key in list(history.keys()):
                if key != pred_class:
                    history[key] = 0

            # Check if consensus reached
            if history[pred_class] >= self.consensus_frames:
                self.stable_state[i] = pred_class
            # If no consensus yet, keep previous stable state

    def update_liftoff_prediction(self):
        """Update liftoff prediction based on top stripe"""
        for lane in range(5):
            top_label = self.stable_state[lane]
            
            # If we see liftoff in top ROI, start countdown
            if top_label == "liftoff":
                if not self.liftoff_detected[lane]:
                    self.liftoff_countdown[lane] = 8  # ~8 frames from top to bottom
                    self.liftoff_detected[lane] = True
            
            # Countdown if liftoff detected
            if self.liftoff_detected[lane]:
                self.liftoff_countdown[lane] -= 1
                if self.liftoff_countdown[lane] <= 0:
                    self.liftoff_detected[lane] = False

    def update_bpm_detection(self, lane_events):
        """Update BPM detection based on FSM events"""
        for lane, event in enumerate(lane_events):
            if event == LaneEvent.PRESS:
                # Note started
                self.note_start_frames[lane] = self.frame_counter
                
            elif event == LaneEvent.RELEASE:
                # Note ended, calculate duration
                if self.note_start_frames[lane] > 0:
                    duration = self.frame_counter - self.note_start_frames[lane]
                    self.note_durations[lane].append(duration)
                    
                    # Keep only recent durations for calibration
                    if len(self.note_durations[lane]) > self.bpm_calibration_notes:
                        self.note_durations[lane].pop(0)
                    
                    # Calculate average if we have enough samples
                    if len(self.note_durations[lane]) >= 3:
                        self.average_note_duration[lane] = sum(self.note_durations[lane]) / len(self.note_durations[lane])
                        
                        if len(self.note_durations[lane]) >= self.bpm_calibration_notes:
                            if not self.bmp_calibrated[lane]:
                                self.bmp_calibrated[lane] = True
                    
                    self.note_start_frames[lane] = 0

    def check_note_timeout(self, lane):
        """Check if current note has exceeded timeout threshold"""
        if self.note_start_frames[lane] == 0:
            return False
        
        current_duration = self.frame_counter - self.note_start_frames[lane]
        
        # Use minimum timeout for fast songs or BPM-based timeout if calibrated
        if self.bmp_calibrated[lane]:
            timeout_threshold = max(self.min_note_timeout, 
                                  self.average_note_duration[lane] * self.timeout_multiplier)
        else:
            # For uncalibrated lanes, use minimum timeout for fast notes
            timeout_threshold = self.min_note_timeout
        
        if current_duration >= timeout_threshold:
            return True
        
        return False

    def process_lane_fsms(self):
        """Process bottom stripe through FSMs and return events"""
        lane_events = []
        
        for lane in range(5):
            # Get bottom ROI label
            bottom_roi_idx = lane + 5
            bottom_label = self.stable_state[bottom_roi_idx]
            
            # Check if liftoff countdown expired (predicted release)
            liftoff_triggered = self.liftoff_countdown[lane] == 0 and self.liftoff_detected[lane]
            
            # Check for BPM timeout (grouped notes)
            timeout_triggered = self.check_note_timeout(lane)
            
            # FSM input: bottom stripe + liftoff prediction
            fsm_input = bottom_label if bottom_label else "blank"
            
            # Process through FSM with liftoff info and timeout
            event = self.lane_fsms[lane].step(fsm_input, liftoff_triggered, timeout_triggered, self.frame_counter)
            lane_events.append(event)
        
        return lane_events

    def update_button_states(self, lane_events):
        """Update button states based on FSM events with timing offset"""
        
        # Process current FSM events into pending queues
        for lane, event in enumerate(lane_events):
            target_frame = self.frame_counter + self.timing_offset_frames
            
            if event == LaneEvent.PRESS:
                # Aggressive double-tap prevention: only queue if no recent pending presses
                recent_presses = [f for f in self.pending_presses[lane] 
                                if f > self.frame_counter - 10]  # Check last 10 frames
                
                if not recent_presses:  # Only queue if no recent presses pending
                    # Queue a button press for future execution
                    self.pending_presses[lane].append(target_frame)
                    # Clear any conflicting releases
                    self.pending_releases[lane] = [f for f in self.pending_releases[lane] if f > target_frame]
                
            elif event == LaneEvent.RELEASE:
                # Queue a button release for future execution
                self.pending_releases[lane].append(target_frame)
                # Clear any conflicting presses that are after this release
                self.pending_presses[lane] = [f for f in self.pending_presses[lane] if f > target_frame]
            
            # HOLD events don't change timing - keep current state
            
            self.last_events[lane] = event
        
        # Process pending button actions that are ready to execute
        for lane in range(5):
            # Check for pending presses ready to execute
            ready_presses = [f for f in self.pending_presses[lane] if f <= self.frame_counter]
            if ready_presses:
                # Only execute the earliest press to prevent double-taps
                earliest_press = min(ready_presses)
                if earliest_press <= self.frame_counter:
                    self.button_states[lane] = 1
                    # Remove ALL pending presses to prevent double execution
                    self.pending_presses[lane] = []
            
            # Check for pending releases ready to execute
            ready_releases = [f for f in self.pending_releases[lane] if f <= self.frame_counter]
            if ready_releases:
                self.button_states[lane] = 0
                # Remove executed releases
                self.pending_releases[lane] = [f for f in self.pending_releases[lane] if f > self.frame_counter]
            
            # Handle immediate state based on FSM state for HOLD events
            fsm_state = self.lane_fsms[lane].state
            if lane_events[lane] == LaneEvent.HOLD:
                # For HOLD events, maintain button state if we're in a holding state
                if fsm_state in [LaneState.TAP_HOLD, LaneState.LINE_HOLD]:
                    # Only set if no pending releases are about to execute
                    if not ready_releases:
                        self.button_states[lane] = 1
            elif lane_events[lane] == LaneEvent.NONE:
                # For NONE events, only release if FSM is IDLE and no pending actions
                if (fsm_state == LaneState.IDLE and 
                    not self.pending_presses[lane] and 
                    not ready_presses):
                    self.button_states[lane] = 0

    def draw_overlays(self, warped_frame, lane_events):
        """Draw visualization overlays"""
        overlay = warped_frame.copy()
        
        # Draw ROI boxes with states
        for i, roi in enumerate(self.rois):
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            
            # Determine color based on stable state
            label = self.stable_state[i] if self.stable_state[i] else 'blank'
            color = self.colors.get(label, (255, 255, 255))
            
            # Draw ROI rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            cv2.putText(overlay, label.upper(), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw lane information and events
        for lane in range(5):
            lane_x = 50 + lane * 90
            info_y = 30
            
            # Draw lane number and liftoff prediction
            liftoff_info = f"L{self.liftoff_countdown[lane]}" if self.liftoff_detected[lane] else "---"
            lane_text = f"Lane{lane+1} ({liftoff_info})"
            cv2.putText(overlay, lane_text, (lane_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw FSM state
            fsm_state = self.lane_fsms[lane].state
            state_color = self.state_colors.get(fsm_state, (255, 255, 255))
            note_type = self.lane_fsms[lane].current_note_type or ""
            state_text = f"{fsm_state.value.upper()} ({note_type})" if note_type else fsm_state.value.upper()
            cv2.putText(overlay, state_text, (lane_x, info_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1)
            
            # Draw current event
            event = lane_events[lane]
            event_color = self.event_colors.get(event, (255, 255, 255))
            cv2.putText(overlay, event.value, (lane_x, info_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, event_color, 1)
            
            # Draw button state
            button_color = (0, 255, 0) if self.button_states[lane] else (128, 128, 128)
            button_text = "BTN ON" if self.button_states[lane] else "BTN OFF"
            cv2.putText(overlay, button_text, (lane_x, info_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, button_color, 1)
            
            # Draw BPM calibration status
            if self.bmp_calibrated[lane]:
                bmp_color = (0, 255, 0)
                bmp_text = f"BPM:{self.average_note_duration[lane]:.1f}f"
            else:
                bmp_color = (128, 128, 128)
                bmp_text = f"CAL:{len(self.note_durations[lane])}/{self.bpm_calibration_notes}"
            cv2.putText(overlay, bmp_text, (lane_x, info_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, bmp_color, 1)
            
            # Draw timing offset status
            pending_presses = len(self.pending_presses[lane])
            pending_releases = len(self.pending_releases[lane])
            if pending_presses > 0 or pending_releases > 0:
                timing_color = (255, 255, 0)  # Yellow for pending actions
                timing_text = f"PENDING:P{pending_presses}/R{pending_releases}"
                cv2.putText(overlay, timing_text, (lane_x, info_y + 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, timing_color, 1)
            
            # Draw line hold status if in line hold state (moved down)
            if self.lane_fsms[lane].state == LaneState.LINE_HOLD:
                hold_frames = self.lane_fsms[lane].frames_held
                blank_count = self.lane_fsms[lane].line_blank_count
                blank_threshold = self.lane_fsms[lane].line_blank_threshold
                
                # Show blank count status for lines
                if blank_count > 0:
                    status_color = (255, 165, 0)  # Orange for counting blanks
                    status_text = f"BLANK:{blank_count}/{blank_threshold}"
                else:
                    status_color = (0, 255, 0)  # Green for stable line
                    status_text = f"LINE:{hold_frames}f"
                
                cv2.putText(overlay, status_text, (lane_x, info_y + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
        
        # Draw frame info
        cv2.putText(overlay, "TOP STRIPE (Early Detection)", (10, self.OUTPUT_HEIGHT - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, "BOTTOM STRIPE (Action Zone)", (10, self.OUTPUT_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay

    def add_info_panel(self, frame, lane_events):
        """Add information overlay to frame"""
        overlay_frame = frame.copy()
        
        # Create semi-transparent overlay area at the top
        overlay_height = 120
        overlay_area = overlay_frame[0:overlay_height, 0:overlay_frame.shape[1]]
        overlay_area[:] = overlay_area[:] * 0.3  # Darken background
        
        # Frame counter and FPS info
        frame_text = f"Frame: {self.frame_counter}"
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        
        cv2.putText(overlay_frame, frame_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay_frame, fps_text, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Model info
        model_info = f"FSM Model | Consensus: {self.consensus_frames}f | Timing Offset: {self.timing_offset_frames}f ({self.timing_offset_frames * 16.67:.0f}ms)"
        cv2.putText(overlay_frame, model_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Active events summary
        active_events = []
        for i, event in enumerate(lane_events):
            if event != LaneEvent.NONE:
                active_events.append(f"L{i+1}:{event.value}")
        
        if active_events:
            summary = " | ".join(active_events)
            cv2.putText(overlay_frame, f"Events: {summary}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Button states
        button_summary = []
        for i, state in enumerate(self.button_states):
            if state:
                button_summary.append(f"L{i+1}")
        
        if button_summary:
            buttons_text = f"Buttons: {', '.join(button_summary)}"
        else:
            buttons_text = "Buttons: None"
        
        cv2.putText(overlay_frame, buttons_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # BPM calibration status
        calibrated_lanes = [i+1 for i in range(5) if self.bmp_calibrated[i]]
        if calibrated_lanes:
            bpm_text = f"BPM Calibrated: Lanes {', '.join(map(str, calibrated_lanes))}"
        else:
            bpm_text = "BPM Calibration: In progress..."
        
        cv2.putText(overlay_frame, bpm_text, (400, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return overlay_frame

    def _convert_to_gcv_format(self, lane_events):
        """Convert FSM events to GCV data format for GPC script"""
        gcvdata = bytearray(8)
        
        # Button states for lanes 1-5 (bytes 0-4)
        for i in range(5):
            gcvdata[i] = self.button_states[i]
        
        # Metadata (bytes 5-7)
        gcvdata[5] = 1  # System ready
        gcvdata[6] = self.frame_counter & 0xFF  # Frame counter low byte
        gcvdata[7] = (self.frame_counter >> 8) & 0xFF  # Frame counter high byte
        
        # Debug log active buttons (removed for performance)
        
        return gcvdata

    def process(self, frame):
        """Main processing function called by GTuner CV framework"""
        try:
            # Convert raw frame data to numpy array
            frame_array = np.frombuffer(frame, dtype=np.uint8)
            frame_array = frame_array.reshape((self.height, self.width, 3))
            
            # Increment frame counter
            self.frame_counter += 1
            
            # 1. Perspective Warp
            warped_frame = cv2.warpPerspective(frame_array, self.perspective_matrix, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))

            # 2. Get raw CNN predictions
            raw_predictions = self.predict_on_warped_frame(warped_frame)

            # 3. Update consensus-based stable state
            self.update_consensus(raw_predictions)

            # 4. Update liftoff prediction from top stripe
            self.update_liftoff_prediction()

            # 5. Process lane FSMs and get events
            lane_events = self.process_lane_fsms()

            # 6. Update BPM detection based on events
            self.update_bpm_detection(lane_events)

            # 7. Update button states based on events
            self.update_button_states(lane_events)

            # 8. Draw overlays
            display_frame = self.draw_overlays(warped_frame, lane_events)
            
            # 9. Add info panel
            final_frame = self.add_info_panel(display_frame, lane_events)

            # 10. Convert to GCV format
            gcvdata = self._convert_to_gcv_format(lane_events)

            # Performance tracking
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.frame_counter / (current_time - self.last_fps_time)
                self.fps_counter.append(fps)
                if len(self.fps_counter) > 10:
                    self.fps_counter.pop(0)
                
                if self.frame_counter % 60 == 0:
                    avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
                    print(f"[PERF] Frame {self.frame_counter}: {avg_fps:.1f} FPS | FSM: READY")
            
            self.last_fps_time = current_time

            # Return processed frame and GCV data
            return final_frame, gcvdata
            
        except Exception as e:
            print(f"Error in process: {e}")
            # Return empty GCV data on error
            return frame, bytearray([0, 0, 0, 0, 0, 1, 0, 0]) 