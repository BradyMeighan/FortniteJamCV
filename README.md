# Fortnite Jam - AI-Powered Game Assistant

An AI-powered system that uses computer vision to detect notes in Fortnite Jam and automatically presses the corresponding buttons using a Titan Two controller adapter.

## 🎮 Overview

This project combines computer vision, machine learning, and controller automation to create an AI assistant for Fortnite Festival. The system:

1. **Captures game footage** using screen capture
2. **Analyzes frames** using a trained CNN model to detect notes
3. **Predicts timing** and automatically presses the correct buttons
4. **Uses FSM (Finite State Machine)** for robust note detection and timing

## 📁 Project Structure

```
fortnite-jam/
├── src/                    # Main source code
│   ├── training/           # Model training scripts
│   │   └── train_model.py  # CNN training script
│   ├── analysis/           # Live analysis and detection
│   │   └── gtuner_live_analysis_fsm.py  # Main FSM-based analyzer
│   └── utils/              # Utility functions
│       ├── perspective_warp.py  # Image perspective correction
│       ├── roi.py          # Region of interest utilities
│       └── frames.py       # Frame processing utilities
├── models/                 # Trained neural network models
│   └── best_note_detector.pth  # Trained CNN model
├── training_data/          # Training datasets
│   ├── training_data_classified/  # Manually classified training data
│   └── training_data_augmented/   # Augmented training data
├── tools/                  # Data preparation and analysis tools
│   ├── classifier_tool.py  # Manual data classification tool
│   ├── augment_dataset.py  # Data augmentation script
│   ├── run_classifier.py   # Classification runner
│   └── frame_viewer.py     # Frame visualization tool
├── gpc/                    # GPC controller scripts
│   └── FortniteJamFSM.gpc  # Main controller script
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- **Titan Two Controller Adapter** (required for controller automation)
- **GTuner IV Software** (for uploading GPC scripts)
- **Python 3.8+** with required packages
- **Fortnite Festival** game

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fortnite-jam
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Titan Two:**
   - Connect your Titan Two adapter
   - Install GTuner IV software and setup Computer Vision
   - Upload `gpc/FortniteJamFSM.gpc` to your Titan Two

### Usage

1. **Start the live analysis:**
   - Configure your video input as 1920x1080p @ 60 fps
   - Start the Gtuner Live Analysis FSM file in the computer visions ection of GTuner IV

2. **Configure your game:**
   - Go to Settings > Game > Festival and reduce latency to start out. Try 0 and adjsut based on your avergae input altency at the end of a song. (Mine is at -150 and im acheieving 95% perfect notes with very low latency. If you need to adjsut this further than what is allwoed byt he agme there is a mnaul altency value you can adjust inthe gpc script called INPUT_DELAY)
   - Ensure the game is visible and not minimized

3. **Start playing:**
   - The system will automatically detect notes
   - After loading the script into your Titan Two you must click dpad down to start acceptign button preses fromt eh comptuer vision script
   - It is best practice to nto rpess this until you see a blank game or right before the ntoes start coming to prevent button spamming while the game is transitioning.
   - Monitor the console for detection feedback

## 🧠 How It Works

### 1. Computer Vision Pipeline
- **Screen Capture**: Continuously captures game footage
- **ROI Extraction**: Focuses on the note detection area
- **Preprocessing**: Applies perspective correction and normalization
- **CNN Classification**: Uses trained model to classify frame content

### 2. Note Detection Classes
- **Blank**: No note present
- **Note**: Standard note to press
- **Line**: Hold note (sustain)
- **Liftoff**: End of hold note

### 3. FSM (Finite State Machine)
The system uses a state machine to handle different game states:
- **Idle**: Waiting for notes
- **Note Detection**: Active note processing
- **Hold Processing**: Managing sustained notes
- **Error Recovery**: Handling detection failures

### 4. Controller Integration
- **GPC Script**: Handles button presses and timing
- **Communication**: Python ↔ Titan Two via USB
- **Synchronization**: Ensures precise timing with game

## 📊 Model Performance

The trained CNN model achieves:
- **Accuracy**: ~95% on test data
- **Inference Speed**: ~30ms per frame
- **Real-time Performance**: 60+ FPS analysis

## ⚠️ Important Notes

- **Fair Play**: This tool is for educational purposes. Check game terms of service.
- **Performance**: Results may vary based on hardware and game settings.
- **Configuration**: May require tuning for different screen resolutions and game setups.

## 🔧 Known Issues & Tuning Guide

### Known Issues

1. **Long Spaces Between Liftoffs**: The script may struggle with songs that have extended periods between note releases (liftoffs). This is a limitation of the current FSM implementation and may require manual intervention for certain song patterns.

2. **Performance Dependency**: The script's performance is entirely reliant on proper tuning of:
   - **Video latency** in Fortnite Festival settings
   - **Input delay** configuration in the Titan Two script
   - **Hardware performance** and system optimization

### Critical Tuning Variables

#### Fortnite Festival Settings
- **Video Latency**: Start with 0ms and adjust based on your average input latency at the end of songs
  - If hitting too early: Increase video latency
  - If hitting too late: Decrease video latency
  - Recommended range: -200ms to +100ms

#### Titan Two Script Variables (`gpc/FortniteJamFSM.gpc`)
```c
#define INPUT_DELAY 6  // 10ms delay before button presses
```
- **INPUT_DELAY**: Fine-tune button press timing
  - Increase for early hits
  - Decrease for late hits
  - Each unit ≈ 1.67ms at 1000fps

#### Python Script Variables (`gtuner_live_analysis_fsm.py`)

**Timing Configuration:**
```python
self.timing_offset_frames = 0  # Adjust for early/late timing
```
- **timing_offset_frames**: Frame-based timing offset
  - Increase to delay button presses (if hitting too early)
  - Decrease to speed up button presses (if hitting too late)
  - Each frame = ~16.67ms at 60fps

**FSM Configuration:**
```python
self.min_hold = 2  # Minimum frames to hold a note
self.line_blank_threshold = 2  # Consecutive blanks to end line
self.release_cooldown_frames = 2  # Anti-double-tap protection
```
- **min_hold**: Minimum frames before releasing a note
- **line_blank_threshold**: How many consecutive blank frames before ending a line hold
- **release_cooldown_frames**: Prevents accidental double-taps

**BPM Detection:**
```python
self.timeout_multiplier = 1.15  # BPM-based timeout multiplier
self.min_note_timeout = 7  # Minimum frames before timeout
```
- **timeout_multiplier**: How much longer than average to hold notes
- **min_note_timeout**: Minimum frames before forcing note release

**Consensus Mechanism:**
```python
self.consensus_frames = 2  # Frames needed for stable detection
```
- **consensus_frames**: How many consecutive frames must agree for detection

### Tuning Process

1. **Start with Default Settings**: Use the default values provided
2. **Test on Easy Songs**: Begin with slower, simpler songs
3. **Monitor Performance**: Watch the console output for detection feedback
4. **Adjust Video Latency**: Fine-tune in Fortnite Festival settings first
5. **Fine-tune Script Variables**: Adjust Python script variables for precision
6. **Test on Harder Songs**: Gradually test on more complex songs
7. **Iterate**: Continue adjusting until optimal performance

### Performance Optimization Tips

- **Close Background Applications**: Reduce system load
- **Use Windowed Mode**: Ensure consistent frame capture
- **Monitor GPU Usage**: Ensure stable 60fps capture
- **Check USB Latency**: Use high-quality USB cables
- **Regular Calibration**: Recalibrate after system updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational purposes. Use responsibly and in accordance with game terms of service.

## 🆘 Troubleshooting

### Common Issues

1. **Detection not working:**
   - Check screen resolution and game window position
   - Verify ROI settings in the code
   - Go to Settings > Game > Festival and try reducing video latency

2. **Controller not responding:**
   - Verify Titan Two connection
   - Check GPC script upload
   - Test button mapping

3. **Poor performance:**
   - Reduce screen capture fps
   - Close unnecessary applications
   - Check GPU drivers

### Getting Help

- Review the setup guides
- Open an issue for bugs or questions

---

**Disclaimer**: This tool is for educational and research purposes. Users are responsible for complying with game terms of service and local regulations. 