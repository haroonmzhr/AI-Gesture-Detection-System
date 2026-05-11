# Gesture Detection System - Usage Guide

## Quick Start

1. **Run the main launcher:**

   ```bash
   python main.py
   ```

2. **Follow the workflow:**
   - Test hand detection (Option 1)
   - Collect gesture data (Option 2)
   - Train the model (Option 3)
   - Use gesture control (Option 4)

## Detailed Instructions

### Step 1: Test Hand Detection

- Run option 1 to verify your camera and MediaPipe are working
- You should see hand landmarks drawn on your hand when detected
- Press 'q' to quit

### Step 2: Collect Training Data

- Run option 2 to start data collection
- For each gesture (0-6):
  - Press the number key to select the gesture
  - Position your hand and press SPACE
  - Wait for countdown, then perform the gesture
  - Hold steady until 100 samples are collected
- Collect data for all 7 gestures:
  - 0: Fist
  - 1: Open Palm
  - 2: Peace Sign (V sign)
  - 3: Thumbs Up
  - 4: Thumbs Down
  - 5: Pointing (index finger extended)
  - 6: OK Sign (thumb and index finger circle)

### Step 3: Train the Model

- Run option 3 after collecting data
- Training takes 5-15 minutes depending on your computer
- Model will be saved automatically when training completes

### Step 4: Use Gesture Control

- Run option 4 to start real-time gesture control
- **WARNING: This will control your mouse and keyboard!**
- Gesture actions:
  - **Fist**: Left mouse click
  - **Open Palm**: Stop cursor movement
  - **Peace Sign**: Right mouse click
  - **Thumbs Up**: Scroll up
  - **Thumbs Down**: Scroll down
  - **Pointing**: Move cursor (follows hand movement)
  - **OK Sign**: Double click

## Tips for Best Results

### Data Collection:

- Use good lighting (avoid shadows on hands)
- Clear background behind your hand
- Hold gestures steady and clearly
- Vary hand positions slightly between samples
- Ensure your hand fills a good portion of the camera view

### During Use:

- Maintain good lighting
- Keep hand clearly visible to camera
- Make gestures deliberately and hold briefly
- Position camera at comfortable viewing angle
- Start with simple gestures (fist, open palm) before complex ones

### Troubleshooting:

- If gestures aren't recognized: Retrain with more varied data
- If cursor is jumpy: Adjust smoothing settings in `config.py`
- If actions trigger too often: Increase confidence threshold in `config.py`
- Camera not working: Check that no other applications are using it

## Configuration

Edit `config.py` to customize:

- `DETECTION_CONFIG['confidence_threshold']`: Minimum confidence for actions (0.0-1.0)
- `DETECTION_CONFIG['cursor_speed']`: How fast cursor moves (1.0-5.0)
- `DETECTION_CONFIG['smoothing_factor']`: Cursor smoothing (0.0-1.0)
- `MODEL_CONFIG['num_epochs']`: Training duration (50-200)

## Safety Notes

**Important**: The gesture controller will move your mouse cursor and perform clicks!

- Always have a way to quickly stop the program (press 'q' in video window)
- Don't use near important documents or sensitive applications
- Test thoroughly before real use
- Keep PyAutoGUI failsafe enabled (move mouse to corner to stop)

## File Structure

```
Gesture Detection/
├── main.py                 # Main launcher
├── setup_verification.py   # Check if everything works
├── test_hand_detection.py  # Test basic hand detection
├── data_collector.py       # Collect training data
├── train_model.py          # Train the neural network
├── gesture_controller.py   # Real-time gesture control
├── gesture_model.py        # PyTorch model definition
├── utils.py                # Utility functions
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── data/                   # Training data storage
├── models/                 # Saved models
└── README.md              # Project documentation
```

Enjoy your AI gesture detection system!
