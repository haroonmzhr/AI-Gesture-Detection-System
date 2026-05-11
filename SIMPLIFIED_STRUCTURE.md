# Simplified Gesture Detection System

## Project Structure

### Before (12 Python Files)
- main.py (launcher)
- config.py (settings)
- utils.py (utilities)
- gesture_model.py (neural network)
- data_collector.py
- enhanced_data_collector.py
- train_model.py
- improved_train_model.py
- advanced_train_model.py
- gesture_controller.py
- improved_gesture_controller.py
- diagnostic.py

### After (3 Python Files)
```
project/
├── main.py              # Launcher and menu system
├── gesture.py           # ALL functionality consolidated
├── config.py            # Configuration (unchanged)
├── requirements.txt
├── README.md
└── USAGE_GUIDE.md
```

## What's Inside gesture.py?

All functionality is now consolidated into a single well-organized module with these classes and functions:

### 1. HandLandmarkProcessor
- `extract_landmarks()` - Extract hand landmarks from video
- `draw_landmarks()` - Visualize landmarks on frame
- `normalize_landmarks()` - Normalize for network input

### 2. GestureNet (Neural Network)
- 4-layer fully connected network
- Batch normalization and dropout
- 63 inputs → 128 hidden → 7 gesture classes

### 3. GestureDataset
- PyTorch dataset for gesture data loading
- Automatic scaling and normalization

### 4. DataCollector
- Interactive gesture data collection
- Keyboard controls for gesture selection
- Real-time sample counting
- Automatic data saving

### 5. ModelTrainer
- Data loading and preprocessing
- Training with validation
- Early stopping
- Model and scaler saving

### 6. GestureController
- Loads trained model for inference
- Real-time gesture prediction
- Mouse/keyboard automation
- Gesture action mapping

### 7. Utilities
- `smooth_predictions()` - Gesture smoothing
- `run_diagnostics()` - Data analysis

## How to Use

```bash
# Run the main menu
python main.py

# Choose options:
# 1. Collect gesture data
# 2. Train model
# 3. Run gesture controller
# 4. Run diagnostics
# 5. View project info
# 6. Exit
```

## Advantages of Simplified Structure

✅ **Single source of truth** - All logic in one place
✅ **Easier to understand** - No file jumping between modules
✅ **Faster imports** - Only 2 files to import from
✅ **Easier debugging** - Clear module organization within gesture.py
✅ **Better maintainability** - Changes don't require coordination across files
✅ **Smaller codebase** - No code duplication
✅ **Faster development** - Find and edit code quickly

## File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| config.py | 50 | Configuration constants |
| gesture.py | 650+ | All functionality |
| main.py | 150+ | Menu and launcher |

**Total: ~850 lines of clean, organized code**

## Training Data Flow

```
Raw Video (Webcam)
    ↓
HandLandmarkProcessor.extract_landmarks()
    ↓
Normalized landmarks (21 x 3 coordinates)
    ↓
GestureDataset
    ↓
ModelTrainer
    ↓
Trained GestureNet
    ↓
Saved model.pth + scaler.pkl
```

## Inference Flow

```
Live Video (Webcam)
    ↓
HandLandmarkProcessor.extract_landmarks()
    ↓
GestureController.predict_gesture()
    ↓
GestureNet.predict()
    ↓
Get gesture + confidence
    ↓
Execute action (mouse/keyboard)
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- scikit-learn
- numpy
- matplotlib
- pyautogui
- pynput

Install with: `pip install -r requirements.txt`

## Key Features

1. **Data Collection** - 500 samples per gesture with real-time visualization
2. **Training** - 70/15/15 train/val/test split with early stopping
3. **Inference** - Real-time gesture detection with smoothing
4. **Control** - Click, scroll, move cursor, double-click
5. **Diagnostics** - Analyze data distribution and class balance

## Notes

- All configuration is in config.py
- Training saves model to models/gesture_model.pth
- Feature scaler saved to models/scaler.pkl
- Training data saved to data/gesture_data.npz
- No external file dependencies - everything self-contained
