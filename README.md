# AI Gesture Detection System

This project implements a real-time gesture recognition system using PyTorch, MediaPipe, and OpenCV that can control system functions through hand gestures.

## Features

- Real-time hand gesture detection using MediaPipe
- Custom PyTorch neural network for gesture classification
- System control through gestures (mouse control, keyboard shortcuts, etc.)
- Data collection tool for training custom gestures
- Pre-trained model with common gestures

## Supported Gestures

1. **Fist** - Left click
2. **Open Palm** - Stop/Reset cursor
3. **Peace Sign** - Right click
4. **Thumbs Up** - Scroll up
5. **Thumbs Down** - Scroll down
6. **Pointing** - Move cursor
7. **OK Sign** - Double click

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Training Data

```bash
python data_collector.py
```

### 2. Train the Model

```bash
python train_model.py
```

### 3. Run Real-time Detection

```bash
python gesture_controller.py
```

## Project Structure

- `data_collector.py` - Collect and label gesture data
- `train_model.py` - Train the gesture recognition model
- `gesture_controller.py` - Real-time gesture detection and system control
- `gesture_model.py` - PyTorch model architecture
- `utils.py` - Utility functions for hand landmark processing
- `config.py` - Configuration settings
- `data/` - Training data directory
- `models/` - Saved model checkpoints

## Configuration

Edit `config.py` to customize:

- Gesture mappings
- Model parameters
- Camera settings
- System control sensitivity
