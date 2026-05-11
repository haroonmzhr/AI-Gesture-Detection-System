"""
Configuration file for the gesture detection system
"""

# Model configuration
MODEL_CONFIG = {
    'input_size': 63,  # 21 landmarks * 3 coordinates (x, y, z)
    'hidden_size': 128,
    'num_classes': 7,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100,
    'model_path': 'models/gesture_model.pth'
}

# Gesture classes
GESTURES = {
    0: 'fist',
    1: 'open_palm',
    2: 'peace',
    3: 'thumbs_up',
    4: 'thumbs_down',
    5: 'pointing',
    6: 'ok_sign'
}

# System control mappings
GESTURE_ACTIONS = {
    'fist': 'left_click',
    'open_palm': 'stop_cursor',
    'peace': 'right_click',
    'thumbs_up': 'scroll_up',
    'thumbs_down': 'scroll_down',
    'pointing': 'move_cursor',
    'ok_sign': 'double_click'
}

# Camera settings
CAMERA_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30
}

# Detection settings
DETECTION_CONFIG = {
    'confidence_threshold': 0.5,  # Lowered from 0.7
    'smoothing_factor': 0.7,      # Reduced for more responsiveness  
    'cursor_speed': 2.0,          # Reduced for smoother movement
    'scroll_amount': 3
}

# Data collection settings
DATA_CONFIG = {
    'samples_per_gesture': 500,  # Increased from 100
    'data_file': 'data/gesture_data.npz'
}
