AI Gesture Detection System
A real-time, end-to-end gesture recognition and system control interface built with PyTorch, MediaPipe, and OpenCV. This project enables users to control their computer (mouse movements, clicking, and scrolling) using hand gestures captured via a standard webcam.

The system features a consolidated pipeline that includes a custom data collection tool, a neural network training module, and a real-time inference engine with smooth cursor mapping.

Key Features
Real-Time Hand Tracking: Leverages MediaPipe for high-fidelity 21-point hand landmark extraction with low latency.

Custom Neural Network: Implements GestureNet, a 4-layer fully connected architecture with Batch Normalization and Dropout for robust gesture classification.

System Automation: Integrates with PyAutoGUI and pynput to map detected gestures to system-level actions (Left/Right Click, Double Click, Scrolling, and Cursor Movement).

End-to-End Workflow: Includes built-in tools to:

Test hand detection accuracy.

Collect personalized training data for custom hand shapes.

Train a deep learning model locally in minutes.

Execute real-time control.

Intelligent Smoothing: Uses a sliding window prediction algorithm and exponential smoothing to ensure stable cursor movement and prevent accidental triggers.

Supported Gestures & Mappings
Gesture	System Action
Pointing (Index)	Move Cursor
Fist	Left Click
Peace Sign	Right Click
OK Sign	Double Click
Open Palm	Stop/Reset Cursor
Thumbs Up	Scroll Up
Thumbs Down	Scroll Down
Project Architecture
The project has been optimized into a streamlined 3-module structure:

main.py: The central launcher and interactive menu.

gesture.py: The core engine containing the Landmark Processor, Dataset handler, Neural Network architecture, and Controller.

config.py: Centralized configuration for model hyperparameters, camera settings, and gesture sensitivity.

Tech Stack
Language: Python 3.8+

Deep Learning: PyTorch

Computer Vision: MediaPipe, OpenCV

Data Science: NumPy, Scikit-learn, Matplotlib

Automation: PyAutoGUI, Pynput

🏁 Quick Start
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the System:

Bash
python main.py
Follow the Workflow:

Test your camera using Option 1.

(Optional) Collect fresh data and retrain using Options 2 & 3.

Start controlling your PC using Option 4.

🛡️ Safety Features
PyAutoGUI Failsafe: Move the mouse to any corner of the screen to instantly abort the program.

Emergency Stop: Press q in the video window to stop detection.

Confidence Thresholding: Only executes actions when the model's prediction confidence exceeds a configurable limit (default 90%+).
