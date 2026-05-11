"""
Simplified AI Gesture Detection System - Main Launcher
"""

import os
import numpy as np
from gesture import (
    DataCollector, ModelTrainer, GestureController, 
    HandLandmarkProcessor, run_diagnostics
)
from config import GESTURES

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*60)
    print("      AI GESTURE DETECTION SYSTEM")
    print("="*60 + "\n")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'cv2', 'mediapipe', 'numpy', 'sklearn', 
        'matplotlib', 'pyautogui', 'pynput'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        return False
    
    # Check required files
    required_files = ['config.py', 'gesture.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Missing file: {file}")
            return False
    
    print("✓ All requirements satisfied!")
    return True

def main_menu():
    """Display main menu"""
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Collect gesture data")
        print("2. Train model")
        print("3. Run gesture controller")
        print("4. Run diagnostics")
        print("5. View project info")
        print("6. Exit")
        print("="*50)
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_controller()
        elif choice == '4':
            run_diagnostics()
        elif choice == '5':
            show_info()
        elif choice == '6':
            print("\nGoodbye!\n")
            break
        else:
            print("Invalid choice. Try again.")

def collect_data():
    """Collect training data"""
    print("\n" + "="*50)
    print("DATA COLLECTION")
    print("="*50)
    print("\nMake sure you have:")
    print("- Good lighting")
    print("- Clear background")
    print("- Working webcam\n")
    
    print("Supported Gestures:")
    for idx, gesture in GESTURES.items():
        print(f"  {idx}: {gesture}")
    
    samples = input("\nSamples per gesture (default 500): ").strip()
    samples = int(samples) if samples.isdigit() else 500
    
    collector = DataCollector(samples_per_gesture=samples)
    collector.collect()

def train_model():
    """Train the model"""
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    if not os.path.exists('data/gesture_data.npz'):
        print("\n✗ No training data found!")
        print("Run data collection first (option 1)")
        return
    
    epochs = input("\nNumber of epochs (default 100): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 100
    
    batch_size = input("Batch size (default 32): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 32
    
    trainer = ModelTrainer()
    trainer.train(num_epochs=epochs, batch_size=batch_size)

def run_controller():
    """Run gesture controller"""
    print("\n" + "="*50)
    print("GESTURE CONTROLLER")
    print("="*50)
    
    if not os.path.exists('models/gesture_model.pth'):
        print("\n✗ No trained model found!")
        print("Train the model first (option 2)")
        return
    
    print("\n⚠ WARNING:")
    print("- System will control your mouse and keyboard")
    print("- Press 'q' in video window to stop")
    print("- Make sure webcam is working\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm == 'y':
        controller = GestureController()
        controller.run()
    else:
        print("Cancelled")

def show_info():
    """Display project information"""
    print("\n" + "="*50)
    print("PROJECT INFORMATION")
    print("="*50)
    
    # Data status
    if os.path.exists('data/gesture_data.npz'):
        data = np.load('data/gesture_data.npz')
        print(f"\n✓ Training Data: {len(data['X'])} samples")
        for idx in range(len(GESTURES)):
            count = np.sum(data['y'] == idx)
            print(f"  - {GESTURES[idx]}: {count}")
    else:
        print("\n✗ No training data")
    
    # Model status
    if os.path.exists('models/gesture_model.pth'):
        print("✓ Trained model available")
    else:
        print("✗ No trained model")
    
    if os.path.exists('models/scaler.pkl'):
        print("✓ Feature scaler available")
    else:
        print("✗ No feature scaler")
    
    print("\nSupported Gestures:")
    actions = {
        'fist': 'Left click',
        'open_palm': 'Stop cursor',
        'peace': 'Right click',
        'thumbs_up': 'Scroll up',
        'thumbs_down': 'Scroll down',
        'pointing': 'Move cursor',
        'ok_sign': 'Double click'
    }
    
    for gesture, action in actions.items():
        print(f"  • {gesture.replace('_', ' ').title()}: {action}")
    
    print(f"\nDirectory: {os.getcwd()}")

def main():
    """Main entry point"""
    print_banner()
    
    if not check_requirements():
        print("\n✗ Requirements check failed!")
        print("Install missing packages with: pip install -r requirements.txt")
        return
    
    main_menu()

if __name__ == "__main__":
    main()

