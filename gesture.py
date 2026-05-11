"""
Consolidated gesture detection system - All in one module
Includes: data collection, training, inference, and control
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import pickle
from typing import Tuple, Optional, List, Dict
import mediapipe as mp
import pyautogui
import time
from collections import deque
from config import (
    MODEL_CONFIG, GESTURES, GESTURE_ACTIONS, CAMERA_CONFIG,
    DETECTION_CONFIG, DATA_CONFIG
)

# ============================================================================
# HAND LANDMARK PROCESSING
# ============================================================================

class HandLandmarkProcessor:
    """Processes hand landmarks from MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def draw_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to wrist"""
        landmarks = landmarks.reshape(21, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        ref_distance = np.linalg.norm(landmarks[9] - wrist)
        if ref_distance > 0:
            normalized = normalized / ref_distance
        return normalized.flatten()

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class GestureNet(nn.Module):
    """Neural network for gesture classification"""
    
    def __init__(self, input_size: int = 63, hidden_size: int = 128, 
                 num_classes: int = 7, dropout_rate: float = 0.3):
        super(GestureNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.batch_norm_input = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if x.size(0) > 1:
            x = self.batch_norm_input(x)
        
        x = F.relu(self.fc1(x))
        if x.size(0) > 1:
            x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        if x.size(0) > 1:
            x = self.batch_norm3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Make prediction with confidence"""
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()

# ============================================================================
# DATA HANDLING
# ============================================================================

class GestureDataset(Dataset):
    """PyTorch dataset for gesture data"""
    
    def __init__(self, X, y, scaler=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.scaler = scaler
        
        if self.scaler is not None:
            self.X = torch.FloatTensor(self.scaler.transform(X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataCollector:
    """Collects gesture training data"""
    
    def __init__(self, samples_per_gesture: int = 500):
        self.hand_processor = HandLandmarkProcessor()
        self.gesture_data = {gesture: [] for gesture in GESTURES.values()}
        self.samples_per_gesture = samples_per_gesture
        self.current_gesture = None
        self.collecting = False
        self.countdown = 0
        self.sample_count = 0
    
    def collect(self):
        """Main data collection loop"""
        print("\n" + "="*50)
        print("GESTURE DATA COLLECTION")
        print("="*50)
        print("\nInstructions:")
        print("- Press number keys (0-6) to select gesture:")
        for idx, gesture in GESTURES.items():
            print(f"  {idx}: {gesture}")
        print("- Press SPACE to start/stop collecting")
        print("- Press 'q' to quit and save")
        print("- Press 'r' to reset current gesture")
        print("- Press 'c' to clear all data\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame = self.hand_processor.draw_landmarks(frame)
                landmarks = self.hand_processor.extract_landmarks(frame)
                
                # Display info
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Samples: {self.sample_count}/{len(GESTURES)*self.samples_per_gesture}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.current_gesture:
                    color = (0, 255, 0) if not self.collecting else (0, 0, 255)
                    cv2.putText(frame, f"Gesture: {self.current_gesture} - {'COLLECTING' if self.collecting else 'READY'}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if self.collecting and landmarks is not None:
                        self.gesture_data[self.current_gesture].append(landmarks)
                        self.sample_count += 1
                
                cv2.imshow("Data Collection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if self.current_gesture:
                        self.collecting = not self.collecting
                elif ord('0') <= key <= ord('6'):
                    gesture_idx = int(chr(key))
                    if gesture_idx in GESTURES:
                        self.current_gesture = GESTURES[gesture_idx]
                        self.collecting = False
                elif key == ord('r'):
                    if self.current_gesture:
                        self.gesture_data[self.current_gesture] = []
                        self.sample_count = max(0, self.sample_count - len(self.gesture_data[self.current_gesture]))
                elif key == ord('c'):
                    self.gesture_data = {gesture: [] for gesture in GESTURES.values()}
                    self.sample_count = 0
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_data()
    
    def save_data(self):
        """Save collected data to file"""
        os.makedirs('data', exist_ok=True)
        
        X = []
        y = []
        
        for class_idx, gesture in GESTURES.items():
            samples = self.gesture_data[gesture]
            X.extend(samples)
            y.extend([class_idx] * len(samples))
        
        if len(X) == 0:
            print("No data collected!")
            return
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        np.savez(DATA_CONFIG['data_file'], X=X, y=y)
        print(f"\nData saved: {len(X)} samples")
        for class_idx in range(len(GESTURES)):
            count = np.sum(y == class_idx)
            print(f"  {GESTURES[class_idx]}: {count}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """Handles model training"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.scaler = StandardScaler()
        self.model = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data"""
        if not os.path.exists(DATA_CONFIG['data_file']):
            raise FileNotFoundError(f"Data file not found: {DATA_CONFIG['data_file']}")
        
        data = np.load(DATA_CONFIG['data_file'])
        X = data['X']
        y = data['y']
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print("Data distribution:")
        for class_idx in range(len(GESTURES)):
            count = np.sum(y == class_idx)
            print(f"  {GESTURES[class_idx]}: {count}")
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train(self, num_epochs: int = 100, batch_size: int = 32):
        """Train the model"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Load data
        X, y = self.load_data()
        
        # Create dataset and loaders
        dataset = GestureDataset(X, y, self.scaler)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        
        # Create model
        self.model = GestureNet(
            input_size=MODEL_CONFIG['input_size'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_classes=MODEL_CONFIG['num_classes'],
            dropout_rate=MODEL_CONFIG['dropout_rate']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=MODEL_CONFIG['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Test
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        self.model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_acc = 100.0 * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%\n")
        
        print(classification_report(all_targets, all_preds, 
                                   target_names=[GESTURES[i] for i in range(len(GESTURES))]))
    
    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_classes': self.model.num_classes
            }
        }, MODEL_CONFIG['model_path'])
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Model saved!")

# ============================================================================
# GESTURE CONTROLLER
# ============================================================================

class GestureController:
    """Controls computer using gestures"""
    
    def __init__(self):
        self.hand_processor = HandLandmarkProcessor()
        self.model = self.load_model()
        self.scaler = self.load_scaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gesture_buffer = deque(maxlen=5)
        self.mouse_pos = pyautogui.position()
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.1
    
    def load_model(self) -> GestureNet:
        """Load trained model"""
        checkpoint = torch.load(MODEL_CONFIG['model_path'], map_location='cpu')
        config = checkpoint['config']
        
        model = GestureNet(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_classes=config['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def load_scaler(self):
        """Load feature scaler"""
        with open('models/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    
    def predict_gesture(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Predict gesture from landmarks"""
        landmarks = self.hand_processor.normalize_landmarks(landmarks)
        landmarks = self.scaler.transform([landmarks])[0]
        
        x = torch.FloatTensor([landmarks]).to(self.device)
        class_idx, confidence = self.model.predict(x)
        
        return GESTURES[class_idx], confidence
    
    def execute_action(self, gesture: str):
        """Execute action for gesture"""
        action = GESTURE_ACTIONS.get(gesture)
        if not action:
            return
        
        if action == 'left_click':
            pyautogui.click()
        elif action == 'right_click':
            pyautogui.click(button='right')
        elif action == 'double_click':
            pyautogui.click(clicks=2)
        elif action == 'scroll_up':
            pyautogui.scroll(DETECTION_CONFIG['scroll_amount'])
        elif action == 'scroll_down':
            pyautogui.scroll(-DETECTION_CONFIG['scroll_amount'])
        elif action == 'move_cursor':
            pass  # Handled separately
        elif action == 'stop_cursor':
            pass  # Handled separately
    
    def run(self):
        """Main gesture control loop"""
        print("Gesture Controller Starting...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame = self.hand_processor.draw_landmarks(frame)
                
                landmarks = self.hand_processor.extract_landmarks(frame)
                
                if landmarks is not None:
                    gesture, confidence = self.predict_gesture(landmarks)
                    
                    if confidence > DETECTION_CONFIG['confidence_threshold']:
                        self.gesture_buffer.append((gesture, confidence))
                        
                        if len(self.gesture_buffer) > 2:
                            # Majority voting
                            gestures = [g for g, _ in self.gesture_buffer]
                            most_common = max(set(gestures), key=gestures.count)
                            
                            now = time.time()
                            if now - self.last_gesture_time > self.gesture_cooldown:
                                self.execute_action(most_common)
                                self.last_gesture_time = now
                    
                    cv2.putText(frame, f"{gesture}: {confidence:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Gesture Controller", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics():
    """Run diagnostic analysis"""
    print("\n" + "="*50)
    print("DIAGNOSTIC ANALYSIS")
    print("="*50)
    
    if not os.path.exists(DATA_CONFIG['data_file']):
        print("No training data found!")
        return
    
    data = np.load(DATA_CONFIG['data_file'])
    X = data['X']
    y = data['y']
    
    print(f"\nDataset Size: {len(X)} samples")
    print("\nGesture Distribution:")
    for class_idx in range(len(GESTURES)):
        count = np.sum(y == class_idx)
        percentage = 100.0 * count / len(X)
        print(f"  {GESTURES[class_idx]}: {count} ({percentage:.1f}%)")
    
    print("\nFeature Statistics:")
    print(f"  Min: {X.min():.4f}")
    print(f"  Max: {X.max():.4f}")
    print(f"  Mean: {X.mean():.4f}")
    print(f"  Std: {X.std():.4f}")
    
    # Class imbalance check
    counts = [np.sum(y == i) for i in range(len(GESTURES))]
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2:
        print("  WARNING: High class imbalance detected!")

# ============================================================================
# UTILITIES
# ============================================================================

def smooth_predictions(predictions: List[int], window_size: int = 5) -> int:
    """Smooth predictions using sliding window"""
    if len(predictions) == 0:
        return None
    
    if len(predictions) < window_size:
        window_size = len(predictions)
    
    recent = predictions[-window_size:]
    return max(set(recent), key=recent.count)
