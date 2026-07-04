import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

class HandGestureRecognition:
    """
    Comprehensive hand gesture recognition system using MediaPipe and deep learning
    """
    
    def __init__(self, model_type='ensemble', use_mediapipe=True):
        """
        Initialize gesture recognition system
        
        Args:
            model_type: 'cnn', 'svm', 'rf', 'knn', or 'ensemble'
            use_mediapipe: Use MediaPipe for hand landmark detection
        """
        self.model_type = model_type
        self.use_mediapipe = use_mediapipe
        
        # Initialize MediaPipe
        if use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        
        # Initialize models
        self.cnn_model = None
        self.svm_model = None
        self.rf_model = None
        self.knn_model = None
        
        # Data processing
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
        # Gesture classes
        self.gesture_classes = [
            'thumbs_up', 'thumbs_down', 'peace', 'ok', 'fist',
            'open_palm', 'point', 'love', 'rock', 'spock',
            'five', 'two', 'one', 'three', 'four'
        ]
        self.is_trained = False
        
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from image using MediaPipe
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            landmarks: Normalized hand landmarks
            detected: Boolean indicating if hand was detected
        """
        if not self.use_mediapipe:
            return None, False
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None, False
        
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks), True
    
    def preprocess_image(self, image_path=None, image_array=None, target_size=(224, 224)):
        """
        Preprocess image for CNN model
        
        Args:
            image_path: Path to image file
            image_array: Numpy array of image
            target_size: Target size for CNN
            
        Returns:
            Preprocessed image
        """
        if image_path:
            image = cv2.imread(image_path)
        elif image_array is not None:
            image = image_array
        else:
            raise ValueError("Either image_path or image_array must be provided")
        
        # Resize and normalize
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_landmark_features(self, landmarks):
        """
        Create additional features from hand landmarks
        
        Args:
            landmarks: Raw landmarks from MediaPipe
            
        Returns:
            Feature vector with geometric relationships
        """
        if landmarks is None or len(landmarks) < 63:  # 21 landmarks * 3 coordinates
            return None
            
        # Reshape to (21, 3)
        landmarks = np.array(landmarks).reshape(-1, 3)
        
        # Calculate distances between landmarks
        features = []
        
        # Distance from wrist (landmark 0) to all fingers
        wrist = landmarks[0]
        for i in range(1, 21):
            dist = np.linalg.norm(landmarks[i] - wrist)
            features.append(dist)
        
        # Distance between fingertips
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                dist = np.linalg.norm(landmarks[finger_tips[i]] - landmarks[finger_tips[j]])
                features.append(dist)
        
        # Angles between fingers
        for i in range(1, 5):
            # MCP, PIP, TIP joints for each finger
            mcp = landmarks[i*4]
            pip = landmarks[i*4 + 1]
            tip = landmarks[i*4 + 2]
            
            # Compute angle at PIP joint
            v1 = mcp - pip
            v2 = tip - pip
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1))
            features.append(angle)
        
        return np.array(features)
    
    def create_dataset(self, data_dir, augment=False, save_features=True):
        """
        Create dataset from image directory
        
        Args:
            data_dir: Directory containing gesture folders
            augment: Apply data augmentation
            save_features: Save extracted features to disk
        """
        X_images = []
        X_landmarks = []
        y = []
        
        print("Creating dataset from:", data_dir)
        
        for gesture in self.gesture_classes:
            gesture_dir = os.path.join(data_dir, gesture)
            if not os.path.exists(gesture_dir):
                print(f"Warning: {gesture_dir} not found")
                continue
                
            print(f"Processing {gesture}...")
            image_files = [f for f in os.listdir(gesture_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(gesture_dir, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # Extract landmarks
                landmarks, detected = self.extract_landmarks(image)
                if not detected:
                    continue
                
                # Add features
                X_landmarks.append(landmarks)
                y.append(gesture)
                
                # For CNN, also store images
                if self.model_type == 'cnn' or self.model_type == 'ensemble':
                    processed_img = self.preprocess_image(image_array=image)
                    X_images.append(processed_img)
                    
                    # Data augmentation
                    if augment:
                        # Rotation
                        augmented = self.augment_image(image)
                        X_images.append(self.preprocess_image(image_array=augmented))
                        X_landmarks.append(landmarks)
                        y.append(gesture)
        
        X_landmarks = np.array(X_landmarks)
        y = np.array(y)
        
        print(f"Created dataset with {len(X_landmarks)} samples")
        
        if save_features:
            np.save('X_landmarks.npy', X_landmarks)
            np.save('y_labels.npy', y)
            if len(X_images) > 0:
                np.save('X_images.npy', np.array(X_images))
        
        if self.model_type == 'cnn' or self.model_type == 'ensemble':
            return X_images, X_landmarks, y
        else:
            return X_landmarks, y
    
    def augment_image(self, image):
        """Apply data augmentation"""
        h, w = image.shape[:2]
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            
        return image
    
    def build_cnn_model(self, input_shape=(224, 224, 3), num_classes=15):
        """
        Build CNN model using MobileNetV2 architecture
        
        Args:
            input_shape: Input image shape
            num_classes: Number of gesture classes
            
        Returns:
            Compiled CNN model
        """
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_ml_models(self):
        """Build traditional ML models"""
        self.svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        self.knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    
    def train_landmark_models(self, X_train, y_train, X_val, y_val):
        """Train traditional ML models on landmark features"""
        self.build_ml_models()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        
        print("Training SVM...")
        self.svm_model.fit(X_train_pca, y_train)
        svm_acc = self.svm_model.score(X_val_pca, y_val)
        print(f"SVM Validation Accuracy: {svm_acc:.4f}")
        
        print("Training Random Forest...")
        self.rf_model.fit(X_train_pca, y_train)
        rf_acc = self.rf_model.score(X_val_pca, y_val)
        print(f"RF Validation Accuracy: {rf_acc:.4f}")
        
        print("Training KNN...")
        self.knn_model.fit(X_train_pca, y_train)
        knn_acc = self.knn_model.score(X_val_pca, y_val)
        print(f"KNN Validation Accuracy: {knn_acc:.4f}")
        
        return svm_acc, rf_acc, knn_acc
    
    def train_ensemble(self, X_train_landmarks, y_train, X_val_landmarks, y_val,
                      X_train_images=None, y_train_images=None):
        """Train ensemble of all models"""
        print("Training ensemble model...")
        
        # Train landmark models
        self.train_landmark_models(X_train_landmarks, y_train, X_val_landmarks, y_val)
        
        # Train CNN if images available
        if X_train_images is not None and len(X_train_images) > 0:
            print("Training CNN...")
            self.cnn_model = self.build_cnn_model(
                input_shape=X_train_images[0].shape,
                num_classes=len(np.unique(y_train))
            )
            
            early_stopping = callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            )
            
            self.cnn_model.fit(
                np.array(X_train_images),
                y_train,
                validation_data=(np.array(X_val_landmarks), y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
        
        self.is_trained = True
    
    def predict_landmark_models(self, X_test):
        """
        Make predictions using ensemble of ML models
        
        Args:
            X_test: Test features
            
        Returns:
            Ensemble predictions
        """
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Get predictions from each model
        svm_pred = self.svm_model.predict(X_test_pca)
        rf_pred = self.rf_model.predict(X_test_pca)
        knn_pred = self.knn_model.predict(X_test_pca)
        
        # Ensemble voting
        predictions = []
        for i in range(len(X_test_pca)):
            votes = [svm_pred[i], rf_pred[i], knn_pred[i]]
            predictions.append(max(set(votes), key=votes.count))
        
        return np.array(predictions)
    
    def predict_ensemble(self, landmarks, image=None):
        """
        Make prediction using ensemble approach
        
        Args:
            landmarks: Hand landmarks
            image: Original image for CNN prediction
            
        Returns:
            Predicted gesture and confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare landmark features
        landmark_features = self.create_landmark_features(landmarks)
        if landmark_features is None:
            return None, 0
        
        # Scale and transform
        landmark_features = landmark_features.reshape(1, -1)
        landmark_features_scaled = self.scaler.transform(landmark_features)
        landmark_features_pca = self.pca.transform(landmark_features_scaled)
        
        # Get predictions
        svm_prob = self.svm_model.predict_proba(landmark_features_pca)[0]
        rf_prob = self.rf_model.predict_proba(landmark_features_pca)[0]
        knn_prob = self.knn_model.predict_proba(landmark_features_pca)[0]
        
        # Average probabilities
        avg_probs = (svm_prob + rf_prob + knn_prob) / 3
        pred_class = np.argmax(avg_probs)
        confidence = avg_probs[pred_class]
        
        # If CNN is available and image is provided
        if image is not None and self.cnn_model is not None:
            processed_img = self.preprocess_image(image_array=image)
            processed_img = processed_img.reshape(1, *processed_img.shape)
            cnn_pred = self.cnn_model.predict(processed_img)[0]
            
            # Combine predictions
            combined_probs = (avg_probs + cnn_pred) / 2
            pred_class = np.argmax(combined_probs)
            confidence = combined_probs[pred_class]
        
        gesture = self.label_encoder.inverse_transform([pred_class])[0]
        return gesture, confidence
    
    def evaluate_model(self, X_test, y_test, model_type='ensemble'):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_type: Model to evaluate
            
        Returns:
            Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        if model_type == 'svm':
            y_pred = self.svm_model.predict(X_test_pca)
        elif model_type == 'rf':
            y_pred = self.rf_model.predict(X_test_pca)
        elif model_type == 'knn':
            y_pred = self.knn_model.predict(X_test_pca)
        else:  # ensemble
            y_pred = self.predict_landmark_models(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.gesture_classes,
                   yticklabels=self.gesture_classes)
        plt.title('Confusion Matrix - Hand Gesture Recognition')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path='hand_gesture_model.h5'):
        """Save trained models"""
        model_data = {
            'svm_model': self.svm_model,
            'rf_model': self.rf_model,
            'knn_model': self.knn_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoder': self.label_encoder,
            'gesture_classes': self.gesture_classes,
            'model_type': self.model_type,
            'use_mediapipe': self.use_mediapipe,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.cnn_model is not None:
            self.cnn_model.save('cnn_model.h5')
        
        print(f"Models saved to {model_path}")
    
    def load_model(self, model_path='hand_gesture_model.h5'):
        """Load trained models"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm_model = model_data['svm_model']
        self.rf_model = model_data['rf_model']
        self.knn_model = model_data['knn_model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.label_encoder = model_data['label_encoder']
        self.gesture_classes = model_data['gesture_classes']
        self.model_type = model_data['model_type']
        self.use_mediapipe = model_data['use_mediapipe']
        self.is_trained = model_data['is_trained']
        
        if os.path.exists('cnn_model.h5'):
            self.cnn_model = tf.keras.models.load_model('cnn_model.h5')
        
        print(f"Models loaded from {model_path}")
    
    def real_time_recognition(self, camera_id=0):
        """
        Real-time hand gesture recognition from webcam
        
        Args:
            camera_id: Camera device ID
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        cap = cv2.VideoCapture(camera_id)
        print("Starting real-time gesture recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, detected = self.extract_landmarks(frame)
            
            if detected:
                # Make prediction
                gesture, confidence = self.predict_ensemble(landmarks, frame)
                
                # Display results
                if gesture and confidence > 0.5:
                    cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No gesture recognized", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Example usage of hand gesture recognition system"""
    
    # Initialize gesture recognition system
    gesture_system = HandGestureRecognition(model_type='ensemble', use_mediapipe=True)
    
    # Option 1: Train on your own dataset
    # data_dir = "path/to/gesture/dataset"
    # X_images, X_landmarks, y = gesture_system.create_dataset(data_dir, augment=True)
    
    # Option 2: Use pre-trained model
    # gesture_system.load_model('hand_gesture_model.h5')
    # gesture_system.real_time_recognition()
    
    # Create synthetic dataset for demonstration
    print("Creating synthetic dataset for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 63  # 21 landmarks * 3 coordinates
    
    # Generate synthetic landmark data with some patterns
    X = np.random.normal(0, 1, (n_samples, n_features))
    y = np.random.choice(gesture_system.gesture_classes[:5], n_samples)
    
    # Add some structure to make it realistic
    for i, gesture in enumerate(gesture_system.gesture_classes[:5]):
        mask = y == gesture
        X[mask, i*10:(i+1)*10] += 2.0
    
    # Encode labels
    y_encoded = gesture_system.label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train models
    print("\nTraining models...")
    gesture_system.train_landmark_models(X_train, y_train, X_val, y_val)
    gesture_system.is_trained = True
    
    # Evaluate models
    print("\nEvaluating models...")
    results = gesture_system.evaluate_model(X_test, y_test, model_type='ensemble')
    print(f"Ensemble Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    gesture_system.plot_confusion_matrix(y_test, results['predictions'])
    
    # Save models
    gesture_system.save_model('hand_gesture_model.pkl')
    
    # Test a single prediction
    print("\nTesting single prediction...")
    test_sample = X_test[0].reshape(1, -1)
    test_gesture = gesture_system.svm_model.predict(test_sample)[0]
    test_gesture_name = gesture_system.label_encoder.inverse_transform([test_gesture])[0]
    print(f"Sample prediction: {test_gesture_name}")
    
    # Real-time recognition (uncomment to run)
    # gesture_system.real_time_recognition()

if __name__ == "__main__":
    main()