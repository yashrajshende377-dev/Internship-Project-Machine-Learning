import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class CatDogSVMClassifier:
    """
    Support Vector Machine classifier for cat vs dog image classification
    """
    
    def __init__(self, img_size=64, use_hog=True, use_pca=True, random_state=42):
        """
        Initialize the classifier
        
        Args:
            img_size: Size to resize images to (img_size x img_size)
            use_hog: Whether to use HOG features
            use_pca: Whether to use PCA for dimensionality reduction
            random_state: Random seed for reproducibility
        """
        self.img_size = img_size
        self.use_hog = use_hog
        self.use_pca = use_pca
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, random_state=random_state)  # Keep 95% variance
        self.svm = None
        self.is_fitted = False
        
    def load_data(self, data_dir, sample_size=None):
        """
        Load and preprocess images from directory
        
        Args:
            data_dir: Path to directory containing 'cat' and 'dog' subdirectories
            sample_size: Number of images to load per class (None for all)
        
        Returns:
            X: Feature matrix
            y: Labels (0=cat, 1=dog)
        """
        images = []
        labels = []
        
        # Define class mappings
        classes = {'cat': 0, 'dog': 1}
        
        for class_name, label in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            # Get image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample if specified
            if sample_size and len(image_files) > sample_size:
                image_files = np.random.choice(image_files, sample_size, replace=False)
            
            print(f"Loading {len(image_files)} {class_name} images...")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize
                resized = cv2.resize(gray, (self.img_size, self.img_size))
                
                images.append(resized)
                labels.append(label)
        
        return np.array(images), np.array(labels)
    
    def extract_features(self, images):
        """
        Extract features from images
        
        Args:
            images: Array of grayscale images
            
        Returns:
            features: Extracted feature vectors
        """
        features = []
        
        for img in images:
            if self.use_hog:
                # Extract HOG features
                hog_features = hog(img, 
                                  orientations=9,
                                  pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2),
                                  visualize=False,
                                  feature_vector=True)
                features.append(hog_features)
            else:
                # Flatten image pixels
                features.append(img.flatten())
        
        return np.array(features)
    
    def prepare_data(self, images, labels, test_size=0.2, validation_size=0.1):
        """
        Prepare data for training by extracting features and splitting
        
        Args:
            images: Array of images
            labels: Array of labels
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Extract features
        print("Extracting features...")
        X = self.extract_features(images)
        y = np.array(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val, y_val, kernel='rbf', C=10, gamma='scale'):
        """
        Train the SVM model with optional hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            kernel: SVM kernel ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            
        Returns:
            Best model and parameters
        """
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply PCA if enabled
        if self.use_pca:
            print(f"Applying PCA (keeping 95% variance)...")
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_val_scaled = self.pca.transform(X_val_scaled)
            print(f"Reduced to {X_train_scaled.shape[1]} components")
        
        # Grid search for hyperparameter tuning
        print("Performing grid search for optimal parameters...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.001],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # For large datasets, use a subset for grid search
        if len(X_train_scaled) > 10000:
            print("Large dataset detected. Using subset for grid search...")
            idx = np.random.choice(len(X_train_scaled), 5000, replace=False)
            X_subset = X_train_scaled[idx]
            y_subset = y_train[idx]
            cv = 3
        else:
            X_subset = X_train_scaled
            y_subset = y_train
            cv = 5
        
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='accuracy',
            verbose=1
        )
        
        grid_search.fit(X_subset, y_subset)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train final model on full training data
        print("Training final model...")
        self.svm = SVC(
            **grid_search.best_params_,
            random_state=self.random_state,
            probability=True
        )
        self.svm.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_accuracy = self.svm.score(X_val_scaled, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        self.is_fitted = True
        
        return self.svm, grid_search.best_params_
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale and transform
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.svm.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale and transform
        X_test_scaled = self.scaler.transform(X_test)
        if self.use_pca:
            X_test_scaled = self.pca.transform(X_test_scaled)
        
        # Predict
        y_pred = self.svm.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                      target_names=['Cat', 'Dog'])
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        return metrics, y_pred
    
    def plot_results(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cat', 'Dog'],
                   yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def predict_single_image(self, image_path):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction (0=cat, 1=dog) and confidence
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.img_size, self.img_size))
        
        # Extract features
        features = self.extract_features([resized])
        
        # Scale and transform
        features_scaled = self.scaler.transform(features)
        if self.use_pca:
            features_scaled = self.pca.transform(features_scaled)
        
        # Predict
        pred = self.svm.predict(features_scaled)[0]
        proba = self.svm.predict_proba(features_scaled)[0]
        
        class_names = ['Cat', 'Dog']
        confidence = proba[pred]
        
        return pred, class_names[pred], confidence
    
    def save_model(self, filepath='cat_dog_svm_model.pkl'):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'pca': self.pca if self.use_pca else None,
            'use_pca': self.use_pca,
            'img_size': self.img_size,
            'use_hog': self.use_hog
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='cat_dog_svm_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.use_pca = model_data['use_pca']
        self.img_size = model_data['img_size']
        self.use_hog = model_data['use_hog']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


# Example usage and demonstration
def main():
    # Initialize classifier
    classifier = CatDogSVMClassifier(
        img_size=64,      # Resize images to 64x64
        use_hog=True,     # Use HOG features for better performance [citation:9]
        use_pca=True,     # PCA reduces dimensionality [citation:3][citation:8]
        random_state=42
    )
    
    # Define dataset paths
    # You need to download the dataset from Kaggle:
    # https://www.kaggle.com/c/dogs-vs-cats/data
    data_dir = "path/to/dataset/train"  # Update this path
    
    print("=" * 60)
    print("CAT vs DOG CLASSIFICATION USING SVM")
    print("=" * 60)
    
    # Load data (use sample_size to limit for faster testing)
    print("\n1. Loading data...")
    images, labels = classifier.load_data(data_dir, sample_size=500)  # Use 500 per class for demo
    print(f"Loaded {len(images)} images")
    print(f"Class distribution: Cats = {np.sum(labels==0)}, Dogs = {np.sum(labels==1)}")
    
    # Prepare data
    print("\n2. Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(
        images, labels, test_size=0.2, validation_size=0.1
    )
    
    # Train model
    print("\n3. Training SVM model...")
    model, best_params = classifier.train(
        X_train, y_train, X_val, y_val,
        kernel='rbf', C=10, gamma='scale'
    )
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    metrics, y_pred = classifier.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    classifier.plot_results(y_test, y_pred)
    
    # Save model
    print("\n5. Saving model...")
    classifier.save_model('cat_dog_svm_model.pkl')
    
    # Example: Predict a single image
    print("\n6. Example prediction on a single image:")
    # Uncomment and update path to test
    # image_path = "path/to/test_image.jpg"
    # pred, class_name, confidence = classifier.predict_single_image(image_path)
    # print(f"Prediction: {class_name} (confidence: {confidence:.2f})")
    
    # Demonstrate loading model
    print("\n7. Loading saved model...")
    new_classifier = CatDogSVMClassifier()
    new_classifier.load_model('cat_dog_svm_model.pkl')
    print("Model loaded successfully!")

if __name__ == "__main__":
    main()