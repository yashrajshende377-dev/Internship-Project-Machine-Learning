import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def add_bias_term(self, X):
        """Add a column of ones for bias term"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Add bias term
        X_bias = self.add_bias_term(X)
        
        # Initialize parameters
        n_samples, n_features = X_bias.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self._predict_with_bias(X_bias)
            
            # Compute gradients
            dw = (1/n_samples) * X_bias.T.dot(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            
            # Compute and store loss
            loss = self._compute_mse(y, y_pred)
            self.loss_history.append(loss)
            
            # Early stopping
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                print(f"Converged at iteration {i}")
                break
    
    def _predict_with_bias(self, X_bias):
        """Predict using bias term"""
        return X_bias.dot(self.weights)
    
    def predict(self, X):
        """Make predictions on new data"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_bias = self.add_bias_term(X)
        return self._predict_with_bias(X_bias)
    
    def _compute_mse(self, y_true, y_pred):
        """Compute mean squared error"""
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_loss_curve(self):
        """Plot the loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.show()
    
    def get_coefficients(self, feature_names=None):
        """Get model coefficients"""
        coef_dict = {
            'intercept': self.weights[0]
        }
        if feature_names is not None:
            for i, name in enumerate(feature_names):
                coef_dict[name] = self.weights[i+1]
        else:
            for i in range(1, len(self.weights)):
                coef_dict[f'feature_{i-1}'] = self.weights[i]
        return coef_dict

# Example usage for scratch implementation
def main_scratch():
    # Generate data
    np.random.seed(42)
    n_samples = 500
    
    square_footage = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    
    # True coefficients
    true_coeffs = [100, 30000, 20000]  # intercept, sqft, bedrooms, bathrooms
    prices = (true_coeffs[0] * square_footage + 
              true_coeffs[1] * bedrooms + 
              true_coeffs[2] * bathrooms + 
              np.random.normal(0, 20000, n_samples))
    prices = np.maximum(prices, 50000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': prices
    })
    
    # Prepare data
    X = data[['square_footage', 'bedrooms', 'bathrooms']]
    y = data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features (important for gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training linear regression from scratch...")
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=2000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"MSE: ${mse:,.2f}")
    print(f"RMSE: ${np.sqrt(mse):,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Get coefficients
    coefficients = model.get_coefficients(['sqft', 'bedrooms', 'bathrooms'])
    print("\nModel Coefficients:")
    for name, coef in coefficients.items():
        print(f"{name}: {coef:.2f}")
    
    # Plot loss curve
    model.plot_loss_curve()

if __name__ == "__main__":
    print("=" * 50)
    print("Scikit-learn Implementation")
    print("=" * 50)
    main()
    
    print("\n" + "=" * 50)
    print("From Scratch Implementation (Gradient Descent)")
    print("=" * 50)
    main_scratch()