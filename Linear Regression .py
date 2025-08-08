# California Housing Price Prediction


# Step 0: Import necessary libraries
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Mount Google Drive & Load Dataset
drive.mount('/content/drive')

# Load dataset from Google Drive (zipped CSV)
df = pd.read_csv('/content/drive/MyDrive/Machine Learning/housing.csv.zip')

# Step 2: Data Normalization (Z-score standardization)
# Select all numeric features
features = df.select_dtypes(include=np.number).columns.tolist()
# Standardize features: (value - mean) / std deviation
df[features] = (df[features] - df[features].mean()) / df[features].std()

# Step 3: Feature Matrix (X) & Target Vector (y)
# Drop target and categorical column from X
X = df.drop(['median_house_value', 'ocean_proximity'], axis=1).values.astype(np.float64)
# Target vector (reshape to column vector)
y = df['median_house_value'].values.reshape(-1, 1)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Initialize Model Parameters
w = np.random.randn(1, X_train.shape[1])  # Random initial weights
b = 0  # Bias term
learning_rate = 0.02  # Step size for gradient descent

# Step 6: Define Linear Regression Functions

# Forward Propagation: Predict values
def forward_prop(w, b, X):
    z = np.dot(w, X.T) + b
    return z.T  # Return column vector

# Cost Function: Mean Squared Error (MSE)
def cost(y_pred, y):
    m = y.shape[0]
    return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

# Backward Propagation: Calculate gradients for w and b
def back_prop(y_pred, y, X):
    m = y.shape[0]
    dz = y_pred - y
    dw = (1 / m) * np.dot(dz.T, X)
    db = (1 / m) * np.sum(dz)
    return dw, db

# Gradient Descent: Update weights and bias
def gradient_descent(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Training Function
def linear_model(X_train, y_train, epochs):
    global w, b
    losses = []
    for i in range(epochs):
        # Forward pass
        y_pred = forward_prop(w, b, X_train)
        # Compute cost
        c = cost(y_pred, y_train)
        # Backpropagation
        dw, db = back_prop(y_pred, y_train, X_train)
        # Update parameters
        w, b = gradient_descent(w, b, dw, db, learning_rate)
        # Save loss
        losses.append(c)
        # Print progress every 100 epochs
        if i % 100 == 0:
            print(f"Epoch {i} - Cost: {c:.4f}")
    return w, b, losses

# Step 7: Train the Model
w, b, losses = linear_model(X_train, y_train, epochs=1000)

# Step 8: Predictions & Model Evaluation
y_pred = forward_prop(w, b, X_test)
# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score on Test Data: {r2:.4f}")

# Step 9: Plot Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Loss during Training")
plt.show()

# Step 10: Regression Plot for One Feature
feature_name = 'median_income'  # Feature to visualize
if feature_name in df.columns:
    feature_index = df.columns.get_loc(feature_name)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, feature_index], y_test, label='Actual')
    plt.scatter(X_test[:, feature_index], y_pred, color='red', alpha=0.5, label='Predicted')
    plt.xlabel(f"{feature_name} (Standardized)")
    plt.ylabel("Median House Value (Standardized)")
    plt.title(f"Regression: {feature_name} vs Price")
    plt.legend()
    plt.show()
else:
    print(f"Feature '{feature_name}' not found.")

# Step 11: Residual Plot
residuals = y_pred - y_test
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()
