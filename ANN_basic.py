"""
File Name: ann_basic.py
Purpose:
This file demonstrates a very basic Artificial Neural Network (ANN).
ANN is mainly used for structured/tabular data like marks, salary prediction,
house price prediction, etc.

In this example:
- We create a simple ANN using Keras
- Train it on dummy numeric data
"""

# Import required libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Step 1: Create sample dataset
# -------------------------------

# Input data (features)
# Example: [hours studied]
X = np.array([[1], [2], [3], [4], [5]], dtype=float)

# Output data (labels)
# Example: [marks scored]
y = np.array([[20], [40], [60], [80], [100]], dtype=float)

# -------------------------------
# Step 2: Build ANN model
# -------------------------------

# Sequential model means layers are added one after another
model = Sequential()

# Hidden layer with 10 neurons and ReLU activation
model.add(Dense(10, activation='relu', input_shape=(1,)))

# Output layer with 1 neuron (regression problem)
model.add(Dense(1))

# -------------------------------
# Step 3: Compile the model
# -------------------------------

model.compile(
    optimizer='adam',      # Optimizer to minimize loss
    loss='mean_squared_error'  # Loss function for regression
)

# -------------------------------
# Step 4: Train the model
# -------------------------------

model.fit(
    X, y,
    epochs=200,            # Number of training iterations
    verbose=1              # Show training progress
)

# -------------------------------
# Step 5: Make prediction
# -------------------------------

# Predict marks if student studies 6 hours
prediction = model.predict([[6]])

print("Predicted Marks for 6 hours study:", prediction)
