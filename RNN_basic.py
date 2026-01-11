"""
File Name: rnn_basic.py
Purpose:
This file demonstrates a basic Recurrent Neural Network (RNN).
RNN is mainly used for sequence data like:
- Time series
- Text
- Speech
- Stock prices

In this example:
- We predict the next number in a sequence
"""

# Import required libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# -------------------------------
# Step 1: Prepare sequence data
# -------------------------------

# Example sequences
# Input: 1,2,3 → Output: 4
# Input: 2,3,4 → Output: 5
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]]
], dtype=float)

y = np.array([[4], [5], [6]], dtype=float)

# -------------------------------
# Step 2: Build RNN model
# -------------------------------

model = Sequential()

# Simple RNN layer with 10 neurons
model.add(SimpleRNN(10, activation='tanh', input_shape=(3, 1)))

# Output layer
model.add(Dense(1))

# -------------------------------
# Step 3: Compile model
# -------------------------------

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# -------------------------------
# Step 4: Train model
# -------------------------------

model.fit(
    X, y,
    epochs=300,
    verbose=1
)

# -------------------------------
# Step 5: Make prediction
# -------------------------------

# Predict next number for sequence [4,5,6]
test_input = np.array([[[4], [5], [6]]], dtype=float)
prediction = model.predict(test_input)

print("Predicted next number:", prediction)
