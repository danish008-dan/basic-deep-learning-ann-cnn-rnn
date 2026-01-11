"""
File Name: cnn_basic.py
Purpose:
This file demonstrates a basic Convolutional Neural Network (CNN).
CNN is mainly used for image data like image classification, face detection, etc.

In this example:
- We classify handwritten digits using MNIST dataset
"""

# Import required libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Step 1: Load dataset
# -------------------------------

# MNIST contains images of digits (0-9)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# -------------------------------
# Step 2: Preprocess data
# -------------------------------

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data to fit CNN input
# (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels into one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------------
# Step 3: Build CNN model
# -------------------------------

model = Sequential()

# Convolution layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Pooling layer to reduce size
model.add(MaxPooling2D((2, 2)))

# Flatten feature maps into vector
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer (10 classes)
model.add(Dense(10, activation='softmax'))

# -------------------------------
# Step 4: Compile model
# -------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Step 5: Train model
# -------------------------------

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# -------------------------------
# Step 6: Evaluate model
# -------------------------------

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
