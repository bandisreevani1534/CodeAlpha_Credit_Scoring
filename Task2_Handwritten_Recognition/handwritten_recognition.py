import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("New training shape:", X_train.shape)
print("New testing shape:", X_test.shape)
# Build CNN model
model = Sequential()

# First Convolution Layer
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(28, 28, 1)))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolution Layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convert data into one dimension
model.add(Flatten())

# Hidden Layer
model.add(Dense(64, activation='relu'))

# Output Layer (digits 0 to 9)
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("CNN Model Created Successfully!")
# Train the model
print("\nTraining the model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

print("\nModel Training Completed!")
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy * 100, "%")
print("Test Loss:", test_loss)
# Predict a handwritten digit
prediction = model.predict(X_test)

# Select an image
image_number = 0

# Get predicted digit
predicted_digit = prediction[image_number].argmax()

# Actual digit
actual_digit = y_test[image_number]

print("\nPredicted Digit:", predicted_digit)
print("Actual Digit:", actual_digit)

# Display the image
plt.imshow(X_test[image_number].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_digit} | Actual: {actual_digit}")
plt.axis('off')
plt.show()