import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_PATH, exist_ok=True)

# Load features and labels
X = np.load(os.path.join(MODELS_PATH, "X.npy"))
y = np.load(os.path.join(MODELS_PATH, "y.npy"))

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Build CNN
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=X_train.shape[1:]),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save(os.path.join(MODELS_PATH, "cnn_model.h5"))
print("Model saved at models/cnn_model.h5")
