import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Paths
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_FILE = os.path.join(MODELS_PATH, "cnn_model.h5")
ENCODER_FILE = os.path.join(MODELS_PATH, "encoder_classes.npy")

# Load model and encoder
model = load_model(MODEL_FILE)
classes = np.load(ENCODER_FILE, allow_pickle=True)


# MFCC extraction function (same as feature_extraction)
def extract_mfcc(file_path, n_mfcc=40, max_len=1292):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    # pad/truncate
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc[..., np.newaxis]


# Predict function
def predict_genre(file_path):
    mfcc = extract_mfcc(file_path)
    X = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(X)
    genre = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return genre, confidence


# Example usage
if __name__ == "__main__":
    song_path = input("Enter path to .wav file: ").strip()
    if not os.path.exists(song_path):
        print(f"File {song_path} does not exist!")
        exit()

    genre, confidence = predict_genre(song_path)
    print(f"Predicted Genre: {genre} (Confidence: {confidence * 100:.2f}%)")
