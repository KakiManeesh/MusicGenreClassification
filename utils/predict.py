import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import segment_audio
from sklearn.preprocessing import LabelEncoder

# Load model and classes
model = load_model("../models/cnn_model.h5")
classes = np.load("../models/encoder_classes.npy", allow_pickle=True)


def predict_song(file_path):
    segments = segment_audio(file_path)
    preds = []
    for seg in segments:
        seg = seg[np.newaxis, ..., np.newaxis]  # reshape for CNN
        pred = model.predict(seg)
        preds.append(pred)
    preds = np.array(preds)
    avg_pred = np.mean(preds, axis=0)
    genre_idx = np.argmax(avg_pred)
    return classes[genre_idx]


# Example usage
# genre = predict_song("../data/my_song.wav")
# print("Predicted Genre:", genre)
