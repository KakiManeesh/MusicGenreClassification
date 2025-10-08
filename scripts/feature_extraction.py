import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import librosa
import csv
import warnings

warnings.filterwarnings("ignore")

from utils.preprocess import extract_mfcc, segment_audio
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "genres", "genres_original"
)
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

# Ensure models directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

# Check if data path exists
if not os.path.exists(DATA_PATH):
    print(f"Error: Data path {DATA_PATH} does not exist!")
    sys.exit(1)

genres = os.listdir(DATA_PATH)
print(f"Found genres: {genres}")

X, y = [], []
processed_files = 0
failed_files = 0

for genre in genres:
    genre_path = os.path.join(DATA_PATH, genre)
    if not os.path.isdir(genre_path):
        continue

    print(f"Processing genre: {genre}")
    for file in os.listdir(genre_path):
        if file.endswith(".wav") or file.endswith(".au"):
            file_path = os.path.join(genre_path, file)
            try:
                mfcc = extract_mfcc(file_path)
                if mfcc is not None and mfcc.shape[0] > 0:
                    X.append(mfcc)
                    y.append(genre)
                    processed_files += 1
                else:
                    failed_files += 1
                    print(f"Warning: Empty MFCC for {file_path}")
            except Exception as e:
                failed_files += 1
                print(f"Error processing {file_path}: {str(e)}")

print(f"Processed {processed_files} files successfully, {failed_files} files failed.")

if len(X) == 0:
    print("Error: No files were successfully processed!")
    sys.exit(1)

# Pad/reshape to same size for CNN
max_len = max([mf.shape[0] for mf in X])
print(f"Maximum sequence length: {max_len}")

X_padded = np.array([np.pad(mf, ((0, max_len - mf.shape[0]), (0, 0))) for mf in X])
X_padded = X_padded[..., np.newaxis]  # add channel dimension

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Save for training
try:
    np.save(os.path.join(MODELS_PATH, "X.npy"), X_padded)
    np.save(os.path.join(MODELS_PATH, "y.npy"), y_cat)
    np.save(os.path.join(MODELS_PATH, "encoder_classes.npy"), encoder.classes_)
    print("Feature extraction completed successfully!")
    print(f"Saved X.npy shape: {X_padded.shape}")
    print(f"Saved y.npy shape: {y_cat.shape}")
    print(f"Number of classes: {len(encoder.classes_)}")
except Exception as e:
    print(f"Error saving files: {str(e)}")
    sys.exit(1)
