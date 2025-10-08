import librosa
import numpy as np


def extract_mfcc(file_path, n_mfcc=40, duration=30, sr=22050):
    """
    Extract MFCCs from an audio file.
    Returns shape: (time_steps, n_mfcc) or None if failed
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        if len(y) == 0:
            print(f"Warning: Empty audio file {file_path}")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Check if MFCC extraction produced valid results
        if mfcc.shape[0] == 0 or mfcc.shape[1] == 0:
            print(f"Warning: Invalid MFCC shape for {file_path}: {mfcc.shape}")
            return None

        # Normalize
        mfcc_mean = np.mean(mfcc)
        mfcc_std = np.std(mfcc)
        if mfcc_std > 0:
            mfcc = (mfcc - mfcc_mean) / mfcc_std
        else:
            print(f"Warning: Zero standard deviation for {file_path}")
            return None

        return mfcc.T  # transpose: time_steps x n_mfcc

    except Exception as e:
        print(f"Error extracting MFCC from {file_path}: {str(e)}")
        return None


def segment_audio(file_path, segment_length=30):
    """
    Split a long audio file into segments of 'segment_length' seconds
    Returns list of MFCC arrays or empty list if failed
    """
    try:
        y, sr = librosa.load(file_path, sr=22050)
        if len(y) == 0:
            print(f"Warning: Empty audio file {file_path}")
            return []

        segments = []
        total_samples = len(y)
        samples_per_segment = segment_length * sr

        for start in range(0, total_samples, samples_per_segment):
            end = min(start + samples_per_segment, total_samples)
            y_segment = y[start:end]
            if len(y_segment) < samples_per_segment:
                break  # ignore last short segment

            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=40)

            # Check if MFCC extraction produced valid results
            if mfcc.shape[0] == 0 or mfcc.shape[1] == 0:
                print(
                    f"Warning: Invalid MFCC shape for segment in {file_path}: {mfcc.shape}"
                )
                continue

            # Normalize
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            if mfcc_std > 0:
                mfcc = (mfcc - mfcc_mean) / mfcc_std
            else:
                print(f"Warning: Zero standard deviation for segment in {file_path}")
                continue

            segments.append(mfcc.T)

        return segments

    except Exception as e:
        print(f"Error segmenting audio file {file_path}: {str(e)}")
        return []
