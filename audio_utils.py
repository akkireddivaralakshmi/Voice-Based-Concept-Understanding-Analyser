import librosa
import numpy as np


def extract_audio_features(audio_path):

    y, sr = librosa.load(audio_path)

    rms = np.mean(librosa.feature.rms(y=y))

    silence = np.sum(np.abs(y) < 0.01)

    pause_ratio = silence / len(y)

    return {
        "pause_ratio": float(pause_ratio),
        "rms_energy": float(rms)
    }


def filler_word_ratio(transcript):

    filler_words = [
        "um", "uh", "like", "you know",
        "actually", "basically", "so"
    ]

    words = transcript.lower().split()

    filler_count = sum(
        1 for word in words
        if word in filler_words
    )

    return filler_count / max(len(words), 1)
