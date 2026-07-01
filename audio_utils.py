def extract_audio_features(audio_path):

    return {
        "pause_ratio": 0.10,
        "rms_energy": 0.05
    }


def filler_word_ratio(transcript):

    filler_words = ["um", "uh", "like"]

    words = transcript.lower().split()

    filler_count = sum(
        1 for word in words
        if word in filler_words
    )

    return filler_count / max(len(words), 1)