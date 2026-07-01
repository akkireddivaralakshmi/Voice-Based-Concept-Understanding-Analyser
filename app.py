import streamlit as st

from speech_to_text import speech_to_text
from semantic_eval import semantic_similarity
from scoring_engine import evaluate_understanding
from audio_utils import (
    extract_audio_features,
    filler_word_ratio
)

st.set_page_config(
    page_title="Voice Based Concept Understanding Analyser",
    page_icon="🎤"
)

st.title("🎤 Voice Based Concept Understanding Analyser")

audio_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if audio_file:

    st.success("Audio Uploaded Successfully")

    if st.button("Analyze Concept Understanding"):

        transcript = speech_to_text(
            "sample_audio.wav"
        )

        similarity = semantic_similarity(
            transcript,
            "Machine Learning"
        )

        audio = extract_audio_features(
            "sample_audio.wav"
        )

        filler_ratio = filler_word_ratio(
            transcript
        )

        score, level = evaluate_understanding(
            similarity,
            filler_ratio,
            audio
        )

        st.subheader("Analysis Results")

        st.write("Transcript:")
        st.write(transcript)

        st.write(
            f"Semantic Similarity: {similarity}"
        )

        st.write(
            f"Understanding Score: {score}"
        )

        st.success(
            f"Level: {level}"
        )