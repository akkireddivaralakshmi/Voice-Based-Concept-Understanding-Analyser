import streamlit as st

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