"""
Voice Based Concept Understanding Analyser (VBCUA)
Professional Streamlit Frontend
----------------------------------------------------
Backend modules (NOT modified):
    speech_to_text.py
    semantic_eval.py
    scoring_engine.py
    audio_utils.py
    report_generator.py
"""

import io
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from speech_to_text import speech_to_text
from semantic_eval import semantic_similarity
from scoring_engine import evaluate_understanding
from audio_utils import (
    extract_audio_features,
    filler_word_ratio,
)

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Voice Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #050816, #0a192f);
    color: white;
}

/* Cards */
div[data-testid="stVerticalBlock"] > div {
    border-radius: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2575fc, #6a11cb);
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 15px;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    border: 2px dashed #7c3aed;
    border-radius: 15px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)
REFERENCE_CONCEPT = (
    "Machine Learning is a subset of artificial intelligence that allows systems "
    "to learn patterns from data and improve performance without being explicitly "
    "programmed."
)

# --------------------------------------------------------------------------------------
# CUSTOM DARK THEME CSS
# --------------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e6e6e6;
    }

    .vbcua-header {
        padding: 28px 32px;
        border-radius: 14px;
        background: linear-gradient(135deg, #131a2b 0%, #1b2338 100%);
        border: 1px solid #2a3352;
        margin-bottom: 28px;
    }
    .vbcua-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: 0.3px;
    }
    .vbcua-subtitle {
        font-size: 15px;
        color: #9aa4c4;
        margin-top: 6px;
        font-weight: 400;
    }

    .vbcua-card {
        background-color: #131a2b;
        border: 1px solid #262f4a;
        border-radius: 14px;
        padding: 22px 24px;
        height: 100%;
    }

    .vbcua-card-title {
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .vbcua-reference-text {
        font-size: 14.5px;
        line-height: 1.65;
        color: #c6cde6;
        background-color: #0f1524;
        border-left: 4px solid #4f7cff;
        padding: 14px 16px;
        border-radius: 8px;
    }

    .vbcua-transcript-box {
        font-size: 14.5px;
        line-height: 1.7;
        color: #e6e6e6;
        background-color: #0f1524;
        border: 1px solid #262f4a;
        border-radius: 10px;
        padding: 16px 18px;
        min-height: 160px;
    }

    .vbcua-eval-card {
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        border: 1px solid;
    }
    .vbcua-eval-score {
        font-size: 46px;
        font-weight: 800;
        margin: 4px 0 0 0;
    }
    .vbcua-eval-level {
        font-size: 18px;
        font-weight: 700;
        margin-top: 6px;
        letter-spacing: 0.4px;
    }
    .vbcua-eval-label {
        font-size: 13px;
        color: #9aa4c4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .vbcua-strong {
        background-color: rgba(46, 204, 113, 0.08);
        border-color: #2ecc71;
    }
    .vbcua-strong .vbcua-eval-score, .vbcua-strong .vbcua-eval-level {
        color: #2ecc71;
    }
    .vbcua-moderate {
        background-color: rgba(243, 156, 18, 0.08);
        border-color: #f39c12;
    }
    .vbcua-moderate .vbcua-eval-score, .vbcua-moderate .vbcua-eval-level {
        color: #f39c12;
    }
    .vbcua-poor {
        background-color: rgba(231, 76, 60, 0.08);
        border-color: #e74c3c;
    }
    .vbcua-poor .vbcua-eval-score, .vbcua-poor .vbcua-eval-level {
        color: #e74c3c;
    }

    .vbcua-section-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin: 30px 0 14px 0;
        border-left: 4px solid #4f7cff;
        padding-left: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #131a2b;
        border: 1px solid #262f4a;
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #4f7cff;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
    }
    .stButton > button:hover {
        background-color: #3d63d1;
        color: white;
    }

    .stDownloadButton > button {
        background-color: #1f2740;
        color: #e6e6e6;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #3a4470;
    }
    .stDownloadButton > button:hover {
        border-color: #4f7cff;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# --------------------------------------------------------------------------------------

DEFAULT_STATE = {
    "audio_bytes": None,
    "audio_name": None,
    "audio_ext": None,
    "audio_path": None,
    "analyzed": False,
    "transcript": None,
    "similarity": None,
    "filler_ratio": None,
    "audio_features": None,
    "final_score": None,
    "understanding_level": None,
    "error_message": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results():
    """Clear previous analysis results (called on new upload)."""
    for key in [
        "analyzed",
        "transcript",
        "similarity",
        "filler_ratio",
        "audio_features",
        "final_score",
        "understanding_level",
        "error_message",
    ]:
        st.session_state[key] = DEFAULT_STATE[key]


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """Persist uploaded audio to a temp file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(
        tmp_dir, f"vbcua_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{suffix}"
    )
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


def plot_waveform(audio_path: str, audio_ext: str):
    """
    Attempt to plot the real waveform. If the audio cannot be decoded
    (unsupported codec, corrupted file, missing libraries) fall back to a
    stylised matplotlib placeholder waveform instead of failing.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 2.4), dpi=120)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")

    real_waveform_plotted = False

    if audio_ext == ".wav":
        try:
            import wave

            with wave.open(audio_path, "rb") as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                sampwidth = wf.getsampwidth()
                dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
                signal = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                if wf.getnchannels() > 1:
                    signal = signal[:: wf.getnchannels()]
                if signal.size > 0:
                    signal = signal / (np.max(np.abs(signal)) + 1e-9)
                    x = np.linspace(0, len(signal) / max(wf.getframerate(), 1), num=len(signal))
                    ax.plot(x, signal, color="#4f7cff", linewidth=0.6)
                    real_waveform_plotted = True
        except Exception:
            real_waveform_plotted = False

    if not real_waveform_plotted:
        # Stylised placeholder waveform
        x = np.linspace(0, 6, 800)
        rng = np.random.default_rng(42)
        signal = (
            np.sin(2 * np.pi * 1.5 * x)
            * np.exp(-0.15 * x)
            * (0.6 + 0.4 * rng.random(x.shape))
        )
        ax.plot(x, signal, color="#4f7cff", linewidth=0.8)
        ax.text(
            0.5,
            0.92,
            "Preview waveform (actual waveform unavailable)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#9aa4c4",
        )

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def score_to_css_class(level: str) -> str:
    level_lower = (level or "").lower()
    if "strong" in level_lower:
        return "vbcua-strong"
    if "moderate" in level_lower:
        return "vbcua-moderate"
    return "vbcua-poor"


def build_pdf_report() -> bytes:
    """
    Build a downloadable PDF report from the analysis results stored in
    session_state. report_generator.py is currently empty, so the report
    is composed here using fpdf2 without touching any backend module.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError(
            "The 'fpdf2' package is required to generate PDF reports. "
            "Install it with: pip install fpdf2"
        )

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 30, 60)
    pdf.cell(0, 12, "Voice Based Concept Understanding Analyser", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)

    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)

    pdf.cell(
        0,
        8,
        f"Generated on: {ist_time.strftime('%Y-%m-%d %H:%M:%S IST')}",
        ln=True
    )

    pdf.ln(4)
    def section_title(text):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 30, 60)
        pdf.cell(0, 10, text, ln=True)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.get_x(), pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    def body_text(text):
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 7, text)
        pdf.ln(2)

    section_title("Reference Concept")
    body_text(REFERENCE_CONCEPT)

    section_title("Transcribed Explanation")
    body_text(st.session_state.transcript or "N/A")

    section_title("Evaluation Metrics")
    similarity = st.session_state.similarity
    filler_ratio = st.session_state.filler_ratio
    features = st.session_state.audio_features or {}
    body_text(
        f"Semantic Similarity: {similarity * 100:.1f}%\n"
        f"Filler Word Ratio: {filler_ratio * 100:.1f}%\n"
        f"Confidence (Energy): {features.get('rms_energy', 0):.3f}\n"
        f"Pause Ratio: {features.get('pause_ratio', 0):.3f}"
    )
    section_title("Final Result")

    body_text(
        f"Understanding Score: {st.session_state.final_score} / 100\n"
        f"Understanding Level: {st.session_state.understanding_level}"
    )

    # Footer Credit
    pdf.ln(15)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)

    pdf.cell(
        0,
        5,
        "Developed By Akkireddi Varalakshmi",
        ln=True,
        align="C"
    )

    output = pdf.output(dest="S")

    if isinstance(output, str):
        output = output.encode("latin-1")

    return bytes(output)
   


# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------

st.markdown(
    """
    <div class="vbcua-header">
        <p class="vbcua-title">🎤 Voice Based Concept Understanding Analyser</p>
        <p class="vbcua-subtitle">Automated evaluation of spoken conceptual explanations using AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# TOP SECTION: AUDIO UPLOAD  |  CONCEPT REFERENCE
# --------------------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="vbcua-card">', unsafe_allow_html=True)
    st.markdown('<div class="vbcua-card-title">📤 Audio Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a WAV or MP3 file of the spoken explanation",
        type=["wav", "mp3"],
        help="Supported formats: .wav, .mp3",
    )

    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext not in [".wav", ".mp3"]:
            st.error("❌ Unsupported file format. Please upload a WAV or MP3 file.")
        else:
            # New file uploaded -> reset previous results
            if st.session_state.audio_name != uploaded_file.name:
                reset_results()
                st.session_state.audio_bytes = uploaded_file.getvalue()
                st.session_state.audio_name = uploaded_file.name
                st.session_state.audio_ext = ext
                try:
                    st.session_state.audio_path = save_uploaded_file(uploaded_file)
                except Exception:
                    st.session_state.error_message = "⚠️ The audio file appears to be corrupted or unreadable."

            st.success(f"✅ '{uploaded_file.name}' uploaded successfully")
            st.audio(st.session_state.audio_bytes)
    else:
        if st.session_state.audio_name is not None:
            st.info(f"Currently loaded: {st.session_state.audio_name}")
            st.audio(st.session_state.audio_bytes)
        else:
            st.warning("⚠️ No audio file uploaded yet. Please upload a WAV or MP3 file to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="vbcua-card">', unsafe_allow_html=True)
    st.markdown('<div class="vbcua-card-title">📘 Concept Reference</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="vbcua-reference-text">{REFERENCE_CONCEPT}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# WAVEFORM VISUALISATION
# --------------------------------------------------------------------------------------

if st.session_state.audio_bytes is not None and st.session_state.audio_path:
    st.markdown('<div class="vbcua-section-title">🌊 Waveform Visualisation</div>', unsafe_allow_html=True)
    try:
        fig = plot_waveform(st.session_state.audio_path, st.session_state.audio_ext)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception:
        st.info("Waveform preview unavailable for this file.")

# --------------------------------------------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------------------------------------------

analyze_disabled = st.session_state.audio_bytes is None

if analyze_disabled:
    st.button("🧠 Analyze Concept Understanding", disabled=True)
else:
    if st.button("🧠 Analyze Concept Understanding"):
        if not st.session_state.audio_path or not os.path.exists(st.session_state.audio_path):
            st.error("⚠️ No valid audio file found. Please re-upload your audio.")
        else:
            with st.spinner("Processing and evaluating..."):

                try:
                    transcript = speech_to_text(st.session_state.audio_path)

                    similarity = semantic_similarity(
                        transcript,
                        REFERENCE_CONCEPT
                    )

                    audio_features = extract_audio_features(
                        st.session_state.audio_path
                    )

                    filler_ratio = filler_word_ratio(transcript)

                    final_score, understanding_level = evaluate_understanding(
                        similarity,
                        filler_ratio,
                        audio_features
                    )

                    st.session_state.transcript = transcript
                    st.session_state.similarity = similarity
                    st.session_state.filler_ratio = filler_ratio
                    st.session_state.audio_features = audio_features
                    st.session_state.final_score = final_score
                    st.session_state.understanding_level = understanding_level
                    st.session_state.analyzed = True
                    st.session_state.error_message = None

                except FileNotFoundError:
                    st.session_state.error_message = (
                        "⚠️ The uploaded audio file could not be found or read."
                    )

                except Exception as e:
                    st.session_state.error_message = str(e)

if st.session_state.error_message:
    st.error(st.session_state.error_message)
# --------------------------------------------------------------------------------------
# RESULTS DASHBOARD
# --------------------------------------------------------------------------------------

if st.session_state.analyzed:
    st.markdown('<div class="vbcua-section-title">📊 Results Dashboard</div>', unsafe_allow_html=True)

    res_left, res_right = st.columns([1.3, 1], gap="large")

    with res_left:
        st.markdown('<div class="vbcua-card">', unsafe_allow_html=True)
        st.markdown('<div class="vbcua-card-title">📝 Transcribed Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="vbcua-transcript-box">{st.session_state.transcript}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with res_right:
        css_class = score_to_css_class(st.session_state.understanding_level)
        st.markdown(
            f"""
            <div class="vbcua-eval-card {css_class}">
                <div class="vbcua-eval-label">Understanding Score</div>
                <div class="vbcua-eval-score">{st.session_state.final_score}/100</div>
                <div class="vbcua-eval-level">{st.session_state.understanding_level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- Metric Cards ----------------
    st.markdown('<div class="vbcua-section-title">📈 Detailed Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label="Semantic Similarity",
            value=f"{st.session_state.similarity * 100:.1f}%",
        )

    with m2:
        st.metric(
            label="Filler Word Ratio",
            value=f"{st.session_state.filler_ratio * 100:.1f}%",
        )

    with m3:
        energy = (st.session_state.audio_features or {}).get("rms_energy", 0)
        st.metric(
            label="Confidence (Energy)",
            value=f"{energy:.3f}",
        )

    # ---------------- Download Report ----------------
    st.markdown('<div class="vbcua-section-title">📄 Report</div>', unsafe_allow_html=True)

    try:
        pdf_bytes = build_pdf_report()
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"VBCUA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    except RuntimeError as e:
        st.warning(str(e))
    except Exception:
        st.warning("⚠️ Unable to generate the PDF report at this time.")
st.markdown("""
<hr>

<div style="
text-align:center;
padding:15px;
border-radius:12px;
background:#111827;
color:white;
margin-top:20px;
border:1px solid #1f2937;
">

<h3>Akkireddi Varalakshmi</h3>

<p style="
color:#9ca3af;
font-size:15px;
margin-top:0;
">
Project Lead | AI & Full Stack Developer
</p>

<p style="font-size:14px;">
📧 akkireddi.varalakshmi24@gmail.com
</p>

<p style="font-size:14px;">
🔗 <a href="https://github.com/akkireddivaralakshmi"
target="_blank"
style="color:#60a5fa;text-decoration:none;">
github.com/akkireddivaralakshmi
</a>
</p>

<h4 style="margin-top:15px;">Team Members</h4>

<p style="
color:#9ca3af;
font-size:14px;
line-height:1.8;
">
T. Chaitanya Kumar<br>
I. V. V. Durga Prasad<br>
V. Chanakya Eswara Rao
</p>

<p style="
color:#9ca3af;
font-size:13px;
">
Version 1.0
</p>

<p style="
color:#9ca3af;
font-size:13px;
">
Built with Python • Streamlit • Whisper AI • Machine Learning
</p>

<hr style="
border:0;
height:1px;
background:#374151;
margin:10px 0;
">

<p style="
font-size:12px;
color:#6b7280;
margin:0;
">
© 2026 Voice Based Concept Understanding Analyser
</p>

</div>
""", unsafe_allow_html=True)
