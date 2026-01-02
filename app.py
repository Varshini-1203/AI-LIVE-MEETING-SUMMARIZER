import streamlit as st
import pandas as pd
import time
from models import STTModel, Diarizer, Summarizer
from evaluation import get_benchmark_report, calculate_wer
from export import export_as_json, export_as_markdown, export_as_csv

# Page Configuration
st.set_page_config(page_title="Varshini AI", layout="wide", page_icon="🎙️")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #262730;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #E6E6FA;
        color: black;
    }
    .stProgress .st-bo {
        background-color: #262730;
    }
    .speaker-card {
        padding: 10px;
        border-radius: 10px;
        background-color: #262730;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎙️ Varshini AI")
st.markdown("---")

# Session State Initialization
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Select STT Model", ["Whisper (High Accuracy)", "Vosk (Fast/Local)"])
use_diarization = st.sidebar.checkbox("Enable Speaker Diarization", value=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📝 Analysis & Summary", "📊 Benchmarks", "💾 Export"])

with tab1:
    st.header("Upload Meeting Recording")
    audio_file = st.file_uploader("Upload meeting recording (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Start Processing"):
            import tempfile
            import os

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.getbuffer())
                audio_path = tmp_file.name

            try:
                with st.status("Processing Audio...", expanded=True) as status:

                    st.write("📌 Step 1: Initializing models...")
                    try:
                        stt = STTModel(model_name="whisper" if "Whisper" in model_choice else "vosk")
                        diarizer = Diarizer()
                        summarizer = Summarizer()
                        st.write("✓ Models initialized")
                    except Exception as e:
                        st.error(f"Failed to initialize models: {str(e)}")
                        raise

                    st.write("🎙️ Step 2: Transcribing audio (1-3 minutes for longer files)...")
                    try:
                        full_transcript = stt.transcribe(audio_path)
                        if "ERROR" in full_transcript or "failed" in full_transcript.lower():
                            st.error(f"Transcription error: {full_transcript}")
                            raise Exception(full_transcript)
                        st.write(f"✓ Transcription complete ({len(full_transcript)} characters)")
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                        raise

                    st.write("👥 Step 3: Extracting speaker segments...")
                    try:
                        st.session_state.segments = diarizer.get_segments(audio_path)
                        if st.session_state.segments:
                            st.session_state.segments[0]["text"] = full_transcript
                        st.write(f"✓ Extracted {len(st.session_state.segments)} segment(s)")
                    except Exception as e:
                        st.error(f"Diarization failed: {str(e)}")
                        raise

                    st.write("✨ Step 4: Generating summary...")
                    try:
                        st.session_state.summary = summarizer.summarize(full_transcript)
                        st.write(f"✓ Summary generated ({len(st.session_state.summary)} characters)")
                    except Exception as e:
                        st.error(f"Summarization failed: {str(e)}")
                        raise

                    st.session_state.processing_done = True
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                    st.success("Successfully processed meeting!")

            except Exception as e:
                st.error(f"❌ Processing Error: {str(e)}")
                import traceback
                st.write("**Traceback:**")
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    else:
        st.info("📤 Upload an audio file (WAV, MP3, or M4A) to get started!")

with tab2:
    if st.session_state.processing_done:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Transcript")
            for seg in st.session_state.segments:
                st.markdown(f"""
                <div class="speaker-card">
                    <b>{seg['speaker']}</b> ({seg['start']}s – {seg['end']}s)<br>
                    {seg['text']}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("AI Summary")
            st.info(st.session_state.summary)

            st.subheader("Metrics")
            word_count = sum(len(seg['text'].split()) for seg in st.session_state.segments)
            st.metric("Total Words", word_count)
            st.metric("Speakers Detected", len(set(seg['speaker'] for seg in st.session_state.segments)))
    else:
        st.warning("⚠️ Please process an audio file first in the Upload & Process tab")

with tab3:
    st.header("Model Benchmarks")

    benchmarks = get_benchmark_report()
    benchmark_df = pd.DataFrame(benchmarks).T
    st.dataframe(benchmark_df, use_container_width=True)

    st.subheader("Performance Comparison")
    st.bar_chart(benchmark_df["WER"])

    if st.session_state.processing_done:
        st.subheader("Current Session Quality")
        reference_text = "sample reference text"
        hypothesis_text = st.session_state.segments[0]["text"] if st.session_state.segments else ""
        wer = calculate_wer(reference_text, hypothesis_text)
        st.metric("Word Error Rate (WER)", f"{wer:.2%}")

with tab4:
    st.header("Export Meeting Data")

    if st.session_state.processing_done:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("JSON Format")
            json_data = export_as_json({
                "segments": st.session_state.segments,
                "summary": st.session_state.summary
            })
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="meeting_transcript.json",
                mime="application/json"
            )

        with col2:
            st.subheader("Markdown Format")
            md_data = export_as_markdown(st.session_state.segments, st.session_state.summary)
            st.download_button(
                label="📥 Download Markdown",
                data=md_data,
                file_name="meeting_transcript.md",
                mime="text/markdown"
            )

        with col3:
            st.subheader("CSV Format")
            csv_data = export_as_csv(st.session_state.segments)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="meeting_transcript.csv",
                mime="text/csv"
            )

        st.markdown("---")
        st.subheader("Preview")
        preview_format = st.selectbox("Select format to preview", ["JSON", "Markdown", "CSV"])

        if preview_format == "JSON":
            st.code(json_data, language="json")
        elif preview_format == "Markdown":
            st.markdown(md_data)
        else:
            st.code(csv_data, language="csv")
    else:
        st.warning("⚠️ No data to export. Please process an audio file first.")
