import streamlit as st
import pandas as pd
import time
from models import STTModel, Diarizer, Summarizer
from evaluation import get_benchmark_report, calculate_wer
from export import export_as_json, export_as_markdown, export_as_csv

# Page Configuration
st.set_page_config(page_title="Varshini", layout="wide", page_icon="🎙️")

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
        background-color: 262730;
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
                    <strong>{seg['speaker']}</strong> <span style="color:gray; font-size:0.8em">({seg['start']}s - {seg['end']}s)</span><br>
                    {seg['text']}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("AI Summary")
            st.info(st.session_state.summary)

            st.subheader("Metadata")
            st.write(f"**Speakers Detected:** {len(set(s['speaker'] for s in st.session_state.segments))}")
            st.write(f"**Word Count:** {len(' '.join([s['text'] for s in st.session_state.segments]).split())}")
    else:
        st.warning("Please upload and process an audio file in 'Upload & Process' tab.")

with tab3:
    st.header("Model Benchmarks")
    report = get_benchmark_report()
    df_report = pd.DataFrame(report).T
    st.table(df_report)

    st.subheader("WER Comparison (Comparison Table)")
    chart_data = pd.DataFrame({
        'Model': ['Whisper', 'Vosk'],
        'WER': [report['Whisper']['WER'], report['Vosk']['WER']]
    })
    st.bar_chart(chart_data.set_index('Model'))

    st.markdown("""
    > **WER (Word Error Rate)**: Lower is better. Whisper generally provides higher accuracy but requires more compute.
    """)

with tab4:
    if st.session_state.processing_done:
        st.header("Export Meeting Notes")

        json_str = export_as_json({"segments": st.session_state.segments, "summary": st.session_state.summary})
        md_str = export_as_markdown(st.session_state.segments, st.session_state.summary)
        csv_str = export_as_csv(st.session_state.segments)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download JSON", data=json_str, file_name="meeting_data.json", mime="application/json")
        with col2:
            st.download_button("Download Markdown", data=md_str, file_name="meeting_summary.md", mime="text/markdown")
        with col3:
            st.download_button("Download CSV", data=csv_str, file_name="transcript.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Preview Markdown")
        st.markdown(md_str)
    else:
        st.warning("No data available for export. Please process an audio file first.")
#

def run_streamlit():
    # Launch Streamlit on port 8501
    subprocess.Popen(
        [
            "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ]
    )

Run Streamlit in background
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

Give it a few seconds to start
time.sleep(5)

Create private tunnel (uses your authtoken from Cell 2)
if RESERVED_DOMAIN:
    # Requires paid ngrok plan and configured domain
    tunnel = ngrok.connect(addr=8501, proto="http", domain=RESERVED_DOMAIN)
else:
    tunnel = ngrok.connect(addr=8501, proto="http")

print("Varshini app URL:", tunnel.public_url)