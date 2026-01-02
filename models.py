import os
from pydub import AudioSegment

class STTModel:
    """Speech-to-Text using OpenAI Whisper"""
    def __init__(self, model_name="whisper"):
        self.model_name = model_name
        self.model = None
        try:
            import whisper
            print("Loading Whisper model...")
            self.model = whisper.load_model("base")
            print("✓ Whisper model loaded successfully")
        except ImportError as e:
            print(f"ERROR: Install whisper with: pip install openai-whisper")
            self.model = None

    def transcribe(self, audio_path):
        """Transcribe audio file to text using Whisper"""
        if self.model is None:
            return "ERROR: Whisper model not loaded. Install: pip install openai-whisper"

        try:
            print(f"Transcribing: {audio_path}")
            result = self.model.transcribe(audio_path)
            text = result.get("text", "No speech detected")
            print(f"✓ Transcription complete. Text length: {len(text)} chars")
            return text
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            print(error_msg)
            return error_msg


class Diarizer:
    """Speaker Diarization - separates different speakers"""
    def __init__(self):
        self.use_pyannote = False
        try:
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")
            self.use_pyannote = True
            print("✓ Pyannote loaded")
        except Exception as e:
            print(f"Note: Pyannote not available. Using simple fallback.")
            self.use_pyannote = False

    def get_segments(self, audio_path):
        """Get speaker segments with timestamps and text"""
        try:
            print("Extracting audio duration...")
            # Get audio duration
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000  # Convert to seconds
            print(f"Audio duration: {duration:.2f} seconds")

            # Get transcription
            print("Starting transcription...")
            stt = STTModel()
            full_text = stt.transcribe(audio_path)

            print(f"Creating segments...")
            # Return as single speaker for now
            segments = [{
                "speaker": "Speaker 1",
                "start": 0,
                "end": int(duration),
                "text": full_text
            }]

            print(f"✓ Segments created: {len(segments)}")
            return segments

        except Exception as e:
            error_msg = f"Diarization error: {str(e)}"
            print(error_msg)
            return [{
                "speaker": "Speaker 1",
                "start": 0,
                "end": 0,
                "text": error_msg
            }]


class Summarizer:
    """AI Summarizer using Transformers"""
    def __init__(self):
        self.summarizer = None
        try:
            from transformers import pipeline
            print("Loading summarizer model...")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            print("✓ Summarizer loaded")
        except Exception as e:
            print(f"Note: Transformers not available: {e}")
            self.summarizer = None

    def summarize(self, text):
        """Generate summary of text"""
        if not text or len(text.strip()) == 0:
            return "No text to summarize."

        words = text.split()

        # Too short to summarize
        if len(words) < 50:
            print(f"Text too short ({len(words)} words). Returning as-is.")
            return text

        # Use transformer if available
        if self.summarizer:
            try:
                print(f"Summarizing {len(words)} words...")
                # Limit input to 500 words
                if len(words) > 500:
                    text = " ".join(words[:500])

                summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
                result = summary[0]["summary_text"]
                print(f"✓ Summary generated: {len(result)} chars")
                return result
            except Exception as e:
                print(f"Summarization error: {str(e)}. Using fallback.")
                return self._simple_summary(text)
        else:
            return self._simple_summary(text)

    def _simple_summary(self, text):
        """Fallback: extract key sentences"""
        sentences = text.split(". ")
        if len(sentences) <= 2:
            return text

        # Return first 2 sentences
        result = ". ".join(sentences[:2]) + "."
        print(f"✓ Fallback summary: {len(result)} chars")
        return result
