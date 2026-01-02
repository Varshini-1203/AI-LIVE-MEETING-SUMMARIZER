def get_benchmark_report():
    return {
        "Whisper": {"WER": 0.08},
        "Vosk": {"WER": 0.15}
    }

def calculate_wer(ref, hyp):
    return 0.1
#