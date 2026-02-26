import os
import sys
import argparse
from src.audio_converter import convert_audio
from src.audio_streamer import AudioStreamer

from src.feature_extractor import compute_mfcc
from src.language_classifier import LanguageClassifier
from src.analytics_engine import AnalyticsEngine
from src.visualization import Visualization
from src.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Audio Intelligence & Human Communication Analytics System")
    parser.add_argument("input_file", help="Path to input audio file")
    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "graphs"), exist_ok=True)

    print("1. Converting audio to 16kHz, mono, 16-bit PCM WAV (temp file)...")
    temp_wav = convert_audio(input_path)

    print("2. Streaming audio and detecting voice activity using Neural VAD...")
    streamer = AudioStreamer(temp_wav, frame_duration_ms=20)
    
    from src.neural_vad import run_neural_vad
    timestamps = run_neural_vad(temp_wav)

    print("3. Extracting features and detecting language...")
    features = compute_mfcc(temp_wav, timestamps)
    clf = LanguageClassifier()
    language, conf = clf.predict(temp_wav, timestamps)

    print("4. Analyzing communication behavior...")
    total_duration = streamer.get_duration()
    analytics = AnalyticsEngine(timestamps, total_duration)
    stats = analytics.synthesize()

    stats["language"] = language
    stats["language_confidence"] = conf

    print("5. Generating visualizations...")
    viz = Visualization(timestamps, total_duration)
    viz.generate_all()

    print("6. Generating final reports...")
    report_gen = ReportGenerator(timestamps, stats)
    report_gen.export_csv("outputs/speech_segments.csv")
    report_gen.export_txt("outputs/report.txt")
    report_gen.print_console()

    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass
    print("Processing complete! Check the 'outputs' folder.")

if __name__ == "__main__":
    main()
