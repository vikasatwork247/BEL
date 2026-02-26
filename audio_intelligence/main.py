import os
import sys
import argparse

from src.audio_converter import convert_audio, standardize_audio
from src.vocal_separator import separate_vocals, needs_vocal_separation
from src.audio_streamer import AudioStreamer
from src.feature_extractor import compute_mfcc
from src.language_classifier import LanguageClassifier
from src.analytics_engine import AnalyticsEngine
from src.visualization import Visualization
from src.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Audio Intelligence & Human Communication Analytics System"
    )
    parser.add_argument("input_file", help="Path to input audio file")
    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "graphs"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "temp"), exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Vocal Separation (for music / compressed audio)             #
    # WHY: Neural VAD is trained on clean speech. Instruments mask vocal  #
    # frequencies => every frame stays below threshold => zero segments.  #
    # Demucs isolates only the vocals stem so VAD gets a clean signal.    #
    # ------------------------------------------------------------------ #
    if needs_vocal_separation(input_path):
        print("1. Input is music/compressed — running vocal source separation (Demucs)...")
        vocals_path = separate_vocals(input_path)
    else:
        print("1. Input is plain audio — skipping vocal separation.")
        vocals_path = input_path

    # ------------------------------------------------------------------ #
    # STEP 2: Standardize Audio                                           #
    # WHY: Silero VAD requires EXACTLY 16kHz mono float32 PCM.           #
    # pydub handles resampling, mono conversion, and loudness norm.      #
    # ------------------------------------------------------------------ #
    print("2. Standardizing audio to 16kHz mono -20dBFS WAV (pydub)...")
    standardized_path = standardize_audio(vocals_path)

    # ------------------------------------------------------------------ #
    # STEP 3: Neural VAD — runs ONLY on the standardized file            #
    # ------------------------------------------------------------------ #
    print("3. Running Neural VAD (Silero, threshold=0.35)...")
    from src.neural_vad import run_neural_vad
    timestamps = run_neural_vad(standardized_path)

    if not timestamps:
        print("WARNING: No speech segments detected. Check the audio content.")

    # ------------------------------------------------------------------ #
    # STEP 4: Feature extraction and language detection                   #
    # Language classifier runs ONLY on detected speech segments          #
    # ------------------------------------------------------------------ #
    print("4. Extracting features and detecting language from speech segments...")
    features = compute_mfcc(standardized_path, timestamps)
    clf = LanguageClassifier()
    language, conf = clf.predict(standardized_path, timestamps)

    # ------------------------------------------------------------------ #
    # STEP 5: Analytics                                                   #
    # ------------------------------------------------------------------ #
    print("5. Analyzing communication behavior...")
    streamer = AudioStreamer(standardized_path, frame_duration_ms=20)
    total_duration = streamer.get_duration()
    analytics = AnalyticsEngine(timestamps, total_duration)
    stats = analytics.synthesize()

    stats["language"] = language
    stats["language_confidence"] = conf

    # ------------------------------------------------------------------ #
    # STEP 6: Visualizations                                              #
    # ------------------------------------------------------------------ #
    print("6. Generating visualizations...")
    viz = Visualization(timestamps, total_duration)
    viz.generate_all()

    # ------------------------------------------------------------------ #
    # STEP 7: Reports                                                     #
    # ------------------------------------------------------------------ #
    print("7. Generating final reports...")
    report_gen = ReportGenerator(timestamps, stats)
    report_gen.export_csv("outputs/speech_segments.csv")
    report_gen.export_txt("outputs/report.txt")
    report_gen.print_console()

    print("\nProcessing complete! Check the 'outputs' folder.")


if __name__ == "__main__":
    main()
