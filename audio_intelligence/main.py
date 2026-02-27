import os
import sys
import argparse

# --------------------------------------------------------------------------- #
# Perception pipeline (replaces VAD as the source of speech timestamps)       #
# --------------------------------------------------------------------------- #
from src.perception.pipeline import run_pipeline, load_segments_json

# --------------------------------------------------------------------------- #
# Analytics modules (unchanged — receive better timestamps from perception)   #
# --------------------------------------------------------------------------- #
from src.audio_streamer import AudioStreamer
from src.language_classifier import LanguageClassifier
from src.analytics_engine import AnalyticsEngine
from src.visualization import Visualization
from src.report_generator import ReportGenerator
from src.feature_extractor import compute_mfcc


def main():
    parser = argparse.ArgumentParser(
        description="Audio Intelligence & Human Communication Analytics System"
    )
    parser.add_argument("input_file", nargs="?", help="Path to input audio file")
    parser.add_argument("--input", dest="input_flag", help="Path to input audio file")
    args = parser.parse_args()

    input_path = args.input_flag or args.input_file
    if not input_path:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "graphs"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "temp"), exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEPS 1-4 — 4-Stage Perception Pipeline                            #
    #                                                                     #
    # Stage 1: Demucs vocal separation  → outputs/vocals.wav             #
    # Stage 2: SpeechBrain neural SAD   → rough speech regions           #
    # Stage 3: Boundary refinement      → clean, merged segments         #
    # Stage 4: WhisperX forced align    → ms-accurate timestamps         #
    #                                    → outputs/segments.json         #
    #                                                                     #
    # vad_engine.py / neural_vad.py / feature_extractor.py are NO LONGER #
    # used to determine timestamps. They remain available for analytics. #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  PERCEPTION PIPELINE — 4-Stage Speech Boundary Detection")
    print("=" * 60)

    run_pipeline(input_path)

    # Load standardised seconds-based timestamps for analytics
    timestamps = load_segments_json()       # [{'start': s, 'end': s}, ...]

    n_segments = len(timestamps)
    print(f"\nSpeech segments detected: {n_segments}")

    if not timestamps:
        print("WARNING: No speech segments detected. "
              "Check the audio content and try again.")

    # ------------------------------------------------------------------ #
    # STEP 5 — Feature extraction & language detection                   #
    # (operates on outputs/vocals.wav — already 16 kHz mono)             #
    # ------------------------------------------------------------------ #
    vocals_path = os.path.join("outputs", "vocals.wav")
    analysis_path = vocals_path if os.path.exists(vocals_path) else input_path

    print("\n" + "=" * 60)
    print("  ANALYTICS  — Language, Features, Behaviour")
    print("=" * 60)

    print("5. Extracting features and detecting language …")
    features = compute_mfcc(analysis_path, timestamps)
    clf = LanguageClassifier()
    language, conf = clf.predict(analysis_path, timestamps)

    # ------------------------------------------------------------------ #
    # STEP 6 — Analytics engine                                           #
    # ------------------------------------------------------------------ #
    print("6. Analysing communication behaviour …")
    streamer = AudioStreamer(analysis_path, frame_duration_ms=20)
    total_duration = streamer.get_duration()
    analytics = AnalyticsEngine(timestamps, total_duration)
    stats = analytics.synthesize()

    stats["language"]            = language
    stats["language_confidence"] = conf

    # ------------------------------------------------------------------ #
    # STEP 7 — Visualisations                                             #
    # ------------------------------------------------------------------ #
    print("7. Generating visualisations …")
    viz = Visualization(timestamps, total_duration)
    viz.generate_all()

    # ------------------------------------------------------------------ #
    # STEP 8 — Reports                                                    #
    # ------------------------------------------------------------------ #
    print("8. Generating final reports …")
    report_gen = ReportGenerator(timestamps, stats)
    report_gen.export_csv("outputs/speech_segments.csv")
    report_gen.export_txt("outputs/report.txt")
    report_gen.print_console()

    print("\nProcessing complete! Check the 'outputs' folder.")


if __name__ == "__main__":
    main()
