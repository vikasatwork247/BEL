"""
test_sad.py — Quick test for the SpeechActivityDetector
========================================================
Usage:
    python test_sad.py                        # uses test.wav in current dir
    python test_sad.py path/to/my_audio.wav   # custom file
"""

import sys
import os

# ── Make sure we can find sad_detector.py even if run from a sub-folder ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sad_detector import SpeechActivityDetector


def main():
    # Accept optional CLI argument for audio file path
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

    if not os.path.exists(audio_file):
        print(f"\n[ERROR] Audio file not found: '{audio_file}'")
        print("Usage: python test_sad.py <path_to_audio_file>")
        print("Example: python test_sad.py inputs/sample.wav")
        sys.exit(1)

    print(f"\n[Test] Running SAD on: {audio_file}")
    print("-" * 50)

    sad = SpeechActivityDetector()
    segments = sad.get_speech_segments(audio_file)

    print("\nSpeech detected at:")
    print("-" * 50)

    if not segments:
        print("  ⚠  No speech segments detected.")
        print("     Try a different audio file or check that the file has audible speech.")
    else:
        for i, seg in enumerate(segments, 1):
            duration_ms = seg["end"] - seg["start"]
            print(
                f"  Segment {i:>2}: "
                f"start={seg['start']:>6} ms  →  end={seg['end']:>6} ms  "
                f"(duration: {duration_ms} ms  /  {duration_ms/1000:.2f}s)"
            )

    print("-" * 50)
    print(f"Total segments: {len(segments)}")
    total_speech_ms = sum(s["end"] - s["start"] for s in segments)
    print(f"Total speech  : {total_speech_ms} ms  /  {total_speech_ms/1000:.2f}s")
    print("\n✅  SAD test completed successfully!")


if __name__ == "__main__":
    main()
