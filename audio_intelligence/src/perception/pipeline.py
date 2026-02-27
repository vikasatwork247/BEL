"""
pipeline.py — Perception Pipeline Orchestrator
================================================
Runs the 4-stage speech-perception pipeline in strict order:

  Stage 1 — separation.py   : Demucs vocal isolation  → outputs/vocals.wav
  Stage 2 — sad.py          : SpeechBrain neural SAD   → rough regions (seconds)
  Stage 3 — segmentation.py : Boundary refinement      → clean regions (seconds)
  Stage 4 — alignment.py    : WhisperX forced align    → ms timestamps + score

Final output: outputs/segments.json
  {
    "segments": [
      {"start_ms": int, "end_ms": int, "confidence": float},
      ...
    ]
  }

The function also returns the segment list so main.py can pass it to the
existing analytics pipeline (converting ms → seconds where needed).
"""

from __future__ import annotations

import json
import os
from typing import List, Dict


SEGMENTS_JSON = os.path.join("outputs", "segments.json")


# --------------------------------------------------------------------------- #
# Public entry-point                                                           #
# --------------------------------------------------------------------------- #

def run_pipeline(input_audio: str) -> List[Dict]:
    """
    Execute all 4 perception stages on *input_audio*.

    Parameters
    ----------
    input_audio : str
        Path to any audio file (wav, mp3, m4a, flac …)

    Returns
    -------
    list of {'start_ms': int, 'end_ms': int, 'confidence': float}
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "temp"), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 1 — Demucs vocal isolation                                    #
    # ------------------------------------------------------------------ #
    print("\n── Stage 1 / 4 : Source Separation (Demucs) ─────────────────")
    from src.perception.separation import run_separation
    vocals_wav = run_separation(input_audio)

    # ------------------------------------------------------------------ #
    # Stage 2 — Neural Speech Activity Detection                          #
    # ------------------------------------------------------------------ #
    print("\n── Stage 2 / 4 : Neural SAD (SpeechBrain / Silero) ──────────")
    from src.perception.sad import run_sad
    sad_segments = run_sad(vocals_wav)

    if not sad_segments:
        print("[Pipeline] WARNING: SAD found no speech. "
              "Check audio content or lower VAD threshold.")
        _write_json([])
        return []

    # ------------------------------------------------------------------ #
    # Stage 3 — Segmentation refinement                                   #
    # ------------------------------------------------------------------ #
    print("\n── Stage 3 / 4 : Segmentation Refinement ─────────────────────")
    from src.perception.segmentation import run_segmentation
    refined_segments = run_segmentation(vocals_wav, sad_segments)

    if not refined_segments:
        print("[Pipeline] WARNING: Segmentation produced no segments.")
        _write_json([])
        return []

    # ------------------------------------------------------------------ #
    # Stage 4 — WhisperX forced alignment                                 #
    # ------------------------------------------------------------------ #
    print("\n── Stage 4 / 4 : Forced Alignment (WhisperX) ────────────────")
    from src.perception.alignment import run_alignment
    final_segments = run_alignment(vocals_wav, refined_segments)

    # ------------------------------------------------------------------ #
    # Write outputs/segments.json                                         #
    # ------------------------------------------------------------------ #
    _write_json(final_segments)

    n = len(final_segments)
    print(f"\n✅  Speech segments detected: {n}")
    print(f"    Saved to: {SEGMENTS_JSON}")

    return final_segments


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _write_json(segments: List[Dict]) -> None:
    """Serialise *segments* to outputs/segments.json."""
    payload = {"segments": segments}
    with open(SEGMENTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_segments_json() -> List[Dict]:
    """
    Read outputs/segments.json and return the segment list.
    Converts {'start_ms', 'end_ms'} → {'start': float_s, 'end': float_s}
    so the existing analytics_engine / report_generator can consume them
    without modification.
    """
    if not os.path.exists(SEGMENTS_JSON):
        raise FileNotFoundError(
            f"segments.json not found at '{SEGMENTS_JSON}'. "
            "Run run_pipeline() first."
        )
    with open(SEGMENTS_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return [
        {
            "start": seg["start_ms"] / 1000.0,
            "end":   seg["end_ms"]   / 1000.0,
        }
        for seg in data.get("segments", [])
    ]
