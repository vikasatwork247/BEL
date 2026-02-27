"""
sad.py — Stage 2: Neural Speech Activity Detection
====================================================
Primary  : SpeechBrain VAD-CRDNN (speechbrain/vad-crdnn-libriparty)
Fallback : Silero VAD (already cached from previous pipeline)

SpeechBrain VAD-CRDNN is a BLSTM-based model trained on LibriParty —
a corpus of overlapping speech, music, noises, and reverb.  It produces
frame-level speech probabilities that are much more robust to music
backgrounds than energy/spectral-flatness heuristics.

Output: list of {'start': float_sec, 'end': float_sec}
"""

from __future__ import annotations

import os
from typing import List, Dict


SAMPLE_RATE = 16_000
# Path where SpeechBrain will cache the pretrained model
_SB_SAVEDIR = os.path.join(
    os.path.expanduser("~"), ".cache", "speechbrain", "vad-crdnn-libriparty"
)


# --------------------------------------------------------------------------- #
# Public entry-point                                                           #
# --------------------------------------------------------------------------- #

def run_sad(vocals_wav: str) -> List[Dict[str, float]]:
    """
    Detect speech regions in *vocals_wav* (must be 16 kHz mono WAV).

    Returns
    -------
    list of {'start': float, 'end': float}   (seconds)
    """
    print("[SAD] Starting Speech Activity Detection …")

    # Try SpeechBrain first
    segments = _run_speechbrain(vocals_wav)
    if segments is not None:
        print(f"[SAD] SpeechBrain found {len(segments)} raw speech region(s).")
        return segments

    # Fallback: Silero VAD (already installed)
    print("[SAD] Falling back to Silero VAD …")
    segments = _run_silero(vocals_wav)
    print(f"[SAD] Silero found {len(segments)} raw speech region(s).")
    return segments


# --------------------------------------------------------------------------- #
# SpeechBrain backend                                                          #
# --------------------------------------------------------------------------- #

def _run_speechbrain(vocals_wav: str):
    """
    Returns list[dict] on success, None on any failure (triggers fallback).
    """
    try:
        from speechbrain.pretrained import VAD  # type: ignore

        os.makedirs(_SB_SAVEDIR, exist_ok=True)

        print("[SAD] Loading SpeechBrain VAD-CRDNN (downloads ~200 MB on first run) …")
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir=_SB_SAVEDIR,
            run_opts={"device": "cpu"},
        )

        # Compute frame-level probabilities
        prob_chunks = vad.get_speech_prob_file(vocals_wav)

        # Apply double-thresholding
        prob_th = vad.apply_threshold(
            prob_chunks,
            activation_th=0.5,
            deactivation_th=0.25,
        )

        # Get boundaries in seconds
        boundaries = vad.get_boundaries(prob_th, output_value="seconds")

        segments = []
        for row in boundaries:
            start_s = float(row[0])
            end_s   = float(row[1])
            if end_s > start_s:
                segments.append({"start": start_s, "end": end_s})

        return segments

    except Exception as exc:
        print(f"[SAD] SpeechBrain failed ({exc}). Switching to Silero fallback.")
        return None


# --------------------------------------------------------------------------- #
# Silero VAD fallback                                                          #
# --------------------------------------------------------------------------- #

def _run_silero(vocals_wav: str) -> List[Dict[str, float]]:
    """
    Uses Silero VAD as a reliable offline fallback.
    Same model that was already used by neural_vad.py.
    """
    import torch
    import soundfile as sf

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps, *_ = utils

    with sf.SoundFile(vocals_wav, "r") as f:
        sr = f.samplerate
        audio_np = f.read(dtype="float32")

    import numpy as np  # noqa: F401
    chunk_tensor = torch.from_numpy(audio_np)

    raw = get_speech_timestamps(
        chunk_tensor,
        model,
        sampling_rate=sr,
        threshold=0.35,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=30,
    )

    return [
        {
            "start": ts["start"] / sr,
            "end":   ts["end"]   / sr,
        }
        for ts in raw
    ]
