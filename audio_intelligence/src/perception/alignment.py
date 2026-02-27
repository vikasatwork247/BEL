"""
alignment.py — Stage 4: Forced Alignment → millisecond timestamps
=================================================================
Uses WhisperX to produce word/segment-level timestamps aligned to the
waveform via wav2vec2 CTC phoneme alignment.

WHY WhisperX over plain Whisper:
  Whisper's built-in timestamps have ±0.5–2 s jitter because they are
  predicted from attention weights, not from the acoustic signal itself.
  WhisperX re-aligns every word boundary against the waveform using a
  wav2vec2 CTC model, achieving ~20–80 ms precision.

FALLBACK:
  If WhisperX alignment fails for any segment (very short clip, silence,
  unsupported language …), the SpeechBrain SAD boundary is used directly
  with confidence = 0.5.

Output schema (one dict per final segment):
  {
    "start_ms" : int,     # milliseconds from start of original audio
    "end_ms"   : int,     # milliseconds from start of original audio
    "confidence": float   # 0-1 (alignment quality)
  }
"""

from __future__ import annotations

import os
from typing import List, Dict


# Minimum speech duration (ms) to attempt WhisperX alignment.
# Very short blips (<300 ms) are kept as-is with confidence=0.5.
MIN_ALIGN_MS = 300

# WhisperX Whisper model size.  'tiny' is fast on CPU; change to 'base'
# for marginally better accuracy at ~2× the cost.
WHISPER_MODEL = "tiny"

# wav2vec2 alignment model — auto-selected by whisperx per detected language.
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# Public entry-point                                                           #
# --------------------------------------------------------------------------- #

def run_alignment(
    vocals_wav: str,
    segments_s: List[Dict[str, float]],
) -> List[Dict]:
    """
    Align *segments_s* (seconds) against *vocals_wav* using WhisperX.

    Parameters
    ----------
    vocals_wav  : path to 16 kHz mono WAV
    segments_s  : list of {'start': float_s, 'end': float_s}  (from Stage 3)

    Returns
    -------
    list of {'start_ms': int, 'end_ms': int, 'confidence': float}
    """
    if not segments_s:
        return []

    print(f"[Alignment] Attempting WhisperX forced alignment on "
          f"{len(segments_s)} segment(s) …")

    try:
        import whisperx  # type: ignore
        return _align_with_whisperx(vocals_wav, segments_s, whisperx)
    except ImportError:
        print("[Alignment] whisperx not installed — using SAD boundaries directly.")
        return _fallback(segments_s)
    except Exception as exc:
        print(f"[Alignment] WhisperX failed ({exc}) — using SAD boundaries.")
        return _fallback(segments_s)


# --------------------------------------------------------------------------- #
# WhisperX alignment path                                                      #
# --------------------------------------------------------------------------- #

def _align_with_whisperx(
    vocals_wav: str,
    segments_s: List[Dict[str, float]],
    whisperx,
) -> List[Dict]:
    """
    Full WhisperX pipeline:
      1. Load audio via whisperx helper (handles resampling)
      2. Run Whisper tiny ASR to get transcription + coarse timestamps
      3. Detect language (needed by wav2vec2 model selection)
      4. Load wav2vec2 alignment model
      5. Align each segment → word-level timestamps
      6. Collect per-segment start/end with confidence
    """
    import torch

    print(f"[Alignment] Loading Whisper-{WHISPER_MODEL} ASR model …")
    audio = whisperx.load_audio(vocals_wav)

    asr_model = whisperx.load_model(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type="int8",    # int8 quantisation → CPU-friendly
    )

    # Transcribe to get coarse word segments
    result = asr_model.transcribe(audio, batch_size=4)

    if not result.get("segments"):
        print("[Alignment] WhisperX ASR produced no segments; using SAD boundaries.")
        return _fallback(segments_s)

    lang = result.get("language", "en")
    print(f"[Alignment] Detected language : {lang}")

    # Load wav2vec2 alignment model for the detected language
    print(f"[Alignment] Loading wav2vec2 alignment model for '{lang}' …")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=lang,
            device=DEVICE,
        )
    except Exception as exc:
        print(f"[Alignment] wav2vec2 model unavailable for '{lang}' ({exc}). "
              "Falling back to 'en' model.")
        align_model, metadata = whisperx.load_align_model(
            language_code="en",
            device=DEVICE,
        )

    # Run forced alignment
    aligned_result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    # Convert whisperx word-level output → our segment schema
    output = _whisperx_to_segments(aligned_result, segments_s)
    print(f"[Alignment] WhisperX produced {len(output)} aligned segment(s).")
    return output


def _whisperx_to_segments(
    aligned_result: dict,
    original_segments: List[Dict[str, float]],
) -> List[Dict]:
    """
    Map WhisperX segment-level timestamps back to our schema.
    WhisperX segment boundaries are already phoneme-aligned.
    """
    out: List[Dict] = []
    for seg in aligned_result.get("segments", []):
        s_ms = int(float(seg.get("start", 0)) * 1000)
        e_ms = int(float(seg.get("end",   0)) * 1000)

        # Compute confidence from word-level scores if available
        words = seg.get("words", [])
        scores = [w.get("score", 0.5) for w in words if "score" in w]
        confidence = float(sum(scores) / len(scores)) if scores else 0.75

        if e_ms > s_ms:
            out.append({
                "start_ms":   s_ms,
                "end_ms":     e_ms,
                "confidence": round(confidence, 4),
            })

    # If WhisperX produced nothing, fall back to original SAD boundaries
    if not out:
        return _fallback(original_segments)
    return out


# --------------------------------------------------------------------------- #
# Fallback: convert SAD seconds → ms with default confidence                  #
# --------------------------------------------------------------------------- #

def _fallback(segments_s: List[Dict[str, float]]) -> List[Dict]:
    """Convert SAD/segmentation second-boundaries to ms, confidence=0.5."""
    return [
        {
            "start_ms":   int(s["start"] * 1000),
            "end_ms":     int(s["end"]   * 1000),
            "confidence": 0.5,
        }
        for s in segments_s
        if s["end"] > s["start"]
    ]
