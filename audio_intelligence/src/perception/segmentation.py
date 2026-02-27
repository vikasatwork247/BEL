"""
segmentation.py — Stage 3: Boundary Refinement
================================================
Takes rough SAD regions from Stage 2 and produces clean, natural segments by:

  1. Merging regions that are very close together (< MERGE_GAP_S seconds)
     — prevents one sentence from appearing as 5 fragmented micro-segments.

  2. Splitting regions that are very long (> SPLIT_THRESH_S seconds) at the
     lowest-energy valley inside the region.
     — avoids one 45-second continuous segment that makes analytics useless.

No external ML model required.  Uses only torchaudio + numpy for RMS energy
analysis.  This replaces pyannote.audio segmentation which requires an HF token.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
import soundfile as sf

# ── Tuning constants ─────────────────────────────────────────────────────── #
MERGE_GAP_S    = 0.40    # merge pair if gap < 400 ms
SPLIT_THRESH_S = 15.0    # split region if longer than 15 s
MIN_SEG_S      = 0.25    # drop any segment shorter than 250 ms after splitting
VALLEY_WIN_MS  = 80      # window (ms) used to compute RMS for valley search
# ──────────────────────────────────────────────────────────────────────────── #


def run_segmentation(
    vocals_wav: str,
    sad_segments: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """
    Refine *sad_segments* against the waveform in *vocals_wav*.

    Parameters
    ----------
    vocals_wav   : path to 16 kHz mono WAV (output of separation.py)
    sad_segments : list of {'start': float_s, 'end': float_s}

    Returns
    -------
    list of {'start': float_s, 'end': float_s}  — refined, sorted segments
    """
    if not sad_segments:
        return []

    print(f"[Segmentation] Refining {len(sad_segments)} SAD region(s) …")

    # Load audio once for valley detection
    audio, sr = _load_mono_wav(vocals_wav)

    # --- Stage A: merge close segments ---
    merged = _merge_close(sad_segments)

    # --- Stage B: split long segments ---
    refined: List[Dict[str, float]] = []
    for seg in merged:
        if (seg["end"] - seg["start"]) > SPLIT_THRESH_S:
            splits = _split_at_valley(audio, sr, seg)
            refined.extend(splits)
        else:
            refined.append(seg)

    # --- Stage C: drop too-short segments ---
    refined = [s for s in refined if (s["end"] - s["start"]) >= MIN_SEG_S]

    # --- Sort by start time ---
    refined.sort(key=lambda s: s["start"])

    print(f"[Segmentation] Produced {len(refined)} refined segment(s).")
    return refined


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _load_mono_wav(path: str):
    """Load WAV as float32 mono numpy array.  Returns (array, sample_rate)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data, sr


def _merge_close(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Merge any two consecutive segments whose gap < MERGE_GAP_S."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s["start"])
    merged = [dict(segs[0])]
    for seg in segs[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        if gap < MERGE_GAP_S:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


def _split_at_valley(
    audio: np.ndarray,
    sr: int,
    seg: Dict[str, float],
) -> List[Dict[str, float]]:
    """
    Split *seg* into two parts at the lowest-RMS valley in its middle 60%.
    Recursively splits if either half is still > SPLIT_THRESH_S.
    """
    start_s, end_s = seg["start"], seg["end"]
    duration = end_s - start_s

    # Only search in the middle 60 % to avoid cutting right at boundaries
    search_start_s = start_s + duration * 0.20
    search_end_s   = end_s   - duration * 0.20

    start_idx  = int(start_s * sr)
    end_idx    = int(end_s   * sr)
    s_idx      = int(search_start_s * sr)
    e_idx      = int(search_end_s   * sr)

    if e_idx <= s_idx or end_idx > len(audio):
        return [seg]   # segment is at edge of file — skip splitting

    window = max(1, int(sr * VALLEY_WIN_MS / 1000))
    rms_values = []
    positions  = []

    for pos in range(s_idx, e_idx - window, window // 2):
        frame = audio[pos: pos + window]
        rms   = float(np.sqrt(np.mean(frame ** 2)))
        rms_values.append(rms)
        positions.append(pos)

    if not rms_values:
        return [seg]

    valley_idx = positions[int(np.argmin(rms_values))]
    valley_s   = valley_idx / sr

    part_a = {"start": start_s,  "end": valley_s}
    part_b = {"start": valley_s, "end": end_s}

    result: List[Dict[str, float]] = []
    for part in (part_a, part_b):
        if (part["end"] - part["start"]) > SPLIT_THRESH_S:
            result.extend(_split_at_valley(audio, sr, part))
        else:
            result.append(part)
    return result
