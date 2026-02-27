"""
separation.py — Stage 1: Source Separation via Demucs
=======================================================
Isolates the vocals stem from any audio using Facebook Demucs (htdemucs).
Output is always written to outputs/vocals.wav as 16 kHz mono float32 WAV
so that downstream ML models receive a validated, standardised signal.

WHY: Neural VAD/SAD/alignment models are trained on clean speech.
Instruments and music mask vocal frequencies → models either produce zero
segments or misalign boundaries by hundreds of milliseconds.
By stripping accompaniment first, every downstream stage gets a clean signal.
"""

from __future__ import annotations

import os
import shutil


# --------------------------------------------------------------------------- #
# Public entry-point                                                           #
# --------------------------------------------------------------------------- #

def run_separation(input_path: str) -> str:
    """
    Run Demucs vocal separation on *input_path*.

    Always returns a 16 kHz mono WAV path.  If the input is already a clean
    speech recording (no music), separation is still beneficial — it costs
    ~1-2 min on CPU but guarantees every downstream model works on the same
    standardised format.

    Parameters
    ----------
    input_path : str
        Any audio file (wav, mp3, m4a, flac, ogg …)

    Returns
    -------
    str
        Absolute path to ``outputs/vocals.wav`` (16 kHz mono float32)
    """
    _ensure_dirs()

    try:
        import demucs.separate  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "[Separation] demucs is not installed. Run: pip install demucs"
        )

    out_dir = os.path.join("outputs", "temp", "demucs_out")
    os.makedirs(out_dir, exist_ok=True)

    model_name = "htdemucs"   # 4-stem Hybrid Transformer Demucs (best quality)

    print(f"[Separation] Input           : {input_path}")
    print(f"[Separation] Model           : {model_name}")
    print(f"[Separation] Running Demucs  (CPU — may take 1-3 min) …")

    args = [
        "--two-stems", "vocals",   # only produce vocals + no_vocals (2× faster)
        "-n", model_name,
        "-o", out_dir,
        input_path,
    ]

    try:
        demucs.separate.main(args)
    except SystemExit:
        pass  # demucs calls sys.exit(0) on clean finish
    except Exception as exc:
        raise RuntimeError(f"[Separation] Demucs failed: {exc}")

    vocals_raw = _find_vocals(out_dir, input_path)
    vocals_wav = _standardise_to_wav(vocals_raw)

    print(f"[Separation] Vocals saved to : {vocals_wav}")
    return vocals_wav


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _ensure_dirs() -> None:
    os.makedirs(os.path.join("outputs", "temp"), exist_ok=True)
    os.makedirs(os.path.join("outputs"), exist_ok=True)


def _find_vocals(out_dir: str, input_path: str) -> str:
    """Locate the vocals file produced by Demucs (handles mp3/wav extensions)."""
    track_name = os.path.splitext(os.path.basename(input_path))[0]

    candidates = [
        os.path.join(out_dir, "htdemucs", track_name, "vocals.wav"),
        os.path.join(out_dir, "htdemucs", track_name, "vocals.mp3"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Fallback: walk the tree
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            if "vocals" in fname.lower():
                return os.path.join(root, fname)

    raise RuntimeError(
        f"[Separation] Demucs finished but no vocals file found in {out_dir}. "
        "Check demucs output manually."
    )


def _standardise_to_wav(vocals_path: str) -> str:
    """
    Convert *vocals_path* to 16 kHz mono float32 WAV → outputs/vocals.wav.
    Uses pydub (ffmpeg backend) for robust format handling.
    """
    dest = os.path.join("outputs", "vocals.wav")

    try:
        from pydub import AudioSegment
        from pydub.effects import normalize

        ext = os.path.splitext(vocals_path)[1].lower().lstrip(".")
        audio = AudioSegment.from_file(vocals_path, format=ext or "wav")

        # mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # 16 kHz
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        # 16-bit
        audio = audio.set_sample_width(2)
        # Normalise loudness
        target_dBFS = -20.0
        audio = audio.apply_gain(target_dBFS - audio.dBFS)

        audio.export(dest, format="wav")
        return dest

    except Exception as exc:
        # Last-resort: plain copy (may not be 16 kHz but worth trying)
        print(f"[Separation] pydub standardisation failed ({exc}); copying raw file.")
        shutil.copy2(vocals_path, dest)
        return dest
