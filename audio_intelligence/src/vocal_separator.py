"""
vocal_separator.py

Separates vocals from music using Facebook Demucs.
WHY: Neural VAD models are trained on clean speech. When instruments play
alongside a singer, harmonic frequencies mask the speech fundamentals and
every VAD frame probability stays below threshold → zero segments detected.
By isolating only the vocals stem first, the VAD gets clean solo-voice audio
and correctly identifies speech activity.

Pipeline position: input → separate_vocals() → standardize_audio() → VAD
"""

import os
import shutil


def separate_vocals(input_path: str) -> str:
    """
    Runs Demucs (htdemucs model) on the input audio and extracts the vocals stem.

    Parameters
    ----------
    input_path : str
        Path to the input audio file (mp3, m4a, wav, flac, etc.)

    Returns
    -------
    str
        Path to the extracted vocals WAV: outputs/temp/vocals.wav
    """
    try:
        import torch
        import demucs.separate
    except ImportError:
        raise RuntimeError(
            "demucs is not installed. Run: pip install demucs"
        )

    os.makedirs(os.path.join("outputs", "temp"), exist_ok=True)

    # Demucs writes its output under a structured folder:
    # <out_dir>/<model_name>/<track_name>/<stem>.wav
    # We point it to outputs/temp as the base, then find vocals.wav.
    out_dir = os.path.join("outputs", "temp", "demucs_out")
    os.makedirs(out_dir, exist_ok=True)

    model_name = "htdemucs"  # best quality 4-stem model (vocals/drums/bass/other)

    print(f"[VocalSep] Separating vocals from: {input_path}")
    print(f"[VocalSep] Using model            : {model_name}")
    print(f"[VocalSep] This may take a minute on CPU...")

    # Run demucs programmatically
    # demucs.separate.main() accepts a list of arguments identical to CLI
    args = [
        "--two-stems", "vocals",      # only output vocals + accompaniment (faster)
        "-n", model_name,
        "-o", out_dir,
        "--mp3",                       # compress intermediates to save disk
        input_path,
    ]

    try:
        demucs.separate.main(args)
    except SystemExit:
        pass  # demucs calls sys.exit(0) on success — catch it
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}")

    # Locate the generated vocals file
    # Structure: out_dir/htdemucs/<track_stem>/vocals.mp3 or vocals.wav
    track_name = os.path.splitext(os.path.basename(input_path))[0]
    vocals_candidates = [
        os.path.join(out_dir, model_name, track_name, "vocals.mp3"),
        os.path.join(out_dir, model_name, track_name, "vocals.wav"),
    ]

    vocals_src = None
    for candidate in vocals_candidates:
        if os.path.exists(candidate):
            vocals_src = candidate
            break

    if vocals_src is None:
        # Fallback: walk and find any vocals file
        for root, dirs, files in os.walk(out_dir):
            for fname in files:
                if "vocals" in fname.lower():
                    vocals_src = os.path.join(root, fname)
                    break
            if vocals_src:
                break

    if vocals_src is None:
        raise RuntimeError(
            f"Demucs finished but could not find the vocals output in {out_dir}. "
            "Check demucs output directory manually."
        )

    # Copy to a stable output path
    dest = os.path.join("outputs", "temp", "vocals" + os.path.splitext(vocals_src)[1])
    shutil.copy2(vocals_src, dest)

    print(f"[VocalSep] Vocals extracted to: {dest}")
    return dest


def needs_vocal_separation(input_path: str) -> bool:
    """
    Heuristic: compressed or inherently stereo formats likely contain music
    and should go through vocal separation before VAD.
    """
    ext = os.path.splitext(input_path)[1].lower()
    return ext in {".mp3", ".m4a", ".aac", ".ogg", ".opus"}
