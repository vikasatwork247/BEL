import os
import subprocess
import tempfile
import uuid
import soundfile as sf
import numpy as np

def convert_audio(input_path: str) -> str:
    """
    Converts input audio to 16kHz, mono, 16-bit PCM WAV using FFmpeg.
    Returns the path to the temporary converted file.
    """
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        temp_path
    ]
    
    try:
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please ensure FFmpeg is installed and added to PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}")
        
    return temp_path


def standardize_audio(input_path: str) -> str:
    """
    Standardizes audio to 16kHz, mono, 16-bit PCM WAV with normalized loudness.
    Uses pydub for resampling and normalization, with ffmpeg as backend.
    Output: outputs/temp/standardized.wav

    WHY: Neural VAD models (Silero) are ONLY trained on 16kHz mono PCM speech.
    Feeding 44.1kHz stereo MP3 means every frame has wrong frequency content
    and the model never reaches its speech probability threshold.
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
    except ImportError:
        raise RuntimeError(
            "pydub is not installed. Run: pip install pydub"
        )

    os.makedirs(os.path.join("outputs", "temp"), exist_ok=True)
    out_path = os.path.join("outputs", "temp", "standardized.wav")

    ext = os.path.splitext(input_path)[1].lower().lstrip(".")
    if not ext:
        ext = "wav"

    print(f"[Audio] Loading {input_path} as '{ext}'...")
    try:
        audio = AudioSegment.from_file(input_path, format=ext)
    except Exception as e:
        raise RuntimeError(f"pydub failed to load '{input_path}': {e}")

    # Log original properties
    print(f"[Audio] original sample rate : {audio.frame_rate} Hz")
    print(f"[Audio] channels             : {audio.channels}")
    print(f"[Audio] duration             : {len(audio) / 1000.0:.2f}s")

    # Step 1: Convert stereo -> mono
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Step 2: Resample to 16000 Hz
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    # Step 3: Ensure 16-bit depth
    audio = audio.set_sample_width(2)  # 2 bytes = 16-bit

    # Step 4: Normalize loudness to ~-20 dBFS
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)

    # Export
    audio.export(out_path, format="wav")

    print(f"[After Standardization] sample rate : 16000 Hz")
    print(f"[After Standardization] channels    : 1 (mono)")
    print(f"[After Standardization] saved to    : {out_path}")

    return out_path
