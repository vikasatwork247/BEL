import os
import subprocess
import tempfile
import uuid

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
