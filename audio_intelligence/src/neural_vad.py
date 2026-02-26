import torch
import soundfile as sf
import numpy as np


def _spectral_flatness(frame: np.ndarray) -> float:
    """
    Computes spectral flatness of a frame.
    High flatness (close to 1.0) = noise/music. Low flatness = speech/silence.
    """
    if len(frame) == 0:
        return 0.0
    windowed = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed)) + 1e-10
    geometric_mean = np.exp(np.mean(np.log(spectrum)))
    arithmetic_mean = np.mean(spectrum)
    return float(geometric_mean / (arithmetic_mean + 1e-10))


def _zero_crossing_rate(frame: np.ndarray) -> float:
    """
    Computes zero crossing rate. High ZCR = noise/music. Low ZCR = silence. Medium = speech.
    """
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    crossings = np.sum(np.abs(np.diff(signs))) / 2.0
    return float(crossings / len(frame))


def _is_music_or_noise(audio_chunk: np.ndarray, sample_rate: int,
                        flatness_threshold: float = 0.25,
                        analysis_frame_size: int = 2048) -> bool:
    """
    Checks if an audio chunk is predominantly music or broadband noise
    by analyzing spectral flatness across short analysis frames.
    Returns True if the chunk should be suppressed (not speech).
    """
    if len(audio_chunk) < analysis_frame_size:
        return False

    flatness_values = []
    for start in range(0, len(audio_chunk) - analysis_frame_size, analysis_frame_size // 2):
        frame = audio_chunk[start:start + analysis_frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        # Only analyze frames with meaningful energy (skip silence frames)
        if energy > 0.01:
            flatness_values.append(_spectral_flatness(frame))

    if not flatness_values:
        return False

    # If more than 60% of energetic frames are tonally flat → music/noise
    median_flatness = float(np.median(flatness_values))
    return median_flatness > flatness_threshold


def _refine_segment_boundaries(audio: np.ndarray, start_sample: int, end_sample: int,
                                 sample_rate: int, margin_ms: int = 50) -> tuple:
    """
    Refines speech segment start/end using energy envelope:
    - Pulls start slightly earlier and end slightly later to avoid clipping.
    - Trims silence from edges by checking RMS in short windows.
    """
    margin = int(sample_rate * margin_ms / 1000)
    refined_start = max(0, start_sample - margin)
    refined_end = min(len(audio), end_sample + margin)

    # Trim leading silence from refined_start
    window = int(sample_rate * 0.02)  # 20ms windows
    rms_threshold = 0.005

    while refined_start < end_sample:
        chunk = audio[refined_start:refined_start + window]
        if len(chunk) == 0:
            break
        if np.sqrt(np.mean(chunk ** 2)) >= rms_threshold:
            break
        refined_start += window

    # Trim trailing silence from refined_end
    while refined_end > start_sample + window:
        chunk = audio[refined_end - window:refined_end]
        if len(chunk) == 0:
            break
        if np.sqrt(np.mean(chunk ** 2)) >= rms_threshold:
            break
        refined_end -= window

    return refined_start, refined_end


def run_neural_vad(file_path: str):
    """
    Runs Silero VAD on the audio file with:
    - Pre-VAD music/noise gate to suppress non-speech frames
    - Tighter VAD parameters for better precision
    - Post-VAD boundary refinement for accurate timestamps
    - Chunk-based processing (5 min segments) for low RAM usage

    Returns: list of dicts [{"start": start_s, "end": end_s}, ...]
    with accurate timestamps even in noisy/music conditions.
    """
    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    SAMPLING_RATE = 16000
    CHUNK_DURATION_SEC = 300       # 5-minute chunks for low RAM
    CHUNK_SAMPLES = int(SAMPLING_RATE * CHUNK_DURATION_SEC)

    # Analysis window for music/noise gate: 1 second
    NOISE_CHECK_WINDOW = SAMPLING_RATE

    all_speech_timestamps = []
    full_audio = None  # We'll store full audio for boundary refinement

    with sf.SoundFile(file_path, 'r') as f:
        if f.samplerate != SAMPLING_RATE or f.channels != 1:
            raise ValueError(
                f"Expected 16000 Hz mono, got {f.samplerate} Hz {f.channels} ch. "
                "Run convert_audio() first."
            )

        audio_frames = []
        current_sample_offset = 0

        while True:
            audio_chunk = f.read(frames=CHUNK_SAMPLES, dtype='float32')
            if len(audio_chunk) == 0:
                break

            audio_frames.append(audio_chunk.copy())

            # --- Pre-VAD music/noise suppression ---
            # Build a masked version: zero out 1s sub-windows that are music/noise
            masked_chunk = audio_chunk.copy()
            for sub_start in range(0, len(audio_chunk) - NOISE_CHECK_WINDOW, NOISE_CHECK_WINDOW):
                sub_end = sub_start + NOISE_CHECK_WINDOW
                sub_frame = audio_chunk[sub_start:sub_end]
                if _is_music_or_noise(sub_frame, SAMPLING_RATE):
                    # Suppress this sub-window (zero it out before VAD sees it)
                    masked_chunk[sub_start:sub_end] = 0.0

            chunk_tensor = torch.from_numpy(masked_chunk)

            # --- Run Silero VAD with tighter parameters ---
            speech_timestamps = get_speech_timestamps(
                chunk_tensor,
                model,
                sampling_rate=SAMPLING_RATE,
                threshold=0.6,                  # Higher confidence required (was default ~0.5)
                min_speech_duration_ms=600,     # Reject very short blips (was 400ms)
                min_silence_duration_ms=400,    # Allow natural pause gaps (was 250ms)
                speech_pad_ms=30,               # Cushion around speech edges
            )

            for ts in speech_timestamps:
                global_start_samples = current_sample_offset + ts['start']
                global_end_samples = current_sample_offset + ts['end']
                start_s = global_start_samples / SAMPLING_RATE
                end_s = global_end_samples / SAMPLING_RATE
                all_speech_timestamps.append({
                    "start": start_s,
                    "end": end_s,
                    "_start_sample": int(global_start_samples),
                    "_end_sample": int(global_end_samples),
                })

            current_sample_offset += len(audio_chunk)

    # --- Merge segments across chunk boundaries ---
    merged = []
    MERGE_GAP = 0.5   # Merge gaps ≤ 0.5s (tighter than before)
    for ts in all_speech_timestamps:
        if not merged:
            merged.append(ts)
            continue
        last = merged[-1]
        gap = ts["start"] - last["end"]
        if gap <= MERGE_GAP:
            last["end"] = ts["end"]
            last["_end_sample"] = ts["_end_sample"]
        else:
            merged.append(ts)

    # --- Post-VAD boundary refinement using energy envelope ---
    if audio_frames:
        full_audio = np.concatenate(audio_frames)
        refined = []
        for ts in merged:
            rs, re = _refine_segment_boundaries(
                full_audio,
                ts["_start_sample"],
                ts["_end_sample"],
                SAMPLING_RATE,
                margin_ms=50
            )
            start_s = rs / SAMPLING_RATE
            end_s = re / SAMPLING_RATE
            # Only keep if still meaningfully long after refinement
            if end_s - start_s >= 0.2:
                refined.append({"start": round(start_s, 3), "end": round(end_s, 3)})
        return refined

    # Fallback: return merged without refinement
    return [{"start": round(t["start"], 3), "end": round(t["end"], 3)} for t in merged]
