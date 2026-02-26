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


def run_neural_vad(file_path: str) -> List[Tuple[float, float]]:
    """
    Runs Silero VAD on a PRE-STANDARDIZED (16kHz mono PCM) audio file.
    Returns a list of (start_seconds, end_seconds) speech segment tuples.

    WHY threshold=0.35: Singing voice has different spectral characteristics
    than clean conversational speech. Silero's default threshold (~0.5) is
    calibrated for clean microphone speech and rejects the more tonal,
    harmonically-rich signal of a singing voice. Lowering to 0.35 allows
    the model to detect sustained vocal activity while still rejecting
    pure instrumental passages.
    """
    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    SAMPLING_RATE = 16000
    CHUNK_DURATION_SEC = 300       # 5-minute chunks to keep RAM low
    CHUNK_SAMPLES = int(SAMPLING_RATE * CHUNK_DURATION_SEC)

    all_speech_timestamps: List[dict] = []
    total_frames_analyzed = 0
    total_speech_frames   = 0

    with sf.SoundFile(file_path, 'r') as f:
        # Mandatory: verify the file is already standardized
        if f.samplerate != SAMPLING_RATE:
            raise ValueError(
                f"[VAD] Expected 16000 Hz input, got {f.samplerate} Hz. "
                "Run standardize_audio() before calling run_neural_vad()."
            )
        if f.channels != 1:
            raise ValueError(
                f"[VAD] Expected mono input, got {f.channels} channels. "
                "Run standardize_audio() before calling run_neural_vad()."
            )

        current_sample_offset = 0

        while True:
            audio_chunk = f.read(frames=CHUNK_SAMPLES, dtype='float32')
            if len(audio_chunk) == 0:
                break

            chunk_tensor = torch.from_numpy(audio_chunk)

            # --- Run Silero VAD ---
            # threshold=0.35: lower than default to catch singing + noisy speech.
            # min_speech_duration_ms=250: short sung syllables are valid speech.
            # min_silence_duration_ms=300: natural gaps between phrases.
            speech_timestamps = get_speech_timestamps(
                chunk_tensor,
                model,
                sampling_rate=SAMPLING_RATE,
                threshold=0.35,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,
                speech_pad_ms=30,
            )

            # Accumulate frame counts for debug logging
            frame_samples = int(SAMPLING_RATE * 0.025)   # 25 ms frames
            hop_samples   = int(SAMPLING_RATE * 0.010)   # 10 ms hop
            chunk_frames  = max(0, (len(audio_chunk) - frame_samples) // hop_samples + 1)
            total_frames_analyzed += chunk_frames

            for ts in speech_timestamps:
                seg_frames = max(0, (ts['end'] - ts['start'] - frame_samples) // hop_samples + 1)
                total_speech_frames += int(seg_frames)

                global_start_s = (current_sample_offset + ts['start']) / SAMPLING_RATE
                global_end_s   = (current_sample_offset + ts['end'])   / SAMPLING_RATE
                all_speech_timestamps.append({
                    "start": global_start_s,
                    "end":   global_end_s,
                })

            current_sample_offset += len(audio_chunk)

    # --- Merge segments across chunk boundaries ---
    merged: List[dict] = []
    MERGE_GAP = 0.5
    for ts in all_speech_timestamps:
        if not merged:
            merged.append(ts)
            continue
        last = merged[-1]
        gap = ts["start"] - last["end"]
        if gap <= MERGE_GAP:
            last["end"] = ts["end"]
        else:
            merged.append(ts)

    # --- Debug logging ---
    print(f"[VAD] total frames analyzed  : {total_frames_analyzed}")
    print(f"[VAD] speech frames detected : {total_speech_frames}")
    print(f"[VAD] number of segments     : {len(merged)}")

    return [{"start": round(t["start"], 3), "end": round(t["end"], 3)} for t in merged]
