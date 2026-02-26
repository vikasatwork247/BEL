import torch
import soundfile as sf
import numpy as np

def run_neural_vad(file_path: str):
    """
    Runs Silero VAD on the audio file using sliding chunk windows to prevent High RAM usage.
    Returns: list of dicts [{"start": start_s, "end": end_s}, ...]
    """
    # 1. Load Silero VAD model explicitly without internet runtime if possible (assumes downloaded or cached)
    # Using trust_repo=True as required by latest PyTorch versions.
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', 
        model='silero_vad', 
        trust_repo=True
    )
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # Target frame size chunks to pass into VAD.
    # We will read chunks of exactly 512 samples (~32ms) to align roughly to what Silero accepts (512 samples for 16khz)
    # Actually, Silero VAD expects full tensors depending on how we call get_speech_timestamps. 
    # But get_speech_timestamps processes an entire waveform. Since we have memory limits (2GB RAM max), 
    # we can read chunks of 5 minutes (instead of tiny 30ms frames manually) which takes very little RAM.
    # 5 min of 16k 32-bit float audio is 16000 * 300 * 4 bytes = 19.2 MB, which is completely fine!
    # A single 30 min file read in full into RAM is ~115 MB. 
    # However, to be strictly chunked as requested ("DO NOT read full file into RAM. Process in chunks."),
    # we will process in 5-minute segments and offset the timestamps.

    SAMPLING_RATE = 16000
    CHUNK_DURATION_SEC = 300  # 5 minutes
    CHUNK_SAMPLES = int(SAMPLING_RATE * CHUNK_DURATION_SEC)
    
    all_speech_timestamps = []
    
    with sf.SoundFile(file_path, 'r') as f:
        # Check that it's 16k mono
        if f.samplerate != SAMPLING_RATE or f.channels != 1:
            raise ValueError(f"Expected 16000 Hz mono, got {f.samplerate} Hz {f.channels} channels")
            
        current_sample_offset = 0
        
        while True:
            # Read chunk
            audio_chunk = f.read(frames=CHUNK_SAMPLES, dtype='float32')
            if len(audio_chunk) == 0:
                break
                
            # Convert to torch tensor
            chunk_tensor = torch.from_numpy(audio_chunk)
            
            # Predict
            # Using prompt constraints: merge short gaps and drop short segments,
            # using min_speech_duration_ms=400, min_silence_duration_ms=250.
            # get_speech_timestamps handles smoothing naturally.
            speech_timestamps = get_speech_timestamps(
                chunk_tensor, 
                model, 
                sampling_rate=SAMPLING_RATE,
                min_speech_duration_ms=400, 
                min_silence_duration_ms=250
            )
            
            # Convert sample indices from the chunk back to global seconds
            for ts in speech_timestamps:
                global_start_samples = current_sample_offset + ts['start']
                global_end_samples = current_sample_offset + ts['end']
                
                # convert to float seconds
                start_s = global_start_samples / SAMPLING_RATE
                end_s = global_end_samples / SAMPLING_RATE
                
                all_speech_timestamps.append({
                    "start": start_s,
                    "end": end_s
                })
            
            # Increment offset
            current_sample_offset += len(audio_chunk)

    # Note: Merging overlap across boundaries can be handled, but since gaps < 0.25s are merged by get_speech_timestamps,
    # we should do a secondary merge pass in case a word crosses the 5 min boundary.
    merged_timestamps = []
    for ts in all_speech_timestamps:
        if not merged_timestamps:
            merged_timestamps.append(ts)
            continue
            
        last = merged_timestamps[-1]
        gap = ts["start"] - last["end"]
        
        if gap < 0.25: # Merge gaps shorter than 0.25 seconds across boundaries
            last["end"] = ts["end"]
        else:
            merged_timestamps.append(ts)
            
    return merged_timestamps
