import webrtcvad

class VADEngine:
    def __init__(self, aggressiveness=3, frame_duration_ms=20, sample_rate=16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        
    def process_stream(self, stream):
        """Processes a stream of frames, returning boolean voice activity map."""
        segments = []
        for frame in stream:
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                is_speech = False
            segments.append(is_speech)
        return segments

    def get_smoothed_timestamps(self, segments, min_silence_duration=0.5, min_speech_duration=0.2):
        """
        Smooths out VAD output and returns list of dicts: {'start': s, 'end': e}
        """
        frame_sec = self.frame_duration_ms / 1000.0
        min_silence_frames = int(min_silence_duration / frame_sec)
        min_speech_frames = int(min_speech_duration / frame_sec)
        
        smoothed = []
        in_speech = False
        speech_frames = 0
        silence_frames = 0
        
        current_start = 0.0
        
        for i, is_speech in enumerate(segments):
            if is_speech:
                silence_frames = 0
                speech_frames += 1
                if not in_speech:
                    if speech_frames >= min_speech_frames:
                        in_speech = True
                        current_start = (i - speech_frames + 1) * frame_sec
            else:
                speech_frames = 0
                silence_frames += 1
                if in_speech:
                    if silence_frames >= min_silence_frames:
                        in_speech = False
                        end_time = (i - silence_frames + 1) * frame_sec
                        smoothed.append({"start": current_start, "end": end_time})
                        
        if in_speech:
            end_time = len(segments) * frame_sec
            smoothed.append({"start": current_start, "end": end_time})
            
        return smoothed
