import soundfile as sf
import numpy as np

class AudioStreamer:
    def __init__(self, file_path: str, frame_duration_ms: int = 20):
        self.file_path = file_path
        self.frame_duration_ms = frame_duration_ms
        self.info = sf.info(file_path)
        self.sample_rate = self.info.samplerate
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        
    def stream(self):
        """Yields audio frames as bytes sequentially."""
        with sf.SoundFile(self.file_path, 'r') as f:
            while True:
                data = f.read(self.frame_size, dtype='int16')
                if len(data) == 0:
                    break
                # pad if the last block is too short
                if len(data) < self.frame_size:
                    data = np.pad(data, (0, self.frame_size - len(data)), 'constant')
                yield data.tobytes()
                
    def get_duration(self) -> float:
        return self.info.frames / self.sample_rate
