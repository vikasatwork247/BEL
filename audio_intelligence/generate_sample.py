import numpy as np
import scipy.io.wavfile as wavfile

def generate_test_audio(filename, duration=30, sample_rate=16000):
    # Generates a 30s audio with some "speech" bursts (random noise or tones) 
    # to test the VAD.
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Base background noise
    audio = np.random.normal(0, 0.01, size=t.shape)
    
    # Add some "speech" bursts
    # e.g., tone 1 at 2s-5s, tone 2 at 8s-12s, tone 3 at 15s-25s
    bursts = [
        (2, 5),
        (8, 12),
        (15, 25)
    ]
    
    for start, end in bursts:
        idx_start = int(start * sample_rate)
        idx_end = int(end * sample_rate)
        
        # Add a mixed frequency signal to simulate speech better for VAD
        burst_signal = (0.5 * np.sin(2 * np.pi * 300 * t[idx_start:idx_end]) + 
                        0.5 * np.sin(2 * np.pi * 1500 * t[idx_start:idx_end]))
        burst_signal *= np.random.normal(1, 0.2, size=burst_signal.shape) # modulate
        
        audio[idx_start:idx_end] += burst_signal
        
    # Normalize
    audio = audio / np.max(np.abs(audio))
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    wavfile.write(filename, sample_rate, audio_int16)

if __name__ == "__main__":
    generate_test_audio("inputs/sample.wav")
