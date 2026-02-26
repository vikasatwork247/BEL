import numpy as np
import scipy.signal
import soundfile as sf

def compute_mfcc(file_path, timestamps):
    """
    Computes basic spectral features using scipy to simulate MFCCs
    for speech segments found by the VAD.
    """
    if not timestamps:
        return np.zeros(13)
        
    data, sr = sf.read(file_path, dtype='float32')
    
    # Collect speech samples
    speech_samples = []
    for ts in timestamps:
        start_idx = int(ts['start'] * sr)
        end_idx = int(ts['end'] * sr)
        # Add bounds check
        if start_idx < len(data):
            end_idx = min(end_idx, len(data))
            speech_samples.append(data[start_idx:end_idx])
        
    if not speech_samples:
        return np.zeros(13)
        
    concatenated = np.concatenate(speech_samples)
    if len(concatenated) == 0:
        return np.zeros(13)
    
    # Very simplified feature extraction just using spectral density
    f, t, Sxx = scipy.signal.spectrogram(concatenated, sr)
    if Sxx.size == 0:
        return np.zeros(13)
        
    mean_spectrum = np.mean(Sxx, axis=1)
    
    # Mocking a 13-dim feature vector by pooling spectrum
    pool_size = max(1, len(mean_spectrum) // 13)
    features = []
    for i in range(13):
        start_i = i * pool_size
        end_i = min((i + 1) * pool_size, len(mean_spectrum))
        if start_i < len(mean_spectrum):
            features.append(np.mean(mean_spectrum[start_i:end_i]))
        else:
            features.append(0.0)
            
    # Pad if we didn't get exactly 13 due to small array
    while len(features) < 13:
        features.append(0.0)
        
    return np.array(features)
