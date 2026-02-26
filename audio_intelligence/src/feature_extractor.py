import numpy as np
import soundfile as sf


def _compute_mfcc_manual(signal: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Manual MFCC computation using numpy/scipy (fallback if librosa unavailable).
    Uses mel filterbank + DCT for proper MFCC coefficients.
    """
    import scipy.signal
    import scipy.fftpack

    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Frame parameters
    frame_size = int(0.025 * sr)   # 25ms
    frame_stride = int(0.010 * sr) # 10ms
    signal_length = len(emphasized)
    num_frames = 1 + (signal_length - frame_size) // frame_stride

    if num_frames <= 0:
        return np.zeros(n_mfcc)

    # Frame the signal
    indices = (np.arange(frame_size)[None, :] +
               np.arange(num_frames)[:, None] * frame_stride)
    indices = np.minimum(indices, signal_length - 1)
    frames = emphasized[indices]

    # Hamming window
    frames *= np.hamming(frame_size)

    # FFT
    NFFT = 512
    mag_frames = np.abs(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

    # Mel filterbank
    n_filters = 26
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filters, NFFT // 2 + 1))
    for m in range(1, n_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m        = bin_points[m]
        f_m_plus   = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # DCT to get MFCCs
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # Mean across time axis
    return np.mean(mfcc, axis=0)


def compute_mfcc(file_path: str, timestamps: list, n_mfcc: int = 13) -> np.ndarray:
    """
    Computes real MFCC features from speech segments identified by VAD.
    
    Uses librosa if available (preferred), otherwise falls back to manual
    numpy/scipy implementation.
    
    Returns: np.ndarray of shape (n_mfcc*3,) — MFCCs + delta + delta-delta
             concatenated and averaged across time for a compact feature vector.
    """
    if not timestamps:
        return np.zeros(n_mfcc * 3)

    data, sr = sf.read(file_path, dtype='float32')

    # Handle stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Collect speech-only samples
    speech_samples = []
    for ts in timestamps:
        start_idx = int(ts['start'] * sr)
        end_idx = int(ts['end'] * sr)
        if start_idx < len(data):
            end_idx = min(end_idx, len(data))
            chunk = data[start_idx:end_idx]
            if len(chunk) > 0:
                speech_samples.append(chunk)

    if not speech_samples:
        return np.zeros(n_mfcc * 3)

    concatenated = np.concatenate(speech_samples)
    if len(concatenated) < 400:   # Too short for any meaningful MFCC
        return np.zeros(n_mfcc * 3)

    # --- Try librosa for real MFCCs with delta features ---
    try:
        import librosa

        # Resample to 16kHz if needed (librosa works best at target sr)
        if sr != 16000:
            concatenated = librosa.resample(concatenated, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Real MFCCs
        mfccs = librosa.feature.mfcc(y=concatenated, sr=sr, n_mfcc=n_mfcc,
                                      n_fft=512, hop_length=160, win_length=400)

        # Delta (velocity) features
        delta_mfcc = librosa.feature.delta(mfccs)

        # Delta-delta (acceleration) features
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)

        # Mean across time → compact representation
        mfcc_mean    = np.mean(mfccs,      axis=1)
        delta_mean   = np.mean(delta_mfcc, axis=1)
        delta2_mean  = np.mean(delta2_mfcc, axis=1)

        return np.concatenate([mfcc_mean, delta_mean, delta2_mean])

    except ImportError:
        print("librosa not found, using manual MFCC computation.")

    except Exception as e:
        print(f"librosa MFCC failed: {e}. Using manual computation.")

    # --- Manual fallback ---
    mfcc_manual = _compute_mfcc_manual(concatenated, sr, n_mfcc)
    # Return zero-padded to match (n_mfcc*3) shape expected downstream
    return np.concatenate([mfcc_manual, np.zeros(n_mfcc * 2)])
