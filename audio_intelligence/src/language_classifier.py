import os
import torch
import torchaudio
import numpy as np


# Full language name map for Whisper language codes
LANGUAGE_NAME_MAP = {
    # Indian languages
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "sd": "Sindhi",
    "sa": "Sanskrit",
    "si": "Sinhala",
    # Major foreign languages
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "sw": "Swahili",
    "af": "Afrikaans",
}


class LanguageClassifier:
    """
    Detects language using OpenAI Whisper-tiny (runs fully on CPU).
    Supports 99 languages including all major Indian languages and
    important foreign languages.
    
    Falls back to SpeechBrain ECAPA if Whisper is unavailable.
    """

    def __init__(self):
        self._whisper_model = None
        self._speechbrain_classifier = None
        self._backend = "none"

        # --- Try Whisper-tiny first ---
        try:
            import whisper
            print("Loading Whisper-tiny for language detection (CPU)...")
            self._whisper_model = whisper.load_model("tiny", device="cpu")
            self._backend = "whisper"
            print("Whisper-tiny loaded successfully.")
        except ImportError:
            print("openai-whisper not installed. Trying SpeechBrain fallback...")
        except Exception as e:
            print(f"Whisper load failed: {e}. Trying SpeechBrain fallback...")

        # --- Fallback: SpeechBrain ---
        if self._backend == "none":
            try:
                from huggingface_hub import snapshot_download
                import speechbrain.utils.fetching as fetching
                from speechbrain.inference.classifiers import EncoderClassifier

                original_fetch = fetching.fetch
                def safe_fetch(*args, **kwargs):
                    kwargs['local_strategy'] = fetching.LocalStrategy.COPY
                    if 'collect_in' in kwargs:
                        kwargs['collect_in'] = None
                    return original_fetch(*args, **kwargs)
                fetching.fetch = safe_fetch

                save_dir = os.path.join(
                    os.path.expanduser("~"), ".cache", "speechbrain", "lang-id"
                )
                os.makedirs(save_dir, exist_ok=True)
                snapshot_download(
                    repo_id="speechbrain/lang-id-commonlanguage_ecapa",
                    local_dir=save_dir,
                    local_dir_use_symlinks=False
                )
                self._speechbrain_classifier = EncoderClassifier.from_hparams(
                    source=save_dir,
                    savedir=save_dir,
                    run_opts={"device": "cpu"}
                )
                self._backend = "speechbrain"
                print("SpeechBrain language classifier loaded.")
            except Exception as e:
                print(f"SpeechBrain fallback also failed: {e}")
                self._backend = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, file_path: str, timestamps: list) -> tuple:
        """
        Detects the language from speech segments.
        
        Returns: (language_name: str, confidence: float [0-1])
        """
        if not timestamps:
            return "Unknown", 0.0

        if self._backend == "whisper":
            return self._predict_whisper(file_path, timestamps)
        elif self._backend == "speechbrain":
            return self._predict_speechbrain(file_path, timestamps)
        else:
            return "Unknown", 0.0

    # ------------------------------------------------------------------
    # Whisper backend
    # ------------------------------------------------------------------

    def _predict_whisper(self, file_path: str, timestamps: list) -> tuple:
        """
        Uses Whisper's internal language detection on speech-only audio.
        Whisper is run in detect_language mode (no transcription) for speed.
        """
        try:
            import whisper

            # Load audio and collect speech-only segments
            signal, sr = torchaudio.load(file_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)

            # Take up to 30 seconds of speech for better detection
            speech_chunks = []
            collected = 0.0
            MAX_SEC = 30.0
            for ts in timestamps:
                if collected >= MAX_SEC:
                    break
                start_idx = int(ts["start"] * 16000)
                end_idx = int(ts["end"] * 16000)
                if start_idx < signal.shape[1]:
                    end_idx = min(end_idx, signal.shape[1])
                    chunk = signal[:, start_idx:end_idx]
                    speech_chunks.append(chunk)
                    collected += (end_idx - start_idx) / 16000.0

            if not speech_chunks:
                return "Unknown", 0.0

            speech_signal = torch.cat(speech_chunks, dim=1)

            # Convert to mono float32 numpy (Whisper expects 16kHz mono numpy array)
            audio_np = speech_signal.mean(dim=0).numpy().astype(np.float32)

            # Pad/trim to Whisper's 30s window (480000 samples) for best results
            WHISPER_SAMPLES = 480000
            if len(audio_np) < WHISPER_SAMPLES:
                audio_np = np.pad(audio_np, (0, WHISPER_SAMPLES - len(audio_np)))
            else:
                audio_np = audio_np[:WHISPER_SAMPLES]

            # Run Whisper language detection
            audio_tensor = whisper.pad_or_trim(audio_np)
            mel = whisper.log_mel_spectrogram(audio_tensor).to("cpu")

            _, probs = self._whisper_model.detect_language(mel)

            # Get best language
            best_lang_code = max(probs, key=probs.get)
            confidence = float(probs[best_lang_code])

            # Map code to full name
            lang_name = LANGUAGE_NAME_MAP.get(best_lang_code, best_lang_code.upper())

            return lang_name, round(confidence, 4)

        except Exception as e:
            print(f"Whisper language prediction failed: {e}")
            return "Unknown", 0.0

    # ------------------------------------------------------------------
    # SpeechBrain fallback backend
    # ------------------------------------------------------------------

    def _predict_speechbrain(self, file_path: str, timestamps: list) -> tuple:
        """
        SpeechBrain ECAPA-TDNN based language ID (fallback, ~25 languages).
        """
        try:
            signal, fs = torchaudio.load(file_path)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)

            speech_chunks = []
            for ts in timestamps:
                start_idx = int(ts["start"] * 16000)
                end_idx = int(ts["end"] * 16000)
                if start_idx < signal.shape[1]:
                    end_idx = min(end_idx, signal.shape[1])
                    speech_chunks.append(signal[:, start_idx:end_idx])

            if not speech_chunks:
                return "Unknown", 0.0

            speech_signal = torch.cat(speech_chunks, dim=1)
            max_samples = 16000 * 10
            if speech_signal.shape[1] > max_samples:
                speech_signal = speech_signal[:, :max_samples]

            prediction = self._speechbrain_classifier.classify_batch(speech_signal)
            lang_label = str(prediction[3][0])
            conf = torch.exp(prediction[1].max()).item()

            if ":" in lang_label:
                lang_label = lang_label.split(":")[-1].strip()

            return lang_label, round(float(conf), 4)

        except Exception as e:
            print(f"SpeechBrain language prediction failed: {e}")
            return "Unknown", 0.0
