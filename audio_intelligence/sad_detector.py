"""
sad_detector.py — Speech Activity Detector (Silero VAD, offline, no account needed)
======================================================================================
Uses Silero VAD via torch.hub (no HuggingFace token required).

Requirements (already in requirements.txt):
    torch, torchaudio, soundfile, pydub

Usage:
    from sad_detector import SpeechActivityDetector

    sad = SpeechActivityDetector()
    segments = sad.get_speech_segments("recording.wav")
    # Returns: [{'start': 1200, 'end': 3850}, ...]   (milliseconds)
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from typing import List, Dict


class SpeechActivityDetector:
    """
    Offline Speech Activity Detector powered by Silero VAD.

    Detects where human speech starts and ends in an audio file and
    returns millisecond-accurate timestamps.  Works with any sample
    rate — audio is resampled to 16 kHz internally if needed.

    No HuggingFace account or internet connection required after the
    first run (the model is cached locally by torch.hub).
    """

    # ------------------------------------------------------------------ #
    # Silero VAD tuning constants                                         #
    # threshold=0.35  → lower than default (0.5) so singing/noisy speech #
    #                    is also caught.                                  #
    # min_speech_duration_ms=250 → short syllables are valid speech.     #
    # min_silence_duration_ms=300 → reasonable gap between phrases.      #
    # speech_pad_ms=30 → add 30 ms padding around every segment.         #
    # ------------------------------------------------------------------ #
    SAMPLING_RATE = 16_000
    VAD_THRESHOLD = 0.35
    MIN_SPEECH_MS = 250
    MIN_SILENCE_MS = 300
    SPEECH_PAD_MS = 30
    MERGE_GAP_SEC = 0.5          # merge segments < 0.5 s apart

    def __init__(self):
        print("[SAD] Loading Silero VAD model (cached after first run)...")
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (
            self._get_speech_timestamps,
            _save_audio,
            _read_audio,
            _VADIterator,
            _collect_chunks,
        ) = utils
        print("[SAD] Model ready ✓")

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_speech_segments(self, audio_path: str) -> List[Dict[str, int]]:
        """
        Detect speech in *audio_path* and return a list of dicts:
            [{'start': <ms>, 'end': <ms>}, ...]

        Handles resampling automatically — input does NOT need to be 16 kHz.

        Args:
            audio_path: Path to any WAV/MP3/FLAC/OGG audio file.

        Returns:
            List of {'start': int, 'end': int} dicts (milliseconds).
        """
        waveform, sample_rate = self._load_audio(audio_path)
        waveform = self._ensure_16k_mono(waveform, sample_rate)

        raw_timestamps = self._get_speech_timestamps(
            waveform,
            self._model,
            sampling_rate=self.SAMPLING_RATE,
            threshold=self.VAD_THRESHOLD,
            min_speech_duration_ms=self.MIN_SPEECH_MS,
            min_silence_duration_ms=self.MIN_SILENCE_MS,
            speech_pad_ms=self.SPEECH_PAD_MS,
        )

        segments = self._samples_to_ms(raw_timestamps)
        segments = self._merge_close_segments(segments)

        print(f"[SAD] Detected {len(segments)} speech segment(s)")
        return segments

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_audio(self, path: str):
        """Load audio file via torchaudio, fall back to soundfile for WAVs."""
        try:
            waveform, sr = torchaudio.load(path)
            return waveform, sr
        except Exception:
            # soundfile fallback (handles more WAV variants)
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)  # (channels, samples)
            return waveform, sr

    def _ensure_16k_mono(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Convert to mono and resample to 16 kHz if necessary."""
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sample_rate != self.SAMPLING_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.SAMPLING_RATE
            )

        # Silero expects a 1-D tensor
        return waveform.squeeze(0)

    def _samples_to_ms(self, timestamps) -> List[Dict[str, int]]:
        """Convert Silero's sample-index timestamps → milliseconds."""
        segments = []
        for ts in timestamps:
            start_ms = int(ts["start"] / self.SAMPLING_RATE * 1000)
            end_ms   = int(ts["end"]   / self.SAMPLING_RATE * 1000)
            segments.append({"start": start_ms, "end": end_ms})
        return segments

    def _merge_close_segments(
        self, segments: List[Dict[str, int]]
    ) -> List[Dict[str, int]]:
        """Merge any two segments whose gap is smaller than MERGE_GAP_SEC."""
        merge_gap_ms = int(self.MERGE_GAP_SEC * 1000)
        merged: List[Dict[str, int]] = []
        for seg in segments:
            if not merged:
                merged.append(dict(seg))
                continue
            last = merged[-1]
            if seg["start"] - last["end"] <= merge_gap_ms:
                last["end"] = seg["end"]
            else:
                merged.append(dict(seg))
        return merged
