import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os

class LanguageClassifier:
    def __init__(self):
        # We simulate a trained offline model to meet fully offline requirements.
        # But this time, we use a real neural language ID model cached locally.
        try:
            from huggingface_hub import snapshot_download
            import shutil
            import speechbrain.utils.fetching as fetching
            
            # Absolute monkey patch of fetch to override local_strategy internally
            original_fetch = fetching.fetch
            def safe_fetch(*args, **kwargs):
                # Always force COPY to prevent Windows symlink error
                kwargs['local_strategy'] = fetching.LocalStrategy.COPY
                # Also disable collect_in if present to avoid second symlink warning
                if 'collect_in' in kwargs: 
                    kwargs['collect_in'] = None
                return original_fetch(*args, **kwargs)
            fetching.fetch = safe_fetch
            
            # Explicitly download the model without symlinks using huggingface_hub itself, 
            # bypassing SpeechBrain's internal FetchFrom bug on Windows
            save_dir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "lang-id")
            os.makedirs(save_dir, exist_ok=True)
            
            snapshot_download(
                repo_id="speechbrain/lang-id-commonlanguage_ecapa",
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
            
            self.classifier = EncoderClassifier.from_hparams(
                source=save_dir, 
                savedir=save_dir,
                run_opts={"device": "cpu"}
            )
        except Exception as e:
            print(f"Failed to load SpeechBrain Language model: {e}")
            self.classifier = None
        
    def predict(self, file_path, timestamps):
        """
        Uses raw audio segments matching the speech timestamps to accurately detect the language.
        """
        if not self.classifier or not timestamps:
            return "Unknown", 0.0
            
        try:
            signal, fs = torchaudio.load(file_path)
            
            # Since speechbrain expects 16kHz
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            # Collect audio where speech is detected to avoid analyzing noise/silence
            speech_chunks = []
            for ts in timestamps:
                start_idx = int(ts["start"] * 16000)
                end_idx = int(ts["end"] * 16000)
                # Bounds check
                if start_idx < signal.shape[1]:
                    end_idx = min(end_idx, signal.shape[1])
                    speech_chunks.append(signal[:, start_idx:end_idx])
            
            if not speech_chunks:
                return "Unknown", 0.0
                
            # Concatenate chunks
            speech_signal = torch.cat(speech_chunks, dim=1)
            
            # Sub-sample if too long to prevent extreme CPU time (max 10 seconds of pure speech)
            max_samples = 16000 * 10 
            if speech_signal.shape[1] > max_samples:
                speech_signal = speech_signal[:, :max_samples]
                
            prediction = self.classifier.classify_batch(speech_signal)
            
            # classify_batch returns (out_prob, score, index, text_lab)
            prediction = self.classifier.classify_batch(speech_signal)
            
            # Extract the actual string label from the list of predictions
            lang_label = str(prediction[3][0])
            
            # Get max probability confidence
            # prediction[1] is the log-probabilities tensor
            conf = torch.exp(prediction[1].max()).item()
            
            # Clean up label if it has form 'Language: English' or similar
            if ":" in lang_label:
                lang_label = lang_label.split(":")[-1].strip()
                
            return lang_label, conf
            
        except Exception as e:
            print(f"Language prediction logic failed: {e}")
            return "Unknown", 0.0
