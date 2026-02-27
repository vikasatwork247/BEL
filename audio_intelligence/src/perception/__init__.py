"""
src/perception — 4-stage speech perception pipeline
====================================================
Stage 1: separation.py  — Demucs vocal isolation
Stage 2: sad.py         — SpeechBrain neural VAD
Stage 3: segmentation.py— Boundary refinement
Stage 4: alignment.py   — WhisperX forced alignment → ms timestamps
Stage 5: pipeline.py    — Orchestrator → outputs/segments.json
"""
