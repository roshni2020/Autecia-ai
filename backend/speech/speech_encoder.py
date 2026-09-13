"""Pretrained speech representation: waveform -> WavLM (or wav2vec2) embedding.

Keeps information the transcript loses (how the words were produced). Returns
a pooled utterance vector; frame embeddings only on request, never persisted
by the live pipeline.
"""
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, read_wav

DEFAULT_MODEL = "microsoft/wavlm-base-plus"     # 768-d
FALLBACK_MODEL = "facebook/wav2vec2-base"       # 768-d


@lru_cache(maxsize=1)
def _model():
    if os.getenv("ECHOLOOP_SPEECH_ENCODER", "0") == "0":   # off by default: ~1.5 s/utterance on CPU
        return None
    name = os.getenv("SPEECH_ENCODER", DEFAULT_MODEL)
    for candidate in (name, FALLBACK_MODEL):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
            fe = AutoFeatureExtractor.from_pretrained(candidate)
            model = AutoModel.from_pretrained(candidate).eval()
            torch.set_grad_enabled(False)
            return fe, model, candidate
        except Exception as e:
            print(f"[speech_encoder] {candidate} unavailable: {e}")
    return None


def available() -> bool:
    return _model() is not None


def model_name() -> str | None:
    m = _model()
    return m[2] if m else None


def encode(wav_path: Path, frames: bool = False) -> tuple[np.ndarray, np.ndarray | None] | None:
    m = _model()
    if m is None:
        return None
    fe, model, _ = m
    try:
        import torch
        x, sr = read_wav(wav_path)
        if sr != SAMPLE_RATE or len(x) < SAMPLE_RATE // 10:
            return None
        x = x[: SAMPLE_RATE * 30]                       # cap at 30 s for memory
        inputs = fe(x, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            h = model(**inputs).last_hidden_state[0]      # [T, D]
        pooled = h.mean(dim=0).cpu().numpy().astype(np.float32)
        return pooled, (h.cpu().numpy().astype(np.float32) if frames else None)
    except Exception as e:
        print(f"[speech_encoder] encoding failed: {e}")
        return None


def timed(wav_path: Path) -> tuple[np.ndarray | None, int]:
    t0 = time.perf_counter()
    r = encode(wav_path)
    return (r[0] if r else None), int((time.perf_counter() - t0) * 1000)
