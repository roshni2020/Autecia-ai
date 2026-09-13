"""Transformer text embedding of the raw transcript (RoBERTa CLS pooling).

Separate from backend/embed.py on purpose: that hashed embedding is the cheap,
deterministic key for per-user memory retrieval; this one feeds speech fusion.
Falls back to the hashed embedding so fusion always has a text branch.
"""
import os
import time
from functools import lru_cache

import numpy as np

from ..embed import embed as hashed_embed

DEFAULT_MODEL = "roberta-base"   # 768-d; set TEXT_ENCODER=roberta-large for 1024-d


@lru_cache(maxsize=1)
def _model():
    name = os.getenv("TEXT_ENCODER", DEFAULT_MODEL)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()
        torch.set_grad_enabled(False)
        return tok, model, name
    except Exception as e:
        print(f"[text_encoder] {name} unavailable, hashed fallback: {e}")
        return None


def available() -> bool:
    return _model() is not None


def encode(text: str) -> tuple[np.ndarray, str]:
    m = _model()
    if m is None or not text.strip():
        return hashed_embed(text), "hashed"
    tok, model, name = m
    try:
        import torch
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[0, 0]
        return cls.cpu().numpy().astype(np.float32), name
    except Exception as e:
        print(f"[text_encoder] failed, hashed fallback: {e}")
        return hashed_embed(text), "hashed"


def timed(text: str) -> tuple[np.ndarray, str, int]:
    t0 = time.perf_counter()
    v, name = encode(text)
    return v, name, int((time.perf_counter() - t0) * 1000)
