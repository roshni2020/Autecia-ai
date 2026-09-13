"""Deterministic offline text embedding (hashing trick) + cosine.

ponytail: hashed bag-of-ngrams, no model download, stable across processes.
Swap in a sentence-transformer / API embedding if retrieval quality matters.
"""
import re
from zlib import crc32
import numpy as np

DIM = 256
_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    toks = tokens(text)
    for gram in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
        h = crc32(gram.encode())
        v[h % DIM] += 1.0 if h & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.dot(a, b))
    return max(0.0, min(1.0, d))  # clamp; negatives are noise for our use
