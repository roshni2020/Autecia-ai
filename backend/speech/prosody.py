"""openSMILE GeMAPSv01b functionals: 62 observable acoustic/prosodic features.

Pitch, loudness, voicing, spectral shape, pause/rate related statistics. They
are numbers about the signal. Nothing here maps them to a mood, trait or label.
"""
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

GEMAPS_DIM = 62


@lru_cache(maxsize=1)
def _smile():
    try:
        import opensmile
        return opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSv01b,
                               feature_level=opensmile.FeatureLevel.Functionals)
    except Exception as e:
        print(f"[prosody] openSMILE unavailable: {e}")
        return None


def available() -> bool:
    return _smile() is not None


def feature_names() -> list[str]:
    s = _smile()
    return list(s.feature_names) if s is not None else []


def extract(wav_path: Path) -> np.ndarray | None:
    s = _smile()
    if s is None:
        return None
    try:
        df = s.process_file(str(wav_path))
        v = df.to_numpy(dtype=np.float32).reshape(-1)
        return np.nan_to_num(v)
    except Exception as e:
        print(f"[prosody] extraction failed: {e}")
        return None


def timed(wav_path: Path) -> tuple[np.ndarray | None, int]:
    t0 = time.perf_counter()
    g = extract(wav_path)
    return g, int((time.perf_counter() - t0) * 1000)
