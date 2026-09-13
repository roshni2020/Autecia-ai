"""Voice activity detection: speech segments, pauses, boundaries. Timing only.

Silero VAD when available (torch); otherwise a frame-energy detector so pause
statistics still exist. Silence is a timing fact here, never a feeling.
"""
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, read_wav
from .schemas import SpeechSegment, VadStats

MIN_PAUSE_MS = 250           # gaps shorter than this are articulation, not pauses


@lru_cache(maxsize=1)
def _silero():
    try:
        import torch  # noqa: F401
        from silero_vad import get_speech_timestamps, load_silero_vad
        return load_silero_vad(onnx=False), get_speech_timestamps
    except Exception as e:
        print(f"[vad] silero unavailable, energy VAD: {e}")
        return None


def _energy_segments(x: np.ndarray, sr: int, frame_ms: int = 30) -> list[tuple[float, float]]:
    hop = int(sr * frame_ms / 1000)
    if len(x) < hop:
        return []
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = max(0.01, float(np.percentile(rms, 30)) * 2.0)
    active = rms > thr
    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append((start * frame_ms / 1000, i * frame_ms / 1000)); start = None
    if start is not None:
        segs.append((start * frame_ms / 1000, len(active) * frame_ms / 1000))
    return _merge(segs)


def _merge(segs: list[tuple[float, float]], gap: float = MIN_PAUSE_MS / 1000) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in segs:
        if out and s - out[-1][1] < gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= 0.08]


def analyze(wav_path: Path) -> VadStats:
    x, sr = read_wav(wav_path)
    total = len(x) / sr if sr else 0.0
    provider, segs = "energy", []
    sil = _silero()
    if sil is not None:
        try:
            import torch
            model, get_ts = sil
            ts = get_ts(torch.from_numpy(x), model, sampling_rate=SAMPLE_RATE, return_seconds=True,
                        min_silence_duration_ms=MIN_PAUSE_MS)
            segs = _merge([(t["start"], t["end"]) for t in ts]); provider = "silero"
        except Exception as e:
            print(f"[vad] silero failed, energy VAD: {e}")
    if not segs:
        segs = _energy_segments(x, sr)
    pauses = [round(b[0] - a[1], 3) for a, b in zip(segs, segs[1:]) if b[0] - a[1] >= MIN_PAUSE_MS / 1000]
    return VadStats(
        total_duration=round(total, 3),
        speech_duration=round(sum(e - s for s, e in segs), 3),
        pause_count=len(pauses),
        longest_pause_ms=int(max(pauses) * 1000) if pauses else 0,
        leading_silence_ms=int(segs[0][0] * 1000) if segs else int(total * 1000),
        speech_segments=[SpeechSegment(start=round(s, 3), end=round(e, 3)) for s, e in segs],
        pause_intervals=pauses, provider=provider)


def timed(wav_path: Path) -> tuple[VadStats, int]:
    t0 = time.perf_counter()
    v = analyze(wav_path)
    return v, int((time.perf_counter() - t0) * 1000)
