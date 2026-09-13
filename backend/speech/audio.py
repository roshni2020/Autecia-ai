"""Audio normalisation: anything the browser records -> 16 kHz mono PCM WAV.

Uses the ffmpeg binary bundled with imageio-ffmpeg (no system install). Raw
audio only ever lives in a temp file that is deleted when the context exits.
"""
import base64
import contextlib
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    """'data:audio/webm;base64,....' -> (bytes, suffix)."""
    head, _, b64 = data_url.partition(",")
    mime = head.split(":")[-1].split(";")[0] if ":" in head else "audio/webm"
    suffix = {"audio/webm": ".webm", "audio/wav": ".wav", "audio/x-wav": ".wav",
              "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
              "audio/ogg": ".ogg", "video/webm": ".webm"}.get(mime, ".webm")
    return base64.b64decode(b64), suffix


def convert_to_wav(src: Path, dst: Path) -> None:
    try:                                   # vendored NeuroIntent conversion (same ffmpeg)
        from .neurointent import adapter
        wav, cleanup = adapter.to_wav(str(src))
        if cleanup:
            Path(wav).replace(dst)
            return
    except Exception:
        pass
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("ffmpeg unavailable (pip install imageio-ffmpeg)")
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", "-acodec", "pcm_s16le",
                    str(dst)], check=True, timeout=120)


@contextlib.contextmanager
def temp_wav(data: bytes, suffix: str = ".webm"):
    """Yield a 16 kHz mono WAV path for `data`; both temp files are removed after."""
    tmpdir = tempfile.mkdtemp(prefix="echoloop_audio_")
    src, dst = Path(tmpdir) / f"input{suffix}", Path(tmpdir) / "audio16k.wav"
    try:
        src.write_bytes(data)
        if suffix == ".wav" and _is_16k_mono(src):
            yield src
        else:
            convert_to_wav(src, dst)
            yield dst
    finally:
        for f in (src, dst):
            with contextlib.suppress(FileNotFoundError):
                f.unlink()
        with contextlib.suppress(OSError):
            os.rmdir(tmpdir)


def _is_16k_mono(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
    except Exception:
        return False


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """PCM16 WAV -> float32 samples in [-1, 1], sample rate. Stdlib only."""
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    return raw, sr


def write_wav(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = np.clip(samples, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
