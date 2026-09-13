"""Server-side ASR with faster-whisper. Word timestamps, raw wording preserved.

WHISPER_MODEL=base (default, fast) ... large-v3, or a path to a fine-tuned
CTranslate2 model for atypical speech. Nothing is "cleaned": fillers, repeats
and fragments are the signal downstream agents need.
"""
import os
import time
from functools import lru_cache
from pathlib import Path

from .schemas import SpeechSegment, Transcription, WordTimestamp

# Whisper is trained to drop disfluencies; a disfluent prompt nudges it to keep them.
DISFLUENT_PROMPT = "Um, uh... I, I need... that, um, the... you know... hmm."


@lru_cache(maxsize=1)
def _model():
    name = os.getenv("WHISPER_MODEL", "base")
    try:
        from faster_whisper import WhisperModel
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute = os.getenv("WHISPER_COMPUTE", "int8" if device == "cpu" else "float16")
        return WhisperModel(name, device=device, compute_type=compute), name
    except Exception as e:
        print(f"[whisper] unavailable ({name}): {e}")
        return None


def available() -> bool:
    return _model() is not None


def transcribe(wav_path: Path, language: str | None = None) -> Transcription | None:
    m = _model()
    if m is None:
        return None
    model, name = m
    try:
        segments, info = model.transcribe(
            str(wav_path), language=language or os.getenv("WHISPER_LANGUAGE") or None,
            beam_size=3, word_timestamps=True, condition_on_previous_text=False,
            initial_prompt=DISFLUENT_PROMPT, vad_filter=False)
        words, segs, texts = [], [], []
        for s in segments:
            segs.append(SpeechSegment(start=round(s.start, 3), end=round(s.end, 3)))
            texts.append(s.text.strip())
            for w in (s.words or []):
                words.append(WordTimestamp(word=w.word.strip(), start=round(w.start, 3),
                                           end=round(w.end, 3), probability=round(w.probability, 3)))
        conf = sum(w.probability for w in words) / len(words) if words else 0.0
        return Transcription(text=" ".join(t for t in texts if t), words=words, segments=segs,
                             language=info.language, language_probability=round(info.language_probability, 3),
                             asr_confidence=round(conf, 3), provider="faster-whisper", model=name)
    except Exception as e:
        print(f"[whisper] transcription failed: {e}")
        return None


def timed(wav_path: Path) -> tuple[Transcription | None, int]:
    t0 = time.perf_counter()
    t = transcribe(wav_path)
    return t, int((time.perf_counter() - t0) * 1000)
