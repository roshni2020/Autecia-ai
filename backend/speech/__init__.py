"""Speech subsystem: microphone audio -> observable speech facts + utterance vector.

    audio -> VAD -> Whisper (word timestamps) -> WavLM + GeMAPS + text encoder
          -> fusion -> SpeechAnalysis (observations for Perception, embedding for Learning)

Every stage is optional. Missing torch: Whisper + VAD + prosody still run.
Missing Whisper: the caller keeps the transcript it already has (typed/browser).
"""
import os
import re
import time

import numpy as np

from . import audio, fusion, prosody, speech_encoder, text_encoder, vad, whisper
try:
    from .neurointent import adapter as neurointent
except Exception as _e:  # torch not installed: vendored path off, own modules still work
    print(f"[speech] vendored NeuroIntent path disabled: {_e}")

    class neurointent:  # type: ignore[no-redef]
        ENABLED = False

        @staticmethod
        def available() -> bool:
            return False

        @staticmethod
        def analyze(*_a, **_k):
            return None
from .schemas import SpeechAnalysis, SpeechObservations, SpeechSegment, WordTimestamp

FILLERS = {"um", "uh", "er", "hmm", "mm", "erm", "ah", "like"}
SPEECH_ON = os.getenv("ECHOLOOP_SPEECH", "1") != "0"


def status() -> dict:
    """Cheap capability report (import checks only; models load lazily on first use)."""
    from importlib.util import find_spec
    if not SPEECH_ON:
        return {"enabled": False}
    has = lambda m: find_spec(m) is not None  # noqa: E731
    return {"enabled": True, "ffmpeg": audio.ffmpeg_path() is not None,
            "neurointent": neurointent.ENABLED and has("faster_whisper") and has("opensmile") and has("transformers"),
            "whisper": has("faster_whisper"), "whisper_model": os.getenv("WHISPER_MODEL", "base"),
            "vad": has("silero_vad") and has("torch"), "gemaps": has("opensmile"),
            "speech_encoder": (os.getenv("SPEECH_ENCODER", speech_encoder.DEFAULT_MODEL)
                               if has("transformers") and has("torch") else None),
            "text_encoder": has("transformers") and has("torch")}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def observe(transcript: str, words, v, provider_info: dict) -> SpeechObservations:
    toks = _tokens(transcript)
    content = [t for t in toks if t not in FILLERS]
    repetition = any(a == b for a, b in zip(content, content[1:]))
    trailing = bool(re.search(r"(\.\.\.|,|\bthe\b|\ba\b|\bthat\b|\bmy\b)\s*$", transcript.strip(), re.I))
    fragmented = len(content) < 5 or trailing or v.pause_count >= 2
    wpm = (len(toks) / (v.speech_duration / 60)) if v.speech_duration > 0.3 else 0.0
    return SpeechObservations(
        transcript=transcript, words=words, vad=v, repetition_detected=repetition,
        fragmented=fragmented, filler_count=sum(t in FILLERS for t in toks),
        speaking_rate_wpm=round(min(wpm, 400.0), 1), providers=provider_info)


def analyze_audio(data: bytes, suffix: str = ".webm", transcript_hint: str = "") -> SpeechAnalysis | None:
    """Full pass over one utterance. Returns None only if the audio is unusable."""
    if not SPEECH_ON:
        return None
    providers, latency = {}, {}
    ni = None
    try:
        with audio.temp_wav(data, suffix) as wav:
            v, ms = vad.timed(wav); latency["vad"] = ms; providers["vad"] = v.provider
            t0 = time.perf_counter()
            try:                                   # vendored NeuroIntent pass: GeMAPS + Whisper + RoBERTa (+fusion)
                ni = neurointent.analyze(wav)
            except Exception as e:
                print(f"[speech] neurointent pass failed, own modules: {e}")
            latency["neurointent"] = int((time.perf_counter() - t0) * 1000)
            if ni is not None:
                providers["asr"] = f"faster-whisper:{ni['asr_model']} (neurointent)"
                providers["prosody"] = "opensmile:GeMAPSv01b (neurointent)"
                providers["text_encoder"] = "roberta-large (neurointent)"
                transcript = ni["transcript"]
                words = [WordTimestamp(**w) for w in ni["words"]]
                segs = [SpeechSegment(**s) for s in ni["segments"]]
                lang, conf, g = ni["language"], ni["asr_confidence"], ni["gemaps"]
                t_emb = ni["fusion_vec"] if ni["fusion_vec"] is not None else ni["cls"]
                if ni["fusion_vec"] is not None:
                    providers["fusion"] = "FusionLayer v1 (neurointent)"
            else:
                tr, ms = whisper.timed(wav); latency["asr"] = ms
                if tr is not None:
                    providers["asr"] = f"{tr.provider}:{tr.model}"
                    transcript, words, segs = tr.text, tr.words, tr.segments
                    lang, conf = tr.language, tr.asr_confidence
                else:                               # keep whatever the client heard/typed
                    providers["asr"] = "client"
                    transcript, words, segs, lang, conf = transcript_hint, [], [], None, 0.0
                g, ms = prosody.timed(wav); latency["gemaps"] = ms
                if g is not None:
                    providers["prosody"] = "opensmile:GeMAPSv01b"
            s, ms = speech_encoder.timed(wav); latency["speech_encoder"] = ms
            if s is not None:
                providers["speech_encoder"] = speech_encoder.model_name()
    except Exception as e:
        print(f"[speech] audio unusable: {e}")
        return None

    if ni is None:
        t0 = time.perf_counter()
        t_emb, t_name = text_encoder.encode(transcript) if transcript.strip() else (None, "none")
        latency["text_encoder"] = int((time.perf_counter() - t0) * 1000); providers["text_encoder"] = t_name

    obs = observe(transcript, words, v, providers)
    obs.segments, obs.language, obs.asr_confidence, obs.latency_ms = segs, lang, conf, latency
    obs.gemaps_dim = int(g.shape[0]) if g is not None else 0
    obs.speech_embedding_dim = int(s.shape[0]) if s is not None else 0
    obs.text_embedding_dim = int(t_emb.shape[0]) if t_emb is not None else 0

    timing = fusion.timing_vector(v.model_dump(), obs.fragmented, obs.repetition_detected, obs.speaking_rate_wpm)
    fused = fusion.fuse(t_emb, s, g, timing)
    obs.fused_embedding_dim = int(fused.shape[0])
    return SpeechAnalysis(observations=obs, fused_embedding=[round(float(x), 5) for x in fused],
                          gemaps=[float(x) for x in g] if g is not None else None)
