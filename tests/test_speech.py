"""Speech subsystem checks. Heavy stages skip (not fail) when their model is absent.

python -m tests.test_speech     (or: pytest tests/test_speech.py)
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("ECHOLOOP_TRACE", "0")
os.environ.setdefault("ECHOLOOP_TYPESAFE", "0")
os.environ.setdefault("ECHOLOOP_LLM", "0")
os.environ.setdefault("ECHOLOOP_SPEECH_ENCODER", "1")

from backend import memory, pipeline  # noqa: E402
from backend.agents import perception  # noqa: E402
from backend.schemas import FeedbackReq, ProcessReq, SupportProfile  # noqa: E402
from backend.speech import audio, fusion, prosody, speech_encoder, vad, whisper  # noqa: E402
from backend.speech.schemas import SpeechObservations, VadStats  # noqa: E402

SKIPPED: list[str] = []


def _skip(name, why):
    SKIPPED.append(f"{name}: {why}")
    print(f"skip {name} ({why})")


def _tone_wav(path: Path, seconds=1.2, sr=44100, stereo=True):
    """Two bursts of tone with a 400 ms gap: enough for VAD to find a pause."""
    t = np.arange(int(sr * seconds)) / sr
    x = 0.4 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    gap = (t > 0.5) & (t < 0.9)
    x[gap] = 0.0
    if stereo:
        import wave
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(np.column_stack([pcm, pcm]).tobytes())
    else:
        audio.write_wav(path, x.astype(np.float32), sr)


def _spoken_wav() -> Path | None:
    """Windows SAPI synthesises a real spoken fragment for the ASR test."""
    if sys.platform != "win32":
        return None
    out = Path(tempfile.gettempdir()) / "echoloop_spoken.wav"
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{out}'); $s.Speak('I need... um... the blue one'); $s.Dispose()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60,
                       capture_output=True)
        return out if out.exists() else None
    except Exception:
        return None


def test_audio_conversion_webm_to_16k_mono():
    if audio.ffmpeg_path() is None:
        return _skip("audio_conversion", "imageio-ffmpeg not installed")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.wav"; _tone_wav(src, stereo=True)
        webm = Path(d) / "in.webm"
        subprocess.run([audio.ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:a", "libopus", str(webm)], check=True, timeout=60)
        with audio.temp_wav(webm.read_bytes(), ".webm") as wav:
            x, sr = audio.read_wav(wav)
            assert sr == 16000, sr
            assert x.ndim == 1 and len(x) > 16000, "mono 16 kHz samples expected"
            tmp = wav
        assert not tmp.exists(), "temp WAV must be deleted"


def test_vad_finds_pause():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tone.wav"; _tone_wav(p, stereo=False, sr=16000)
        v = vad.analyze(p)
        assert v.total_duration > 1.0
        assert v.speech_duration > 0.3
        assert v.pause_count >= 1, v
        assert v.longest_pause_ms >= 250, v


def test_whisper_transcribes_speech():
    if not whisper.available():
        return _skip("whisper", "faster-whisper model unavailable")
    spoken = _spoken_wav()
    if spoken is None:
        return _skip("whisper", "no TTS available to synthesise a test utterance")
    with audio.temp_wav(spoken.read_bytes(), ".wav") as wav:
        tr = whisper.transcribe(wav)
    assert tr is not None and tr.text.strip(), "transcript expected"
    assert tr.words and all(w.end >= w.start for w in tr.words), "word timestamps expected"
    assert "blue" in tr.text.lower(), tr.text


def test_gemaps_dimension():
    if not prosody.available():
        return _skip("gemaps", "opensmile not installed")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tone.wav"; _tone_wav(p, stereo=False, sr=16000)
        g = prosody.extract(p)
    assert g is not None and g.shape == (prosody.GEMAPS_DIM,), None if g is None else g.shape
    assert np.all(np.isfinite(g))


def test_speech_encoder_finite():
    if not speech_encoder.available():
        return _skip("speech_encoder", "WavLM/wav2vec2 unavailable")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tone.wav"; _tone_wav(p, stereo=False, sr=16000)
        r = speech_encoder.encode(p)
    assert r is not None
    pooled, _ = r
    assert pooled.ndim == 1 and pooled.shape[0] >= 256 and np.all(np.isfinite(pooled))


def test_fusion_is_unit_and_degrades():
    timing = fusion.timing_vector({"speech_duration": 2, "pause_count": 2, "longest_pause_ms": 900}, True, False, 80)
    full = fusion.fuse(np.ones(8), np.ones(4), np.ones(3), timing)
    text_only = fusion.fuse(np.ones(8), None, None, timing)
    assert math.isclose(float(np.linalg.norm(full)), 1.0, abs_tol=1e-5)
    assert full.shape[0] == 8 + 4 + 3 + fusion.TIMING_DIM
    assert text_only.shape[0] == 8 + fusion.TIMING_DIM
    assert fusion.cosine(full, full) > 0.999 and fusion.cosine(full, text_only) == 0.0


def test_speech_observations_reach_perception():
    obs = SpeechObservations(transcript="I... I need... um... blue... that...",
                             vad=VadStats(pause_count=2, longest_pause_ms=1800, pause_intervals=[1.8, 0.7],
                                          speech_duration=3.1, total_duration=5.0),
                             repetition_detected=True, fragmented=True, filler_count=1,
                             asr_confidence=0.81, providers={"asr": "faster-whisper:base"})
    req = ProcessReq(user_id="t", transcript="", scene_hint=["notebook", "headphones"])
    p = perception.run(req, camera_enabled=True, speech=obs)
    assert p.transcript == obs.transcript, "Whisper wording must reach Perception unchanged"
    assert p.pause_intervals == [1.8, 0.7]
    assert p.repetition_detected and p.speech is not None and p.speech.fragmented
    s = perception.summary(p)
    assert "2 long pause(s)" in s and "fragmented" in s, s
    for banned in ("anxious", "emotion", "autis", "stress", "mood"):
        assert banned not in s.lower()


def test_precomputed_speech_features_flow_end_to_end():
    con = memory.connect(":memory:")
    memory.set_profile(con, "t", SupportProfile(camera_enabled=True))
    feats = {"observations": SpeechObservations(transcript="I need... blue...", fragmented=True,
                                                vad=VadStats(pause_count=1, pause_intervals=[1.2])).model_dump(),
             "fused_embedding": [0.1] * 32}
    r = pipeline.process(con, ProcessReq(user_id="t", transcript="", scene_hint=["notebook", "headphones"],
                                         speech_features=feats))
    assert r["transcript"] == "I need... blue..." and r["speech_observations"]["fragmented"]
    fb = pipeline.feedback(con, FeedbackReq(interaction_id=r["interaction_id"], accepted=False,
                                            confirmed_text="I need my headphones."))
    assert fb["memory_saved"]
    row = con.execute("SELECT utterance_embedding, speech_summary FROM memories").fetchone()
    assert row["utterance_embedding"] is not None and "pause" in row["speech_summary"]
    r2 = pipeline.process(con, ProcessReq(user_id="t", transcript="", scene_hint=["notebook", "headphones"],
                                          speech_features=feats))
    assert r2["memory_matches"][0]["speech_similarity"] > 0.99
    assert r2["top_candidate"] == "I need my headphones."
    assert r2["candidate_detail"][0]["features"]["speech"] > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all speech checks passed" + (f" ({len(SKIPPED)} skipped)" if SKIPPED else ""))
