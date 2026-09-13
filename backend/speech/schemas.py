"""Typed outputs of the speech subsystem. Observable facts only — no state inference."""
from typing import Optional
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 0.0


class SpeechSegment(BaseModel):
    start: float
    end: float


class VadStats(BaseModel):
    total_duration: float = 0.0
    speech_duration: float = 0.0
    pause_count: int = 0
    longest_pause_ms: int = 0
    leading_silence_ms: int = 0
    speech_segments: list[SpeechSegment] = Field(default_factory=list)
    pause_intervals: list[float] = Field(default_factory=list)  # seconds, between segments
    provider: str = "none"


class Transcription(BaseModel):
    text: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)
    segments: list[SpeechSegment] = Field(default_factory=list)
    language: Optional[str] = None
    language_probability: float = 0.0
    asr_confidence: float = 0.0     # mean word probability when available
    provider: str = "none"
    model: Optional[str] = None


class SpeechObservations(BaseModel):
    """What reaches the Perception Agent. JSON-safe, small, never raw audio."""
    transcript: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)
    segments: list[SpeechSegment] = Field(default_factory=list)
    language: Optional[str] = None
    asr_confidence: float = 0.0
    vad: VadStats = Field(default_factory=VadStats)
    repetition_detected: bool = False
    fragmented: bool = False              # short / trailing / pause-broken utterance
    filler_count: int = 0
    speaking_rate_wpm: float = 0.0
    gemaps_dim: int = 0                   # 62 when openSMILE ran
    speech_embedding_dim: int = 0         # WavLM / wav2vec2 pooled size
    text_embedding_dim: int = 0
    fused_embedding_dim: int = 0
    providers: dict[str, str] = Field(default_factory=dict)
    latency_ms: dict[str, int] = Field(default_factory=dict)


class SpeechAnalysis(BaseModel):
    """Observations plus the internal fused embedding (kept server-side)."""
    observations: SpeechObservations
    fused_embedding: Optional[list[float]] = None   # utterance representation
    gemaps: Optional[list[float]] = None            # raw 62 functionals, internal
