"""Typed agent messages + interaction records (spec sections 3, 7, 14)."""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

SUPPORT_MODES = [
    "autism_neurodivergent",
    "speech_language",
    "aphasia_word_finding",
    "cognitive_fatigue",
    "aac",
    "other",
    "prefer_not_to_say",
]


class SupportProfile(BaseModel):
    """User-selected preference. Not a diagnosis, not inferred."""
    support_mode: str = "prefer_not_to_say"
    pause_tolerance: Literal["short", "medium", "long"] = "medium"
    suggestion_count: int = 3
    literal_language: bool = True
    camera_enabled: bool = False
    quick_phrases_enabled: bool = True

    @staticmethod
    def for_mode(mode: str) -> "SupportProfile":
        presets = {
            "autism_neurodivergent": dict(pause_tolerance="long", suggestion_count=3,
                                          literal_language=True, camera_enabled=False),
            "speech_language": dict(pause_tolerance="long", suggestion_count=3,
                                    literal_language=True, camera_enabled=True),
            "aphasia_word_finding": dict(pause_tolerance="long", suggestion_count=3,
                                         literal_language=True, camera_enabled=True),
            "cognitive_fatigue": dict(pause_tolerance="medium", suggestion_count=2,
                                      literal_language=True, camera_enabled=False),
            "aac": dict(pause_tolerance="medium", suggestion_count=4,
                        literal_language=True, camera_enabled=False),
        }
        return SupportProfile(support_mode=mode, **presets.get(mode, {}))


class DetectedObject(BaseModel):
    label: str
    confidence: float = 0.0


class Gesture(BaseModel):
    type: Literal["pointing", "reaching", "none"] = "none"
    target: Optional[str] = None
    confidence: float = 0.0


class Perception(BaseModel):
    transcript: str = ""
    asr_alternatives: list[str] = Field(default_factory=list)
    pause_intervals: list[float] = Field(default_factory=list)
    repetition_detected: bool = False
    objects: list[DetectedObject] = Field(default_factory=list)
    gesture: Gesture = Field(default_factory=Gesture)
    camera_enabled: bool = False
    perception_confidence: float = 0.0


class Candidate(BaseModel):
    id: str
    text: str
    base_score: float = 0.0
    features: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0
    memory_similarity: float = 0.0
    visual_support: float = 0.0


class IntentSet(BaseModel):
    candidates: list[Candidate]
    none_fit_available: bool = True


class MemoryHit(BaseModel):
    fragment: str
    confirmed_text: str
    similarity: float
    success_count: int = 0
    failure_count: int = 0


class Reflection(BaseModel):
    failure_type: str
    reason_code: str
    recommendation: str
    confidence: float


class AgentMessage(BaseModel):
    message_type: str
    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# ---- API bodies -------------------------------------------------------------

class StartReq(BaseModel):
    user_id: str


class ProcessReq(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    transcript: str = ""
    pause_intervals: list[float] = Field(default_factory=list)
    frame: Optional[str] = None          # base64 data URL, optional
    scene_hint: list[str] = Field(default_factory=list)  # offline object fallback
    pointing_hint: Optional[str] = None


class FeedbackReq(BaseModel):
    interaction_id: str
    accepted: bool
    confirmed_text: Optional[str] = None   # chosen other candidate, or edited text
    chosen_candidate_id: Optional[str] = None
    none_fit: bool = False
    elapsed_ms: int = 0
