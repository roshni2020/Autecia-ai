"""Agent 1 — Perception. Audio + optional frame -> observable context only.

Never infers diagnosis, emotion, anxiety, or sensory state (spec §3, §9).
"""
import re

from ...integrations import gemini_frame, op
from ...schemas import DetectedObject, Gesture, Perception, ProcessReq

FILLERS = {"um", "uh", "er", "hmm", "like", "eh"}
INCOMPLETE_TAIL = re.compile(r"(\.\.\.|\bthat\b|\bthe\b|\ba\b|\bsome\b)\s*$", re.I)


def _repetition(words: list[str]) -> bool:
    return any(a == b for a, b in zip(words, words[1:]) if a not in FILLERS)


@op
def run(req: ProcessReq, camera_enabled: bool) -> Perception:
    words = re.findall(r"[a-zA-Z']+", req.transcript.lower())
    objects: list[DetectedObject] = []
    gesture = Gesture()

    vision = gemini_frame(req.frame) if (camera_enabled and req.frame) else None
    if vision:
        objects = [DetectedObject(**o) for o in vision.get("objects", [])[:6]]
        g = vision.get("gesture") or {}
        if g.get("type") in ("pointing", "reaching"):
            gesture = Gesture(**g)
    elif camera_enabled and req.scene_hint:
        # Offline path: objects declared by the UI scene panel (demo/eval mode).
        objects = [DetectedObject(label=o, confidence=0.9) for o in req.scene_hint[:6]]
        if req.pointing_hint:
            gesture = Gesture(type="pointing", target=req.pointing_hint, confidence=0.8)

    fragment_len = len([w for w in words if w not in FILLERS])
    conf = min(0.95, 0.25 + 0.08 * fragment_len + (0.2 if objects else 0.0)
               + 0.15 * gesture.confidence)

    return Perception(
        transcript=req.transcript,
        pause_intervals=req.pause_intervals,
        repetition_detected=_repetition(words),
        objects=objects,
        gesture=gesture,
        camera_enabled=bool(camera_enabled and (vision or req.scene_hint)),
        perception_confidence=round(conf, 2),
    )


def summary(p: Perception) -> str:
    bits = []
    if p.pause_intervals or INCOMPLETE_TAIL.search(p.transcript) or len(p.transcript.split()) < 5:
        bits.append("incomplete speech detected")
    if p.repetition_detected:
        bits.append("repetition detected")
    if p.objects:
        bits.append(f"{len(p.objects)} relevant object(s) in view")
    if p.gesture.type != "none" and p.gesture.target:
        bits.append(f"{p.gesture.type} toward {p.gesture.target}")
    return ", ".join(bits) or "speech only"


def visual_summary(p: Perception) -> str:
    if not p.objects:
        return "no camera context"
    seen = ", ".join(o.label for o in p.objects)
    if p.gesture.target:
        return f"pointed toward {p.gesture.target} (visible: {seen})"
    return f"visible: {seen}"
