"""Agent 4 — Reflection. Label why the shown suggestion succeeded or failed.

Never overrides the user's confirmed label; it only recommends (spec §6).
"""
from ...integrations import op
from ...schemas import Perception, Reflection

FAILURE_TYPES = ["ASR_ERROR", "MISSING_OBJECT", "WRONG_POINTING_TARGET", "MISSING_CANDIDATE",
                 "RANKING_ERROR", "STALE_MEMORY", "CONFLICTING_CONTEXT", "LOW_CONFIDENCE",
                 "FEEDBACK_AMBIGUITY", "SUCCESS"]


@op
def run(prediction: str, confirmed_text: str | None, accepted: bool, none_fit: bool,
        perception: Perception, candidates: list[dict], memory_matches: list[dict]) -> Reflection:
    texts = [c["text"].lower() for c in candidates]
    conf = (confirmed_text or "").strip()

    if accepted:
        agree = bool(memory_matches) and bool(perception.objects)
        return Reflection(
            failure_type="SUCCESS",
            reason_code="VISUAL_AND_MEMORY_AGREE" if agree else "TOP_CANDIDATE_CONFIRMED",
            recommendation="reinforce current feature weights for similar contexts",
            confidence=round(min(0.95, 0.6 + 0.35 * perception.perception_confidence), 2))

    if none_fit or (conf and conf.lower() not in texts):
        if not perception.objects and perception.camera_enabled is False:
            return Reflection(failure_type="MISSING_OBJECT",
                              reason_code="NO_VISUAL_CONTEXT_FOR_REFERENT",
                              recommendation="enable camera or ask a grounding question first",
                              confidence=0.7)
        return Reflection(failure_type="MISSING_CANDIDATE",
                          reason_code="CORRECT_MEANING_NEVER_GENERATED",
                          recommendation="widen intent generation for this fragment shape; "
                                         "add the confirmed sentence to the phrase bank",
                          confidence=0.78)

    # Correct meaning was on the list but not ranked first.
    if perception.gesture.target and perception.gesture.target.lower() not in prediction.lower() \
            and perception.gesture.target.lower() in conf.lower():
        return Reflection(failure_type="WRONG_POINTING_TARGET",
                          reason_code="POINTING_UNDERWEIGHTED",
                          recommendation="increase pointing contribution for similar contexts",
                          confidence=0.8)
    if perception.perception_confidence < 0.5:
        return Reflection(failure_type="LOW_CONFIDENCE",
                          reason_code="WEAK_PERCEPTION_SIGNAL",
                          recommendation="ask one short clarifying question before ranking",
                          confidence=0.65)
    if memory_matches and memory_matches[0]["confirmed_text"].lower() == prediction.lower():
        return Reflection(failure_type="STALE_MEMORY",
                          reason_code="OLD_PREFERENCE_DOMINATED",
                          recommendation="decay memory weight for outdated confirmations",
                          confidence=0.72)
    return Reflection(failure_type="RANKING_ERROR",
                      reason_code="VISUAL_AND_MEMORY_UNDERWEIGHTED",
                      recommendation="increase user-memory and pointing contribution "
                                     "for similar contexts",
                      confidence=0.82)
