"""Agent 4 — Reflection. Label why the shown suggestion succeeded or failed.

Never overrides the user's confirmed label; it only recommends (spec §6).
"""
from ...integrations import op, typesafe_judge
from ...schemas import Perception, Reflection

FAILURE_TYPES = ["ASR_ERROR", "MISSING_OBJECT", "WRONG_POINTING_TARGET", "MISSING_CANDIDATE",
                 "RANKING_ERROR", "STALE_MEMORY", "CONFLICTING_CONTEXT", "LOW_CONFIDENCE",
                 "FEEDBACK_AMBIGUITY", "SUCCESS"]


FAILURE_CRITERIA = {
    "ASR_ERROR": "the transcript itself looks wrong or garbled, so no candidate could fit",
    "MISSING_OBJECT": "the thing the person meant was not among the visible objects",
    "WRONG_POINTING_TARGET": "the pointing target disagrees with what was confirmed",
    "MISSING_CANDIDATE": "the confirmed sentence was never among the candidates",
    "RANKING_ERROR": "the confirmed sentence was a candidate but was not ranked first",
    "STALE_MEMORY": "an older confirmed memory pushed the wrong candidate to the top",
    "CONFLICTING_CONTEXT": "visual evidence and memory evidence pointed different ways",
    "LOW_CONFIDENCE": "the evidence was too weak to support any candidate strongly",
    "FEEDBACK_AMBIGUITY": "the feedback does not make clear what the person meant",
}
RECOMMEND = {
    "ASR_ERROR": "show the transcript for correction before ranking",
    "MISSING_OBJECT": "enable camera or ask a grounding question first",
    "WRONG_POINTING_TARGET": "increase pointing contribution for similar contexts",
    "MISSING_CANDIDATE": "widen intent generation for this fragment shape; "
                         "add the confirmed sentence to the phrase bank",
    "RANKING_ERROR": "increase user-memory and pointing contribution for similar contexts",
    "STALE_MEMORY": "decay memory weight for outdated confirmations",
    "CONFLICTING_CONTEXT": "ask one short clarifying question when visual and memory disagree",
    "LOW_CONFIDENCE": "ask one short clarifying question before ranking",
    "FEEDBACK_AMBIGUITY": "offer an explicit edit box after a rejection",
}


def _classify_with_typesafe(prediction, conf, perception, candidates, memory_matches):
    try:
        from typesafe_sdk import Choice
    except ImportError:
        return None
    resp = typesafe_judge(
        state={"utterance": perception.transcript,
               "shown_suggestion": prediction,
               "confirmed_by_user": conf or "(nothing - user said none fit)",
               "all_candidates": [c["text"] for c in candidates],
               "visible_objects": [o.label for o in perception.objects],
               "pointing_at": perception.gesture.target,
               "perception_confidence": perception.perception_confidence,
               "similar_past_confirmations": [m["confirmed_text"] for m in memory_matches]},
        questions={"failure": Choice(
            instructions="Why did `shown_suggestion` fail to match `confirmed_by_user`? "
                         "Pick the single most likely cause.",
            criteria=FAILURE_CRITERIA)})
    if resp is None:
        return None
    a = resp.choices["failure"]
    return Reflection(failure_type=a.choice, reason_code="SYSTEM_ONE_JUDGMENT",
                      recommendation=RECOMMEND.get(a.choice, "review this interaction"),
                      confidence=round(float(a.confidence), 2))


@op(name="reflection_agent.run")
def run(prediction: str, confirmed_text: str | None, accepted: bool, none_fit: bool,
        perception: Perception, candidates: list[dict], memory_matches: list[dict]) -> Reflection:
    texts = [c["text"].lower() for c in candidates]
    conf = (confirmed_text or "").strip()

    if not accepted:
        judged = _classify_with_typesafe(prediction, conf, perception, candidates, memory_matches)
        if judged:
            return judged

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
