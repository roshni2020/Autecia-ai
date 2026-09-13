"""Agent 3 — Learning. Retrieve memories, build features, rerank, update policy."""
from ...bandit import Policy
from ...embed import cosine, embed
from ...integrations import op, typesafe_judge
from ...schemas import Candidate, IntentSet, MemoryHit, Perception


@op(name="learning_agent.judge")
def judge(p: Perception, memories: list[MemoryHit], texts: list[str]) -> dict | None:
    """Ask System One which candidate the person most likely means.

    Returns {"probabilities": {text: p}, "confidence", "choice", "model"} or None.
    This is a *feature*, not the decision: the bandit learns how much to trust it.
    """
    from typesafe_sdk import Choice  # optional dependency, imported lazily
    resp = typesafe_judge(
        state={
            "utterance": p.transcript,
            "utterance_note": "incomplete speech, kept exactly as spoken",
            "visible_objects": [o.label for o in p.objects],
            "pointing_at": p.gesture.target,
            "sentences_this_person_confirmed_before": [m.confirmed_text for m in memories],
        },
        questions={"meant": Choice(
            instructions="Which candidate sentence is the person most likely trying to "
                         "say? Use `utterance`, `visible_objects`, `pointing_at` and "
                         "`sentences_this_person_confirmed_before`. Pick 'none of these' "
                         "if no candidate fits the evidence.",
            criteria={**{t: None for t in texts}, "none of these": None})})
    if resp is None:
        return None
    a = resp.choices["meant"]
    return {"probabilities": {k: float(v) for k, v in dict(a.probabilities).items()},
            "confidence": float(a.confidence), "choice": a.choice, "model": resp.model}


def features(c: Candidate, p: Perception, memories: list[MemoryHit],
             judgment: dict | None = None) -> dict[str, float]:
    ce = embed(c.text)
    mem_sim, hist = 0.0, 0.0
    for m in memories:
        sim = cosine(ce, embed(m.confirmed_text))
        if sim > mem_sim:
            mem_sim = sim
            total = m.success_count + m.failure_count
            hist = (m.success_count - m.failure_count) / total if total else 0.0
        # context-side similarity: does this memory's *fragment* look like now?
        mem_sim = max(mem_sim, 0.6 * sim * cosine(embed(p.transcript), embed(m.fragment)))
    pointing = p.gesture.confidence if (
        p.gesture.target and p.gesture.target.lower() in c.text.lower()) else 0.0
    judged = judgment["probabilities"].get(c.text, 0.0) if judgment else 0.0
    # Speech evidence: a memory whose *utterance sounded like this one* supports
    # the candidate that matches its confirmed meaning.
    speech = max([0.0] + [m.speech_similarity * cosine(ce, embed(m.confirmed_text))
                          for m in memories if m.speech_similarity > 0])
    return {"base": c.base_score, "memory": round(mem_sim, 4), "visual": c.visual_support,
            "history": round(hist, 4), "pointing": round(pointing, 4),
            "brevity": round(1.0 / (1 + 0.15 * len(c.text.split())), 4),
            "speech": round(speech, 4), "judgment": round(judged, 4), "bias": 1.0}


@op(name="learning_agent.rerank")
def rerank(intents: IntentSet, p: Perception, memories: list[MemoryHit],
           policy: Policy, judgment: dict | None = None) -> list[Candidate]:
    for c in intents.candidates:
        c.features = features(c, p, memories, judgment)
        c.memory_similarity = c.features["memory"]
        c.score = round(policy.score(c.features), 4)
    return sorted(intents.candidates, key=lambda c: -c.score)


def summary(memories: list[MemoryHit], ranked: list[Candidate]) -> str:
    hits = [m for m in memories if m.similarity > 0.15]
    if not hits:
        return "no similar confirmed memory yet — using base ranking"
    return (f"{len(hits)} similar confirmed memory(ies); "
            f"top match {int(hits[0].similarity * 100)}% -> \"{hits[0].confirmed_text}\"")
