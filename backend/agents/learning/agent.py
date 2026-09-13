"""Agent 3 — Learning. Retrieve memories, build features, rerank, update policy."""
from ...bandit import Policy
from ...embed import cosine, embed
from ...integrations import op
from ...schemas import Candidate, IntentSet, MemoryHit, Perception


def features(c: Candidate, p: Perception, memories: list[MemoryHit]) -> dict[str, float]:
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
    return {"base": c.base_score, "memory": round(mem_sim, 4), "visual": c.visual_support,
            "history": round(hist, 4), "pointing": round(pointing, 4),
            "brevity": round(1.0 / (1 + 0.15 * len(c.text.split())), 4), "bias": 1.0}


@op
def rerank(intents: IntentSet, p: Perception, memories: list[MemoryHit],
           policy: Policy) -> list[Candidate]:
    for c in intents.candidates:
        c.features = features(c, p, memories)
        c.memory_similarity = c.features["memory"]
        c.score = round(policy.score(c.features), 4)
    return sorted(intents.candidates, key=lambda c: -c.score)


def summary(memories: list[MemoryHit], ranked: list[Candidate]) -> str:
    hits = [m for m in memories if m.similarity > 0.15]
    if not hits:
        return "no similar confirmed memory yet — using base ranking"
    return (f"{len(hits)} similar confirmed memory(ies); "
            f"top match {int(hits[0].similarity * 100)}% -> \"{hits[0].confirmed_text}\"")
