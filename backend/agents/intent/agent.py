"""Agent 2 — Intent. Generate 2-4 diverse candidate intended messages.

LLM path (Gemini) is validated through TypeSafe AI; on any failure it falls
back to the offline template generator so the demo never stalls.
"""
import re
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

from pydantic import BaseModel, Field

from ...integrations import gemini_text, op, typesafe_validate
from ...schemas import Candidate, IntentSet, MemoryHit, Perception

BANK = Path(__file__).resolve().parents[3] / "data" / "phrasebank.txt"
STOP = {"i", "a", "the", "um", "uh", "that", "this", "need", "want", "can", "you",
        "get", "my", "me", "is", "it", "to", "of", "and", "some", "please", "er"}
MODIFIERS = {"blue", "red", "green", "black", "white", "yellow", "big", "small",
             "other", "thing", "things", "one", "loud", "much", "many"}
CARRIERS = [
    (r"\b(need|want|get|give|bring|find)\b", "I need my {ref}."),
    (r"\b(go|leave|out|home)\b", "I want to go {ref}."),
    (r"\b(too|much|loud|many|noisy)\b", "It is too {ref} for me."),
    (r"\b(help|stuck|cannot|can't)\b", "I need help with {ref}."),
]


class _LLMCandidates(BaseModel):
    """Schema TypeSafe AI enforces on the model output."""
    candidates: list[str] = Field(min_length=2, max_length=4)


@lru_cache(maxsize=1)
def phrase_bank() -> list[str]:
    if not BANK.exists():
        return []
    return [ln.strip() for ln in BANK.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 1]


def _referents(p: Perception, memories: list[MemoryHit], cues: list[str]) -> list[tuple[str, float]]:
    """Candidate things the fragment might be about, with visual support."""
    out: dict[str, float] = {}
    for o in p.objects:
        support = o.confidence
        if p.gesture.target and o.label == p.gesture.target:
            support = min(1.0, support + 0.2 + 0.3 * p.gesture.confidence)
        if any(c in o.label.lower() for c in cues):
            support = min(1.0, support + 0.15)
        out[o.label] = max(out.get(o.label, 0), support)
    for m in memories:  # things this user has confirmed before
        for w in content_words(m.confirmed_text):
            out.setdefault(w, 0.0)
    for c in cues:
        if c not in MODIFIERS:  # "blue" is a descriptor, not a thing to ask for
            out.setdefault(c, 0.0)
    return sorted(out.items(), key=lambda kv: -kv[1])[:4]


def _carrier(fragment: str) -> str:
    for pat, tmpl in CARRIERS:
        if re.search(pat, fragment, re.I):
            return tmpl
    return "I need my {ref}."


def _offline(p: Perception, memories: list[MemoryHit], n: int) -> list[str]:
    cues = content_words(p.transcript)
    tmpl = _carrier(p.transcript)
    from_scene = [tmpl.format(ref=ref) for ref, _ in _referents(p, memories, cues)]
    from_memory = [m.confirmed_text for m in memories if m.similarity > 0.05]
    # Interleave: this user's own confirmed wordings compete for the top slots
    # instead of being appended after every scene guess (they are the whole point).
    texts: list[str] = []
    for a, b in zip_longest(from_scene, from_memory):
        if a:
            texts.append(a)
        if b:
            texts.append(b)
    descr = [c for c in cues if c not in {"thing", "things", "one", "other"}]
    texts.append(f"I need help finding something {descr[-1]}." if descr
                 else "I need help finding something.")
    for phrase in phrase_bank():  # functional fallbacks that share a cue word
        if any(c in phrase.lower() for c in cues):
            texts.append(phrase)
    seen, uniq = set(), []
    for t in texts:
        k = t.lower().strip()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:n]


def _llm(p: Perception, memories: list[MemoryHit], n: int) -> list[str] | None:
    objs = ", ".join(f"{o.label}" for o in p.objects) or "none"
    mem = "; ".join(m.confirmed_text for m in memories) or "none"
    prompt = (
        f'A person said this incomplete utterance: "{p.transcript}".\n'
        f"Objects visible: {objs}. Pointing at: {p.gesture.target or 'nothing'}.\n"
        f"Sentences this same person confirmed before: {mem}.\n"
        f"Give {n} short, plainly-worded, DIFFERENT complete sentences they might have "
        "meant, first person. Use only the objects/actions given — invent nothing. "
        'Return JSON: {"candidates": ["...", "..."]}')
    parsed = typesafe_validate(gemini_text(prompt), _LLMCandidates)
    return parsed.candidates if parsed else None


@op
def run(p: Perception, memories: list[MemoryHit], suggestion_count: int = 3) -> IntentSet:
    n = max(2, min(4, suggestion_count))
    texts = _llm(p, memories, n) or _offline(p, memories, n)
    if len(texts) < 2:
        texts = (texts + ["I need help.", "I want a break."])[:2]

    cues = set(content_words(p.transcript))
    cands = []
    for i, t in enumerate(texts[:n]):
        overlap = len(cues & set(content_words(t))) / max(1, len(cues))
        vis = max([0.0] + [o.confidence + (0.3 if o.label == p.gesture.target else 0.0)
                           for o in p.objects if o.label.lower() in t.lower()])
        cands.append(Candidate(id=f"intent_{chr(97 + i)}", text=t,
                               base_score=round(0.5 * overlap + 0.5 * min(1.0, vis), 4),
                               visual_support=round(min(1.0, vis), 4)))
    total = sum(c.base_score for c in cands) or 1.0
    for c in cands:
        c.base_score = round(c.base_score / total, 4)
    return IntentSet(candidates=cands, none_fit_available=True)
