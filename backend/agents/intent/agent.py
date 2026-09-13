"""Agent 2 — Intent. Generate 2-4 diverse candidate intended messages.

LLM path (Gemini) is validated through TypeSafe AI; on any failure it falls
back to the offline template generator so the demo never stalls.
"""
import re
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

from pydantic import BaseModel, Field

from ...integrations import llm_text, op, parse_llm_json
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
    """Schema the LLM candidate path must satisfy (fail-closed)."""
    candidates: list[str] = Field(min_length=2, max_length=6)


SYSTEM = (
    "You help a person finish a sentence they started but could not complete. "
    "You never guess feelings, moods or diagnoses. You only use the words they said, "
    "the objects visible, where they are pointing, and sentences they confirmed before. "
    "Write short, plain, literal, first-person sentences a person would actually say out loud. "
    "Return only JSON."
)


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


def _dedupe(texts: list[str]) -> list[str]:
    """Drop near-duplicates ('go to bathroom' vs 'go to the bathroom')."""
    seen, uniq = set(), []
    for t in texts:
        k = " ".join(w for w in re.findall(r"[a-z']+", t.lower()) if w not in {"the", "a", "an", "my", "to", "please"})
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


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
    return _dedupe(texts)[:n]


LAST_PROVIDER = {"name": "offline-templates"}


def _llm(p: Perception, memories: list[MemoryHit], n: int) -> list[str] | None:
    objs = ", ".join(o.label for o in p.objects) or "none"
    mem = "; ".join(f'"{m.confirmed_text}"' for m in memories) or "none"
    seeds = "; ".join(f'"{t}"' for t in _offline(p, memories, 3)) or "none"
    facts = ""
    if p.speech is not None:
        s = p.speech
        facts = (f"Speech facts: {s.vad.pause_count} long pause(s), "
                 f"{'fragmented' if s.fragmented else 'complete-sounding'} utterance, "
                 f"{s.filler_count} filler word(s) kept as spoken"
                 f"{', repeated words' if s.repetition_detected else ''}.\n")
    prompt = (
        facts +
        f'Incomplete utterance, exactly as spoken: "{p.transcript}"\n'
        f"Objects visible right now: {objs}\n"
        f"Pointing at: {p.gesture.target or 'nothing'}\n"
        f"Sentences this same person confirmed before: {mem}\n"
        f"Weak template guesses (improve on these): {seeds}\n\n"
        f"Give {n} DIFFERENT complete sentences they most plausibly meant, best first. "
        "Rules: first person; under 10 words each; keep their own words where possible; "
        "only mention objects that are visible or that they confirmed before; if they are "
        "pointing at something, one candidate must be about that thing; one candidate may "
        "be a general request such as asking for help, a break, or to leave. "
        'Return JSON only: {"candidates": ["...", "..."]}')
    raw, provider = llm_text(prompt, SYSTEM)
    LAST_PROVIDER["name"] = provider
    parsed = parse_llm_json(raw, _LLMCandidates)
    return parsed.candidates if parsed else None


@op(name="intent_agent.run")
def run(p: Perception, memories: list[MemoryHit], suggestion_count: int = 3) -> IntentSet:
    n = max(2, min(4, suggestion_count))
    texts = _llm(p, memories, n) or _offline(p, memories, n)
    # Personalisation guarantee: a strong confirmed memory is always on the list.
    for m in memories:
        if m.similarity > 0.3 and m.confirmed_text.lower() not in {t.lower() for t in texts}:
            texts = [m.confirmed_text] + texts
            break
    if len(texts) < 2:
        texts = (texts + ["I need help.", "I want a break."])[:2]

    # Tidy for speech: capital first letter, one terminal mark. Wording is untouched.
    texts = [t.strip()[:1].upper() + t.strip()[1:] for t in texts if t.strip()]
    texts = [t if t[-1] in ".?!" else t + "." for t in texts]
    texts = _dedupe(texts)
    # A suggestion must complete the thought: drop restatements that add no
    # content word beyond what was said ("Can you get the blue thing?" for
    # "can you get... blue thing...").
    said = set(content_words(p.transcript)) | MODIFIERS
    completing = [t for t in texts if set(content_words(t)) - said]
    if len(completing) >= 2:
        texts = completing
    if len(texts) < 2:
        texts = _dedupe(texts + _offline(p, memories, 4))[:max(2, n)]

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
