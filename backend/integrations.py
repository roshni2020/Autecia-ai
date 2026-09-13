"""Optional external services. Every one degrades to an offline path.

W&B Weave  -> tracing/eval        (WANDB_API_KEY)
ElevenLabs -> confirmed speech    (ELEVENLABS_API_KEY, else browser speechSynthesis)
Gemini     -> frame perception    (GEMINI_API_KEY, else scene hints from the UI)
TypeSafe   -> schema-safe LLM out (TYPESAFE_API_KEY, else strict pydantic validation)
CoreWeave  -> batch eval endpoint (COREWEAVE_* , recorded in eval report)
"""
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import httpx


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency); real env vars always win."""
    f = Path(__file__).resolve().parent.parent / path
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env()

# ---- W&B Weave --------------------------------------------------------------
_weave_ready = False


def weave_init(project: str | None = None) -> bool:
    if project is None:
        project = os.getenv("WANDB_PROJECT", "echoloop")
        entity = os.getenv("WANDB_ENTITY")
        if entity and "/" not in project:
            project = f"{entity}/{project}"   # weave wants entity/project
    global _weave_ready
    if _weave_ready or not os.getenv("WANDB_API_KEY"):
        return _weave_ready
    try:
        import weave
        weave.init(project)
        _weave_ready = True
    except Exception as e:  # missing package or offline: keep running
        print(f"[weave] disabled: {e}")
    return _weave_ready


# Per-call tracing is for live/demo use. Batch replays set ECHOLOOP_TRACE=0:
# uploading a span per agent call turns a 5,000-row eval into an hours-long job.
TRACE_CALLS = os.getenv("ECHOLOOP_TRACE", "1") != "0"

if TRACE_CALLS and os.getenv("WANDB_API_KEY"):
    weave_init()  # must run before @op decoration, or those calls are not traced


def op(fn: Callable) -> Callable:
    """@weave.op if weave is live, else identity."""
    try:
        if TRACE_CALLS and os.getenv("WANDB_API_KEY"):
            import weave
            return weave.op()(fn)
    except Exception:
        pass
    return fn


def log_trace(name: str, payload: dict, force: bool = False) -> None:
    """Full-interaction trace record (spec §15). Always written to JSONL too."""
    p = Path(__file__).resolve().parent.parent / "data" / "traces.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"trace": name, **payload}, default=str) + "\n")
    if (TRACE_CALLS or force) and weave_init():
        try:
            import weave
            weave.publish(payload, name=name)
        except Exception:
            pass


# ---- ElevenLabs -------------------------------------------------------------

def tts(text: str) -> bytes | None:
    """Confirmed text only. Returns mp3 bytes, or None -> browser speaks it."""
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        return None
    voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    r = httpx.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                   headers={"xi-api-key": key}, timeout=30,
                   json={"text": text, "model_id": "eleven_turbo_v2_5"})
    r.raise_for_status()
    return r.content


# ---- Gemini (perception only: observable facts, never internal state) -------

VISION_PROMPT = """List only physically observable facts in this frame.
Return JSON: {"objects":[{"label":str,"confidence":0-1}],
"gesture":{"type":"pointing|reaching|none","target":str|null,"confidence":0-1}}
Rules: name at most 6 objects the person could plausibly refer to. Report a
pointing/reaching gesture only if a hand is visibly extended toward something.
Never describe faces, expressions, emotion, mood, or any internal state."""


def gemini_frame(data_url: str) -> dict | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key or not data_url:
        return None
    b64 = data_url.split(",", 1)[-1]
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key}, timeout=20,
            json={"contents": [{"parts": [
                {"text": VISION_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
                "generationConfig": {"responseMimeType": "application/json"}})
        r.raise_for_status()
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(txt)
    except Exception as e:
        print(f"[gemini] frame perception unavailable: {e}")
        return None


def gemini_text(prompt: str) -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key}, timeout=20,
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"responseMimeType": "application/json"}})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"[gemini] text unavailable: {e}")
        return None


# ---- TypeSafe AI ------------------------------------------------------------

def typesafe_validate(raw: str, model) -> Any | None:
    """Contract-check LLM output against a pydantic model.

    Uses the TypeSafe AI service when configured (it repairs off-schema output);
    otherwise strict local validation with the same fail-closed behaviour.
    """
    if raw is None:
        return None
    key, url = os.getenv("TYPESAFE_API_KEY"), os.getenv(
        "TYPESAFE_URL", "https://api.typesafe.ai/v1/validate")
    if key:
        try:
            r = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, timeout=20,
                           json={"schema": model.model_json_schema(), "output": raw,
                                 "repair": True})
            r.raise_for_status()
            body = r.json()
            raw = json.dumps(body.get("repaired") or body.get("output") or body)
        except Exception as e:
            print(f"[typesafe] service unavailable, local validation: {e}")
    try:
        m = re.search(r"[\[{].*[\]}]", raw, re.S)
        return model.model_validate_json(m.group(0) if m else raw)
    except Exception as e:
        print(f"[typesafe] output rejected: {e}")
        return None  # fail closed -> caller falls back to the offline generator


def status() -> dict:
    return {"weave": bool(os.getenv("WANDB_API_KEY")),
            "elevenlabs": bool(os.getenv("ELEVENLABS_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "typesafe": bool(os.getenv("TYPESAFE_API_KEY")),
            "coreweave": bool(os.getenv("COREWEAVE_ENDPOINT"))}
