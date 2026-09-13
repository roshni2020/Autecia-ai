"""EchoLoop API. Run: uvicorn backend.main:app --reload"""
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import bandit, memory
from .integrations import status, tts, weave_init
from .pipeline import feedback as run_feedback
from .pipeline import process as run_process
from .schemas import SUPPORT_MODES, FeedbackReq, ProcessReq, StartReq, SupportProfile

app = FastAPI(title="EchoLoop")
con = memory.connect()
weave_init()

QUICK_PHRASES = ["I need a break.", "It is too loud.", "I want to leave.",
                 "Can you wait a moment?", "I need help."]


@app.get("/api/status")
def api_status():
    return {"integrations": status(), "support_modes": SUPPORT_MODES}


@app.get("/api/profile/{user_id}")
def get_profile(user_id: str):
    p = memory.get_profile(con, user_id)
    return {"profile": p.model_dump(),
            "quick_phrases": QUICK_PHRASES if p.quick_phrases_enabled else [],
            "policy": bandit.load(con, user_id, p.support_mode).as_dict()}


@app.post("/api/profile/{user_id}")
def put_profile(user_id: str, profile: SupportProfile):
    memory.set_profile(con, user_id, profile)
    return {"profile": profile.model_dump()}


class ModeReq(BaseModel):
    support_mode: str


@app.post("/api/onboarding/{user_id}")
def onboarding(user_id: str, req: ModeReq):
    """User-selected support preference. Not a diagnosis, never inferred."""
    if req.support_mode not in SUPPORT_MODES:
        raise HTTPException(400, "unknown support mode")
    p = SupportProfile.for_mode(req.support_mode)
    memory.set_profile(con, user_id, p)
    return {"profile": p.model_dump(),
            "quick_phrases": QUICK_PHRASES if p.quick_phrases_enabled else []}


@app.post("/interaction/start")
def start(req: StartReq):
    return {"session_id": uuid.uuid4().hex[:8], "interaction_id": uuid.uuid4().hex[:12]}


@app.post("/interaction/process")
def process(req: ProcessReq):
    if not req.transcript.strip():
        raise HTTPException(400, "empty transcript")
    return run_process(con, req)


@app.post("/interaction/feedback")
def feedback(req: FeedbackReq):
    try:
        return run_feedback(con, req)
    except KeyError:
        raise HTTPException(404, "unknown interaction_id")


class SpeakReq(BaseModel):
    text: str
    interaction_id: str | None = None


@app.post("/api/speak")
def speak(req: SpeakReq):
    """Speaks CONFIRMED text only — the caller must have a confirmed interaction."""
    if req.interaction_id:
        row = memory.get_interaction(con, req.interaction_id)
        if row is None or not row["feedback"]:
            raise HTTPException(409, "text is not confirmed yet")
    audio = tts(req.text)
    if audio is None:
        return {"fallback": "browser", "text": req.text}  # client speechSynthesis
    return Response(audio, media_type="audio/mpeg")


@app.get("/api/history/{user_id}")
def history(user_id: str):
    p = memory.get_profile(con, user_id)
    return {"history": memory.history(con, user_id),
            "policy": bandit.load(con, user_id, p.support_mode).as_dict(),
            "memories": [dict(r) for r in con.execute(
                "SELECT fragment,visual_summary,confirmed_text,success_count,failure_count"
                " FROM memories WHERE user_id=? ORDER BY timestamp DESC", (user_id,))]}


class ForgetReq(BaseModel):
    confirmed_text: str


@app.post("/api/forget/{user_id}")
def forget(user_id: str, req: ForgetReq):
    return {"deleted": memory.forget(con, user_id, req.confirmed_text)}


@app.get("/api/evaluation")
def evaluation():
    """Latest offline evaluation report (run: python -m eval.run_eval)."""
    import json
    p = Path(__file__).resolve().parent.parent / "data" / "evaluation_report.json"
    if not p.exists():
        raise HTTPException(404, "no report yet — run: python -m eval.run_eval")
    return json.loads(p.read_text(encoding="utf-8"))


# ponytail: one static HTML page, no npm/build step. Move to a React build only if
# the UI outgrows a single file.
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
