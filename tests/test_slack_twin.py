"""Reliability check for the outward action, against an Arga Labs Slack twin.

Provisions nothing: expects SLACK_BASE_URL / SLACK_BOT_TOKEN in .env (a twin
from `python -m eval.arga_twin` or real Slack). Skips cleanly if not configured.

python -m tests.test_slack_twin
"""
import os

os.environ.setdefault("ECHOLOOP_TRACE", "0")
os.environ.setdefault("ECHOLOOP_TYPESAFE", "0")
os.environ.setdefault("ECHOLOOP_LLM", "0")

from backend.integrations import load_env  # noqa: E402

load_env()

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as m  # noqa: E402
from backend import integrations_slack as slack  # noqa: E402


def test_send_requires_confirmation_and_lands_in_channel():
    if not slack.configured():
        print("skip (SLACK_BOT_TOKEN not set)")
        return
    c = TestClient(m.app)
    r = c.post("/interaction/process", json={"user_id": "twin_test", "transcript": "I need... blue...",
                                             "scene_hint": ["book", "cup"]}).json()
    assert c.post("/api/send", json={"interaction_id": r["interaction_id"]}).status_code == 409, \
        "unconfirmed text must never be sent"
    text = f"I need my blue cup. (twin test {r['interaction_id']})"
    c.post("/interaction/feedback", json={"interaction_id": r["interaction_id"], "accepted": False,
                                          "confirmed_text": text})
    s = c.post("/api/send", json={"interaction_id": r["interaction_id"]}).json()
    assert s["ok"], s
    latest = [x["text"] for x in c.get("/api/send/recent").json()["messages"]]
    assert text in latest, latest


if __name__ == "__main__":
    test_send_requires_confirmation_and_lands_in_channel()
    print("ok  test_send_requires_confirmation_and_lands_in_channel")
